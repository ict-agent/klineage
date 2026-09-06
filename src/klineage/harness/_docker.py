"""Private Docker transport for CUDA problem evaluation."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from klineage.harness.eval import ValidationResult
from klineage.kernel import Kernel

_SERVICE = "dev"
_GPU = 0
_CONTAINER_GPU = 0
_TIMEOUT_SECONDS = 900.0
_CLEANUP_TIMEOUT_SECONDS = 10.0
_SEED = 20260903
_WARMUP = 50
_REPEAT = 10
_TRIALS = 2
_COLD_L2 = True
_MAX_JOBS = 4
_PIDS_LIMIT = 512
_SHM_SIZE = "1g"
_TMPFS = "/tmp:rw,nosuid,nodev,size=1g"
_TMP_EXTENSIONS_DIR = Path("/tmp/torch-extensions")
_FLASHINFER_WORKSPACE_BASE = "/tmp"
_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{12,64}$")
_CONTAINER_PYTHON = Path("/opt/klineage/venv/bin/python")
_WORKER_PATH = Path("src/klineage/harness/_cuda_worker.py")


class _DockerError(RuntimeError):
    """Docker could not prepare or run the CUDA evaluator."""


class _DockerRuntime:
    """Run CUDA workers in isolated containers from a prepared image."""

    def __init__(
        self,
        *,
        root: Path,
        docker: str,
        image: str,
        mount: Path,
        gpu: int = _GPU,
        timeout: float = _TIMEOUT_SECONDS,
    ) -> None:
        self._root = root
        self._docker = docker
        self._image = image
        self._mount = mount
        self._gpu = gpu
        self._timeout = timeout

    def inspect(
        self,
        problem_path: Path,
        *,
        log_dir: Path,
    ) -> Mapping[str, Any]:
        """Load a problem in the CUDA runtime and return its contract."""

        logs = _log_dir(log_dir)
        request = {
            "operation": "inspect",
            "problem_path": str(self._container_path(problem_path)),
            "seed": _SEED,
        }
        payload = self._invoke(
            request,
            logs,
            "inspect",
            read_paths=(problem_path,),
        )
        if not isinstance(payload, Mapping):
            raise _DockerError("CUDA worker inspect result must be a JSON object")
        if "error" in payload:
            raise _DockerError(str(payload["error"]))
        _write_json(logs / "inspect-result.json", payload)

        return dict(payload)

    def evaluator(
        self,
        problem_path: Path,
        *,
        include_paths: Sequence[Path],
        build_root: Path,
        log_dir: Path,
        read_paths: Sequence[Path] = (),
    ) -> _DockerEvaluator:
        """Create an evaluator bound to one problem and build workspace."""

        problem = self._container_path(problem_path)
        includes = tuple(
            self._container_path(path) for path in include_paths
        )
        host_build_root = Path(build_root).expanduser().absolute()
        host_build_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        build = self._container_path(host_build_root)

        config = {
            "image_id": self._image,
            "problem_path": str(problem),
            "seed": _SEED,
            "timing": {
                "warmup": _WARMUP,
                "repeat": _REPEAT,
                "trials": _TRIALS,
                "cold_l2": _COLD_L2,
            },
            "include_paths": [str(path) for path in includes],
            "build_root": str(build),
        }
        return _DockerEvaluator(
            self,
            config,
            _log_dir(log_dir),
            read_paths,
        )

    def _container_path(self, path: Path) -> Path:
        host_path = Path(path).expanduser().resolve(strict=True)
        try:
            relative = host_path.relative_to(self._root)
        except ValueError as exc:
            raise ValueError(
                f"path is outside the Docker project mount: {host_path}"
            ) from exc

        return self._mount / relative

    def _kernel(self, kernel: Kernel) -> dict[str, Any]:
        payload = kernel.to_dict()
        # The worker measures fresh evidence; prior timing samples stay on the host.
        payload.pop("validation")
        artifact = kernel.artifact_path
        if artifact is not None:
            payload["artifact_path"] = str(self._container_path(artifact))
        return payload

    def _host_path(self, path: Path) -> Path:
        try:
            relative = path.relative_to(self._mount)
        except ValueError as exc:
            raise _DockerError("container path is outside the project mount") from exc
        host = (self._root / relative).resolve(strict=True)
        if not host.is_relative_to(self._root):
            raise _DockerError("container path escaped the project mount")
        return host

    def _invoke(
        self,
        request: Mapping[str, Any],
        log_dir: Path,
        stem: str,
        *,
        read_paths: Sequence[Path] = (),
    ) -> Any:
        request_path = log_dir / f"{stem}-request.json"
        stdout_path = log_dir / f"{stem}-stdout.log"
        stderr_path = log_dir / f"{stem}-stderr.log"
        process_path = log_dir / f"{stem}-process.json"
        cid_path = log_dir / f".{stem}.cid"
        cid_path.unlink(missing_ok=True)
        _write_json(request_path, request)

        worker = self._mount / _WORKER_PATH
        read_mounts = self._read_mounts(read_paths)
        build_root = _build_root(request, self._mount)
        build_mount: tuple[str, ...] = ()
        if build_root is not None:
            build_mount = (
                "--mount",
                _mount_spec(self._host_path(build_root), build_root),
            )
        extensions = build_root or _TMP_EXTENSIONS_DIR
        command = (
            self._docker,
            "run",
            "--rm",
            "--interactive",
            "--cidfile",
            str(cid_path),
            "--gpus",
            f"device={self._gpu}",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(_PIDS_LIMIT),
            "--shm-size",
            _SHM_SIZE,
            "--tmpfs",
            _TMPFS,
            "--workdir",
            str(self._mount),
            *read_mounts,
            *build_mount,
            "--env",
            f"CUDA_VISIBLE_DEVICES={_CONTAINER_GPU}",
            "--env",
            "CUDA_HOME=/usr/local/cuda",
            "--env",
            f"MAX_JOBS={_MAX_JOBS}",
            "--env",
            f"PYTHONPATH={self._mount / 'src'}",
            "--env",
            "PYTHONUNBUFFERED=1",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "TMPDIR=/tmp",
            "--env",
            "TMP=/tmp",
            "--env",
            "TEMP=/tmp",
            "--env",
            "CUDA_CACHE_PATH=/tmp",
            "--env",
            f"FLASHINFER_WORKSPACE_BASE={_FLASHINFER_WORKSPACE_BASE}",
            "--env",
            f"TORCH_EXTENSIONS_DIR={extensions}",
            "--entrypoint",
            str(_CONTAINER_PYTHON),
            self._image,
            str(worker),
        )
        started_at = _now()
        try:
            try:
                completed = subprocess.run(
                    command,
                    input=json.dumps(request, ensure_ascii=False, sort_keys=True),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self._timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                stdout_path.write_text(_text(exc.stdout), encoding="utf-8")
                stderr_path.write_text(_text(exc.stderr), encoding="utf-8")
                _write_json(
                    process_path,
                    {
                        "started_at": started_at,
                        "finished_at": _now(),
                        "timed_out": True,
                        "timeout_seconds": self._timeout,
                    },
                )
                raise _DockerError(
                    f"CUDA worker timed out after {self._timeout:g}s; "
                    f"see {stderr_path}"
                ) from exc
        finally:
            _remove_container(self._docker, cid_path)

        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        _write_json(
            process_path,
            {
                "started_at": started_at,
                "finished_at": _now(),
                "returncode": completed.returncode,
                "timed_out": False,
            },
        )
        if completed.returncode != 0:
            tail = completed.stderr.strip()[-2000:]
            raise _DockerError(
                "CUDA worker exited with status "
                f"{completed.returncode}: {tail or 'no stderr'}"
            )

        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            tail = completed.stdout.strip()[-2000:]
            raise _DockerError(
                f"CUDA worker did not emit JSON; stdout tail: {tail!r}"
            ) from exc

    def _read_mounts(self, read_paths: Sequence[Path]) -> tuple[str, ...]:
        paths = _coalesce_paths((self._root / "src", *read_paths), self._root)
        mounts: list[str] = []
        for path in paths:
            mounts.extend(
                (
                    "--mount",
                    _mount_spec(
                        path,
                        self._container_path(path),
                        read_only=True,
                    ),
                )
            )
        return tuple(mounts)


class _DockerEvaluator:
    """Evaluate kernels through a prepared Docker runtime."""

    def __init__(
        self,
        runtime: _DockerRuntime,
        config: Mapping[str, Any],
        log_dir: Path,
        read_paths: Sequence[Path],
    ) -> None:
        self._runtime = runtime
        self._config = dict(config)
        self._log_dir = log_dir
        self._read_paths = tuple(read_paths)
        self._sequence = 0

    def evaluate(
        self,
        kernel: Kernel,
        *,
        reference: Kernel | None = None,
    ) -> ValidationResult:
        self._sequence += 1
        stem = f"eval-{self._sequence:04d}"
        request = {
            "operation": "evaluate",
            "kernel": self._runtime._kernel(kernel),
            "reference": (
                self._runtime._kernel(reference) if reference is not None else None
            ),
            "config": self._config,
        }
        read_paths = list(self._read_paths)
        for item in (kernel, reference):
            if item is not None and item.artifact_path is not None:
                read_paths.append(item.artifact_path)
        payload = self._runtime._invoke(
            request,
            self._log_dir,
            stem,
            read_paths=read_paths,
        )
        if not isinstance(payload, Mapping):
            raise _DockerError("CUDA worker evaluation result must be a JSON object")

        try:
            result = ValidationResult.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise _DockerError(
                "CUDA worker did not emit a ValidationResult JSON object"
            ) from exc
        _write_json(self._log_dir / f"{stem}-result.json", result.to_dict())
        return result


def _prepare(project_root: Path) -> _DockerRuntime:
    """Start the Compose evaluator service and resolve its project mount."""

    root = Path(project_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Docker project root is not a directory: {root}")
    compose_file = root / "compose.yaml"
    if not compose_file.is_file():
        raise FileNotFoundError(f"Docker Compose file is missing: {compose_file}")

    docker = shutil.which("docker")
    if docker is None:
        raise FileNotFoundError("Docker executable was not found")
    compose = (
        docker,
        "compose",
        "--project-directory",
        str(root),
        "--file",
        str(compose_file),
    )
    environment = os.environ.copy()
    environment.setdefault("KLINEAGE_UID", str(os.getuid()))
    environment.setdefault("KLINEAGE_GID", str(os.getgid()))
    _checked(
        (*compose, "up", "--detach", _SERVICE),
        timeout=_TIMEOUT_SECONDS,
        environment=environment,
        label="Docker Compose service startup",
    )
    container = _checked(
        (*compose, "ps", "--quiet", _SERVICE),
        timeout=30.0,
        environment=environment,
        label="Docker Compose container lookup",
    ).stdout.strip()
    if not container or "\n" in container:
        raise _DockerError(
            f"expected one running {_SERVICE!r} container, got {container!r}"
        )

    inspection = _checked(
        (docker, "inspect", container),
        timeout=30.0,
        environment=environment,
        label="Docker container inspection",
    )
    mount, image = _container_config(inspection.stdout, root)
    runtime = _DockerRuntime(
        root=root,
        docker=docker,
        image=image,
        mount=mount,
    )
    runtime._container_path(root / _WORKER_PATH)
    return runtime


def _container_config(value: str, root: Path) -> tuple[Path, str]:
    try:
        inspections = json.loads(value)
        inspection = inspections[0]
        mounts = inspection["Mounts"]
        running = inspection["State"]["Running"]
        image = inspection["Image"]
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise _DockerError("Docker inspect returned an invalid container record") from exc
    if running is not True:
        raise _DockerError("Docker Compose evaluator container is not running")
    if not isinstance(mounts, Sequence) or isinstance(mounts, (str, bytes)):
        raise _DockerError("Docker inspect returned invalid container mounts")
    if not isinstance(image, str) or not image:
        raise _DockerError("Docker inspect returned an invalid image")

    for mount in mounts:
        if not isinstance(mount, Mapping):
            continue
        source = mount.get("Source")
        destination = mount.get("Destination")
        if not isinstance(source, str) or not isinstance(destination, str):
            continue
        mounted_source = Path(source).resolve()
        if mounted_source == root:
            return Path(destination), image

    raise _DockerError(f"Docker container does not mount project root {root}")


def _checked(
    command: tuple[str, ...],
    *,
    timeout: float,
    environment: Mapping[str, str],
    label: str,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise _DockerError(f"{label} timed out after {timeout:g}s") from exc
    if completed.returncode == 0:
        return completed

    reason = completed.stderr.strip() or completed.stdout.strip() or "no output"
    raise _DockerError(f"{label} failed: {reason[-2000:]}")


def _build_root(request: Mapping[str, Any], mount: Path) -> Path | None:
    config = request.get("config")
    if not isinstance(config, Mapping):
        return None
    value = config.get("build_root")
    if not isinstance(value, str):
        raise _DockerError("CUDA worker build_root must be a path")
    path = Path(value)
    if not path.is_absolute() or not path.is_relative_to(mount):
        raise _DockerError("CUDA worker build_root is outside the project mount")
    return path


def _mount_spec(source: Path, target: Path, *, read_only: bool = False) -> str:
    source_value = str(source)
    target_value = str(target)
    if "," in source_value or "," in target_value:
        raise _DockerError("Docker mount paths cannot contain commas")
    options = f"type=bind,source={source_value},target={target_value}"
    return options + ",readonly" if read_only else options


def _coalesce_paths(paths: Sequence[Path], root: Path) -> tuple[Path, ...]:
    resolved: list[Path] = []
    for value in paths:
        path = Path(value).expanduser().resolve(strict=True)
        if path == root or not path.is_relative_to(root):
            raise _DockerError("CUDA read path is outside the project inputs")
        if any(path.is_relative_to(parent) for parent in resolved):
            continue
        resolved = [parent for parent in resolved if not parent.is_relative_to(path)]
        resolved.append(path)
    return tuple(resolved)


def _remove_container(docker: str, cid_path: Path) -> None:
    try:
        container = cid_path.read_text(encoding="utf-8").strip()
    except OSError:
        return
    finally:
        cid_path.unlink(missing_ok=True)
    if not _CONTAINER_ID_RE.fullmatch(container):
        return

    try:
        subprocess.run(
            (docker, "rm", "--force", container),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_CLEANUP_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def _log_dir(path: Path) -> Path:
    directory = Path(path).expanduser().absolute()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return directory


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _now() -> str:
    return datetime.now(UTC).isoformat()

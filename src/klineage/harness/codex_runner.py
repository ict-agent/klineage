"""Run ``codex exec`` in a strict filesystem sandbox."""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

ReasoningEffort = Literal[
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
]

_PROFILE_NAME = "klineage_sandbox"
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REASONING_EFFORTS = frozenset(
    ("none", "minimal", "low", "medium", "high", "xhigh", "max")
)
_BWRAP_FLAGS = ("--as-pid-1", "--argv0", "--perms")
_BWRAP_PROBE_TIMEOUT_SECONDS = 2
_TERMINATION_GRACE_SECONDS = 5
_STDERR_SUMMARY_CHARS = 1000


class CodexRunnerError(RuntimeError):
    """A Codex launch or execution failure."""

    def __init__(self, message: str, *, trace_path: Path | None = None) -> None:
        self.trace_path = trace_path
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class CodexRunResult:
    """The output needed by orchestration code."""

    final_message: str
    trace_path: Path


class CodexRunner:
    """Run Codex with no approvals, no network, and one writable workspace."""

    def __init__(
        self,
        sandbox_dir: str | os.PathLike[str],
        *,
        codex_bin: str | os.PathLike[str] = "codex",
        reasoning_effort: ReasoningEffort | None = "xhigh",
        timeout: float | None = None,
        state_dir: str | os.PathLike[str] = ".klineage",
        bwrap_bin: str | os.PathLike[str] | None = None,
        read_roots: Sequence[str | os.PathLike[str]] = (),
    ) -> None:
        if sys.platform != "linux":
            raise CodexRunnerError("CodexRunner requires Linux")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if reasoning_effort not in _REASONING_EFFORTS and reasoning_effort is not None:
            choices = ", ".join(sorted(_REASONING_EFFORTS))
            raise ValueError(f"reasoning_effort must be one of: {choices}, or None")

        sandbox = Path(sandbox_dir).expanduser().absolute()
        sandbox.mkdir(parents=True, exist_ok=True)
        if not sandbox.is_dir():
            raise NotADirectoryError(sandbox)
        sandbox = sandbox.resolve(strict=True)
        if sandbox.parent == sandbox:
            raise ValueError("the filesystem root cannot be used as a sandbox")
        resolved_read_roots = _resolve_read_roots(sandbox, read_roots)

        state_root = _prepare_internal_dir(sandbox, state_dir, "state_dir")
        trace_root = _prepare_internal_dir(state_root, "codex-runs", "trace path")
        temp_root = _prepare_internal_dir(state_root, "tmp", "temporary path")
        codex_program = _resolve_codex_program(codex_bin)
        bwrap_program = _resolve_bwrap_program(codex_program, bwrap_bin)
        launcher_dir = Path(
            tempfile.mkdtemp(prefix="klineage-codex-")
        ).resolve(strict=True)
        if launcher_dir.is_relative_to(sandbox):
            shutil.rmtree(launcher_dir, ignore_errors=True)
            raise CodexRunnerError(
                "the system temporary directory must be outside sandbox_dir"
            )

        self.sandbox_dir = sandbox
        self.state_dir = state_root
        self._trace_root = trace_root
        self._temp_root = temp_root
        self._timeout = timeout
        self._reasoning_effort = reasoning_effort
        self._codex_program = codex_program
        self._bwrap_program = bwrap_program
        self._read_roots = resolved_read_roots
        self._launcher_dir = launcher_dir
        self._launcher_finalizer = weakref.finalize(
            self,
            shutil.rmtree,
            launcher_dir,
            ignore_errors=True,
        )
        self._path = os.pathsep.join(
            value
            for value in (str(launcher_dir), os.environ.get("PATH", ""))
            if value
        )
        _prepare_launcher(launcher_dir, codex_program, bwrap_program)

    def close(self) -> None:
        """Remove the protected launcher directory."""

        self._launcher_finalizer()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __call__(
        self,
        prompt: str,
        *,
        run_id: str | None = None,
    ) -> CodexRunResult:
        """Run one prompt and persist its trace."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        resolved_run_id = run_id or _new_run_id()
        if not _RUN_ID_RE.fullmatch(resolved_run_id):
            raise ValueError(
                "run_id must start with an alphanumeric character and contain "
                "only letters, digits, '.', '_' or '-' (maximum 128 characters)"
            )

        _require_unchanged(self._trace_root, self.state_dir, "trace path")
        _require_unchanged(self._temp_root, self.state_dir, "temporary path")
        run_dir = self._trace_root / resolved_run_id
        run_dir.mkdir(mode=0o700, exist_ok=False)
        prompt_path = run_dir / "prompt.txt"
        trace_path = run_dir / "trace.jsonl"
        final_message_path = run_dir / "final_message.txt"
        stderr_path = run_dir / "stderr.log"
        prompt_path.write_text(prompt, encoding="utf-8")
        final_message_path.touch()

        environment = os.environ.copy()
        environment.update(
            {
                "PATH": self._path,
                "TMPDIR": str(self._temp_root),
                "TMP": str(self._temp_root),
                "TEMP": str(self._temp_root),
            }
        )
        command = self._command(final_message_path)

        with (
            trace_path.open("w", encoding="utf-8") as trace_file,
            stderr_path.open("w", encoding="utf-8") as stderr_file,
        ):
            try:
                process = subprocess.Popen(
                    command,
                    cwd=self.sandbox_dir,
                    env=environment,
                    stdin=subprocess.PIPE,
                    stdout=trace_file,
                    stderr=stderr_file,
                    text=True,
                    start_new_session=True,
                )
            except OSError as error:
                raise CodexRunnerError(
                    f"could not launch Codex: {error}; trace: {trace_path}",
                    trace_path=trace_path,
                ) from error

            try:
                process.communicate(prompt, timeout=self._timeout)
            except subprocess.TimeoutExpired as error:
                _terminate_process_group(process)
                process.communicate()
                raise CodexRunnerError(
                    f"Codex timed out after {self._timeout}s; trace: {trace_path}",
                    trace_path=trace_path,
                ) from error

        if process.returncode != 0:
            stderr = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
            summary = f": {stderr[-_STDERR_SUMMARY_CHARS:]}" if stderr else ""
            raise CodexRunnerError(
                f"Codex exited with exit code {process.returncode}{summary}; "
                f"trace: {trace_path}",
                trace_path=trace_path,
            )

        _require_unchanged(run_dir, self._trace_root, "run path")
        return CodexRunResult(
            final_message=_read_run_file(final_message_path, run_dir),
            trace_path=trace_path,
        )

    def _command(self, final_message_path: Path) -> list[str]:
        profile = _permission_profile(
            self._codex_program,
            self._bwrap_program,
            self._launcher_dir,
            self._read_roots,
        )
        shell_environment = _shell_environment_config(self._path, self._temp_root)
        command = [
            str(self._codex_program),
            "-a",
            "never",
            "-c",
            f'default_permissions="{_PROFILE_NAME}"',
            "-c",
            f"permissions.{_PROFILE_NAME}={profile}",
            "-c",
            f"shell_environment_policy={shell_environment}",
        ]
        if self._reasoning_effort is not None:
            command.extend(
                (
                    "-c",
                    f"model_reasoning_effort={_toml_string(self._reasoning_effort)}",
                )
            )
        command.extend(
            (
                "exec",
                "--strict-config",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--ephemeral",
                "--color",
                "never",
                "--json",
                "--output-last-message",
                str(final_message_path),
                "-C",
                str(self.sandbox_dir),
                "-",
            )
        )
        return command


def _new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{secrets.token_hex(4)}"


def _require_inside(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must be inside sandbox_dir") from error


def _resolve_inside(
    root: Path,
    value: str | os.PathLike[str],
    label: str,
) -> Path:
    path = Path(value).expanduser()
    resolved = (
        path.resolve(strict=False)
        if path.is_absolute()
        else (root / path).resolve(strict=False)
    )
    _require_inside(resolved, root, label)
    return resolved


def _prepare_internal_dir(
    root: Path,
    value: str | os.PathLike[str],
    label: str,
) -> Path:
    directory = _resolve_inside(root, value, label)
    directory.mkdir(parents=True, exist_ok=True)
    directory = directory.resolve(strict=True)
    _require_inside(directory, root, label)
    return directory


def _resolve_read_roots(
    sandbox: Path,
    values: Sequence[str | os.PathLike[str]],
) -> tuple[Path, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("read_roots must be a sequence of paths")

    roots: list[Path] = []
    for value in values:
        path = Path(value).expanduser().absolute()
        if path.is_symlink():
            raise ValueError("read_roots cannot contain symbolic links")
        root = path.resolve(strict=True)
        if root.is_relative_to(sandbox) or sandbox.is_relative_to(root):
            raise ValueError("read_roots cannot overlap sandbox_dir")
        if root not in roots:
            roots.append(root)

    return tuple(roots)


def _require_unchanged(path: Path, root: Path, label: str) -> None:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise CodexRunnerError(f"{label} is unavailable") from error
    if path.is_symlink() or resolved != path or not path.is_dir():
        raise CodexRunnerError(f"{label} was replaced")
    if not path.is_relative_to(root):
        raise CodexRunnerError(f"{label} escaped state_dir")


def _read_run_file(path: Path, run_dir: Path) -> str:
    if path.is_symlink():
        raise CodexRunnerError("Codex replaced its final-message file")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise CodexRunnerError("Codex did not write its final-message file") from error
    if resolved.parent != run_dir or not resolved.is_file():
        raise CodexRunnerError("Codex replaced its final-message file")
    try:
        return resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise CodexRunnerError("Codex final message is not readable UTF-8") from error


def _resolve_codex_program(codex_bin: str | os.PathLike[str]) -> Path:
    candidate = _resolve_executable(codex_bin, "Codex")
    if candidate.name != "codex.js":
        return candidate

    package_root = candidate.parent.parent
    patterns = (
        "node_modules/@openai/codex-*/vendor/*/bin/codex",
        "vendor/*/bin/codex",
    )
    for pattern in patterns:
        for native in package_root.glob(pattern):
            if _looks_native(native) and os.access(native, os.X_OK):
                return native.resolve(strict=True)
    raise CodexRunnerError(
        f"Codex npm launcher has no native executable: {candidate}"
    )


def _looks_native(path: Path) -> bool:
    try:
        with path.open("rb") as executable:
            magic = executable.read(4)
    except OSError:
        return False
    return magic == b"\x7fELF" or magic in {
        b"\xfe\xed\xfa\xce",
        b"\xfe\xed\xfa\xcf",
        b"\xce\xfa\xed\xfe",
        b"\xcf\xfa\xed\xfe",
    }


def _resolve_bwrap_program(
    codex_program: Path,
    bwrap_bin: str | os.PathLike[str] | None,
) -> Path:
    if bwrap_bin is not None:
        candidate = _resolve_executable(bwrap_bin, "bubblewrap")
        if _is_compatible_bwrap(candidate):
            return candidate
        raise CodexRunnerError(f"incompatible bubblewrap executable: {candidate}")

    executable_dir = codex_program.parent
    candidates = (
        executable_dir / "codex-resources" / "bwrap",
        executable_dir.parent / "codex-resources" / "bwrap",
        executable_dir / "bwrap",
    )
    for candidate in candidates:
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        resolved = candidate.resolve(strict=True)
        if _is_compatible_bwrap(resolved):
            return resolved

    located = shutil.which("bwrap")
    if located is not None:
        candidate = _resolve_executable(located, "bubblewrap")
        if _is_compatible_bwrap(candidate):
            return candidate
    raise CodexRunnerError("no compatible bubblewrap executable found")


def _resolve_executable(
    executable: str | os.PathLike[str],
    label: str,
) -> Path:
    value = os.fspath(executable)
    if os.sep in value or (os.altsep and os.altsep in value):
        candidate = Path(value).expanduser().absolute()
    else:
        located = shutil.which(value)
        if located is None:
            raise CodexRunnerError(f"{label} executable not found: {value}")
        candidate = Path(located)
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise CodexRunnerError(f"{label} executable is not runnable: {candidate}")
    return candidate.resolve(strict=True)


def _is_compatible_bwrap(path: Path) -> bool:
    try:
        completed = subprocess.run(
            (str(path), "--help"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=_BWRAP_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return all(flag in completed.stdout for flag in _BWRAP_FLAGS)


def _toml_string(value: str | os.PathLike[str]) -> str:
    return json.dumps(os.fspath(value), ensure_ascii=True)


def _permission_profile(
    codex_program: Path,
    bwrap_program: Path,
    launcher_dir: Path,
    read_roots: Sequence[Path],
) -> str:
    entries = [
        f'{_toml_string(":minimal")} = "read"',
        f'{_toml_string(codex_program)} = "read"',
        f'{_toml_string(bwrap_program)} = "read"',
        f'{_toml_string(launcher_dir)} = "read"',
    ]
    entries.extend(f'{_toml_string(path)} = "read"' for path in read_roots)
    entries.append(f'{_toml_string(":workspace_roots")} = {{ "." = "write" }}')
    return (
        "{ filesystem = { "
        + ", ".join(entries)
        + " }, network = { enabled = false } }"
    )


def _shell_environment_config(path: str, temp_dir: Path) -> str:
    values = {
        "PATH": path,
        "TMPDIR": str(temp_dir),
        "TMP": str(temp_dir),
        "TEMP": str(temp_dir),
    }
    assignments = ", ".join(
        f"{key} = {_toml_string(value)}" for key, value in values.items()
    )
    return f'{{ inherit = "core", set = {{ {assignments} }} }}'


def _prepare_launcher(
    launcher_dir: Path,
    codex_program: Path,
    bwrap_program: Path,
) -> None:
    (launcher_dir / "apply_patch").symlink_to(codex_program)
    (launcher_dir / "applypatch").symlink_to(codex_program)
    (launcher_dir / "bwrap").symlink_to(bwrap_program)


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        process.wait()

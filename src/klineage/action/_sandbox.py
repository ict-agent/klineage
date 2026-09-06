"""Private resource boundary shared by all actions."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Any

from klineage._utils import operation_id
from klineage.contract import (
    EvaluatorInterface,
    KernelABI,
    ProblemSpec,
    relative_source_path,
)
from klineage.harness._docker import _DockerRuntime, _prepare
from klineage.harness.artifacts import (
    read_generated_source,
    read_generated_source_bundle,
)
from klineage.harness.codex_runner import CodexRunner
from klineage.harness.eval import ValidationResult
from klineage.harness.structured import run_json
from klineage.kernel import Kernel, TargetContext
from klineage.repository import stage_repository

_RUN_VERSION = 2
_RUN_FILE = "run.json"
_WORKSPACE_DIR = "agent-workspace"
_RUNS_DIR = "runs"
_INPUT_DIR = "input"
_SOURCE_DIR = "sources"
_ACTIONS_DIR = "actions"
_WORK_DIR = "work"
_ARTIFACTS_DIR = "artifacts"
_EVALUATIONS_DIR = "evaluations"
_BUILD_DIR = "build"
_RUNNER_TIMEOUT_SECONDS = 900.0
_REASONING_EFFORT = "xhigh"
_HASH_BLOCK_BYTES = 1024 * 1024


class _Kind(StrEnum):
    INIT = "init"
    DECOMPOSE = "decompose"
    CODE_GEN = "code-gen"
    APPLY = "apply"


class _PathKind(StrEnum):
    FILE = "file"
    DIRECTORY = "directory"


class _Sandbox:
    """One action's writable work area and trusted runtime inputs."""

    __slots__ = (
        "_action",
        "_artifacts",
        "_cuda",
        "_evaluator",
        "_logs",
        "_problem",
        "_repo",
        "_root",
        "_run",
        "_runner",
        "_work",
    )

    def __init__(
        self,
        *,
        root: Path,
        run: Path,
        action: Path,
        problem: Path,
        repo: Path,
        runner: CodexRunner,
    ) -> None:
        self._root = root
        self._run = run
        self._action = action
        self._work = action / _WORK_DIR
        self._artifacts = action / _ARTIFACTS_DIR
        self._logs = action / _EVALUATIONS_DIR
        self._problem = problem
        self._repo = repo
        self._runner = runner
        self._cuda: _DockerRuntime | None = None
        self._evaluator = None

    def _file(self, name: str) -> str:
        relative = _output_name(name, "sandbox output file")
        path = self._work / relative
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        return relative

    def _dir(self, name: str) -> str:
        relative = _output_name(name, "sandbox output directory")
        path = self._work / relative
        path.mkdir(mode=0o700, parents=True, exist_ok=False)
        return relative

    def _ask(
        self,
        purpose: str,
        instructions: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        return run_json(self._runner, purpose, instructions, payload)

    def _snapshot(self, sources: Mapping[str, str]) -> str:
        sources = {relative_source_path(name, "audit source"): source
                   for name, source in sources.items()}
        directory = self._run / _INPUT_DIR / _SOURCE_DIR / operation_id("source")
        directory.mkdir(mode=0o700, parents=True)

        # Inputs are read-only to Codex; each audit sees its exact kernel state.
        for name, source in sources.items():
            path = directory / name
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        return str(directory.resolve(strict=True))

    def _take_file(self, name: str, label: str) -> tuple[str, Path]:
        relative = _output_name(name, "sandbox output file")
        source = read_generated_source(
            self._work / relative,
            label,
            allowed_root=self._work,
        )
        artifact = self._artifacts / relative
        artifact.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if artifact.exists():
            raise FileExistsError(artifact)
        artifact.write_text(source, encoding="utf-8")
        return source, artifact.resolve(strict=True)

    def _take_bundle(
        self,
        name: str,
        label: str,
        interface: EvaluatorInterface,
    ) -> tuple[dict[str, str], Path]:
        relative = _output_name(name, "sandbox output directory")
        source_files = read_generated_source_bundle(
            self._work / relative,
            label,
            allowed_root=self._work,
            interface=interface,
        )
        artifact = self._artifacts / relative
        artifact.mkdir(mode=0o700, parents=True, exist_ok=False)
        for source_path, source in source_files.items():
            path = artifact / source_path
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        return source_files, artifact.resolve(strict=True)

    def _inspect(self) -> Mapping[str, Any]:
        return self._runtime().inspect(self._problem, log_dir=self._logs)

    def _evaluate(
        self,
        kernel: Kernel,
        *,
        reference: Kernel | None = None,
    ) -> ValidationResult:
        if self._evaluator is None:
            include_paths = tuple(
                path
                for path in (
                    self._repo / "include",
                    self._repo / "tools/util/include",
                )
                if path.is_dir()
            )
            self._evaluator = self._runtime().evaluator(
                self._problem,
                include_paths=include_paths,
                build_root=self._action / _BUILD_DIR,
                log_dir=self._logs,
                read_paths=(self._run / _INPUT_DIR,),
            )

        return self._evaluator.evaluate(kernel, reference=reference)

    def _save(
        self,
        problem: ProblemSpec,
        abi: KernelABI,
        context: TargetContext,
    ) -> None:
        manifest = _read_run(self._run)
        current_problem = manifest.get("problem_spec")
        current_abi = manifest.get("kernel_abi")
        current_target = manifest.get("target")
        problem_value = problem.to_dict()
        abi_value = abi.to_dict()
        target_value = _target(context)
        if current_problem not in (None, problem_value) or current_abi not in (
            None,
            abi_value,
        ) or current_target not in (None, target_value):
            raise ValueError("run contract is already bound")

        manifest["problem_spec"] = problem_value
        manifest["kernel_abi"] = abi_value
        manifest["target"] = target_value
        _write_json(self._run / _RUN_FILE, manifest)

    def _runtime(self) -> _DockerRuntime:
        if self._cuda is None:
            self._cuda = _prepare(self._root)
        return self._cuda


@contextmanager
def _new(
    kind: _Kind,
    problem: str | os.PathLike[str],
    repo: str | os.PathLike[str] | None = None,
) -> Iterator[_Sandbox]:
    """Create a run and open its first action sandbox."""

    root = _project_root()
    source = _problem_path(problem)
    runs = root / _WORKSPACE_DIR / _RUNS_DIR
    runs.mkdir(mode=0o700, parents=True, exist_ok=True)
    run = runs / operation_id("run")
    inputs = run / _INPUT_DIR
    inputs.mkdir(mode=0o700, parents=True)
    problem_path = _copy_problem(source, inputs)
    repo_path = inputs / "repository"
    if repo is None:
        repo_path.mkdir(mode=0o700)
    else:
        repo_path = stage_repository(repo, repo_path)
    _write_json(
        run / _RUN_FILE,
        {
            "version": _RUN_VERSION,
            "problem_path": problem_path.relative_to(run).as_posix(),
            "problem_sha256": _file_hash(problem_path),
            "repository_path": repo_path.relative_to(run).as_posix(),
            "problem_spec": None,
            "kernel_abi": None,
            "target": None,
        },
    )

    with _open(kind, root, run, problem_path, repo_path) as sandbox:
        yield sandbox


@contextmanager
def _next(kind: _Kind, kernel: Kernel) -> Iterator[_Sandbox]:
    """Resume a trusted run from a committed kernel artifact."""

    root = _project_root()
    run = _kernel_run(root, kernel)
    manifest = _read_run(run)
    problem_path, repo_path = _run_inputs(run, manifest)
    _check_contract(kernel, manifest)

    with _open(kind, root, run, problem_path, repo_path) as sandbox:
        yield sandbox


@contextmanager
def _open(
    kind: _Kind,
    root: Path,
    run: Path,
    problem: Path,
    repo: Path,
) -> Iterator[_Sandbox]:
    if not isinstance(kind, _Kind):
        raise TypeError("action kind must be a _Kind")

    action = run / _ACTIONS_DIR / operation_id(kind.value)
    work = action / _WORK_DIR
    artifacts = action / _ARTIFACTS_DIR
    work.mkdir(mode=0o700, parents=True)
    artifacts.mkdir(mode=0o700)
    with CodexRunner(
        work,
        read_roots=(run / _INPUT_DIR,),
        reasoning_effort=_REASONING_EFFORT,
        timeout=_RUNNER_TIMEOUT_SECONDS,
    ) as runner:
        yield _Sandbox(
            root=root,
            run=run,
            action=action,
            problem=problem,
            repo=repo,
            runner=runner,
        )


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (
            parent / "compose.yaml"
        ).is_file():
            return parent
    raise FileNotFoundError(
        "klineage source must be inside a project containing pyproject.toml "
        "and compose.yaml"
    )


def _problem_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser().absolute()
    if path.is_symlink():
        raise ValueError("problem cannot be a symbolic link")
    path = path.resolve(strict=True)
    if not path.is_file() or path.suffix != ".py":
        raise ValueError("problem must identify a Python source file")
    return path


def _output_name(value: str, label: str) -> str:
    relative = relative_source_path(value, label)
    if "/" in relative:
        raise ValueError(f"{label} must be a direct child of action work")
    return relative


def _copy_problem(problem: Path, inputs: Path) -> Path:
    directory = inputs / "problem" / (problem.parent.name or problem.stem)
    directory.mkdir(mode=0o700, parents=True)
    snapshot = directory / problem.name
    snapshot.write_bytes(problem.read_bytes())
    return snapshot.resolve(strict=True)


def _kernel_run(root: Path, kernel: Kernel) -> Path:
    artifact = kernel.artifact_path
    if artifact is None:
        raise ValueError("kernel has no sandbox artifact")
    if artifact.is_symlink():
        raise ValueError("kernel artifact cannot be a symbolic link")
    artifact = artifact.resolve(strict=True)
    runs = (root / _WORKSPACE_DIR / _RUNS_DIR).resolve(strict=True)
    try:
        relative = artifact.relative_to(runs)
    except ValueError as error:
        raise ValueError("kernel artifact is outside the action runs") from error
    if (
        len(relative.parts) != 5
        or relative.parts[1] != _ACTIONS_DIR
        or relative.parts[3] != _ARTIFACTS_DIR
    ):
        raise ValueError("kernel artifact is outside an action sandbox")
    run = runs / relative.parts[0]
    if not (run / _RUN_FILE).is_file():
        raise ValueError("kernel action run has no manifest")
    _check_artifact(kernel, artifact)
    return run


def _check_artifact(kernel: Kernel, artifact: Path) -> None:
    if artifact.is_file():
        if artifact.read_text(encoding="utf-8") != kernel.source:
            raise ValueError("kernel source does not match its sandbox artifact")
        return
    if not artifact.is_dir() or kernel.source_files is None:
        raise ValueError("kernel bundle does not match its sandbox artifact")

    files = _artifact_files(artifact)
    if set(files) != set(kernel.source_files):
        raise ValueError("kernel bundle does not match its sandbox artifact")
    for relative, source in kernel.source_files.items():
        if files[relative].read_text(encoding="utf-8") != source:
            raise ValueError("kernel bundle does not match its sandbox artifact")


def _artifact_files(artifact: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    pending = [artifact]
    while pending:
        directory = pending.pop()
        for path in directory.iterdir():
            if path.is_symlink():
                raise ValueError("kernel bundle contains a symbolic link")
            if path.is_dir():
                pending.append(path)
                continue
            if not path.is_file():
                raise ValueError("kernel bundle contains a non-file artifact")
            relative = path.relative_to(artifact).as_posix()
            files[relative] = path
    return files


def _read_run(run: Path) -> dict[str, Any]:
    path = run / _RUN_FILE
    if path.is_symlink() or not path.is_file():
        raise ValueError("action run manifest is unavailable")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("action run manifest is invalid") from error
    if not isinstance(value, dict) or value.get("version") != _RUN_VERSION:
        raise ValueError("action run manifest version is invalid")
    return value


def _run_inputs(run: Path, manifest: Mapping[str, Any]) -> tuple[Path, Path]:
    problem = _run_path(
        run,
        manifest.get("problem_path"),
        "problem",
        _PathKind.FILE,
    )
    repo = _run_path(
        run,
        manifest.get("repository_path"),
        "repository",
        _PathKind.DIRECTORY,
    )
    digest = manifest.get("problem_sha256")
    if not isinstance(digest, str) or _file_hash(problem) != digest:
        raise ValueError("action run problem snapshot changed")
    return problem, repo


def _run_path(run: Path, value: Any, label: str, kind: _PathKind) -> Path:
    if not isinstance(value, str):
        raise TypeError(f"action run {label} path is invalid")
    relative = relative_source_path(value, f"action run {label} path")
    path = run / relative
    if path.is_symlink():
        raise ValueError(f"action run {label} cannot be a symbolic link")
    path = path.resolve(strict=True)
    inputs = (run / _INPUT_DIR).resolve(strict=True)
    if not path.is_relative_to(inputs):
        raise ValueError(f"action run {label} is outside trusted input")
    valid = path.is_file() if kind is _PathKind.FILE else path.is_dir()
    if not valid:
        raise ValueError(f"action run {label} is not a {kind.value}")
    return path


def _check_contract(kernel: Kernel, manifest: Mapping[str, Any]) -> None:
    if kernel.problem is None or kernel.abi is None:
        raise ValueError("kernel has no problem or ABI contract")
    if manifest.get("problem_spec") != kernel.problem.to_dict():
        raise ValueError("kernel problem does not match its run contract")
    if manifest.get("kernel_abi") != kernel.abi.to_dict():
        raise ValueError("kernel ABI does not match its run contract")
    if manifest.get("target") != _target(kernel.context):
        raise ValueError("kernel target does not match its run contract")


def _target(context: TargetContext) -> dict[str, str]:
    if not isinstance(context, TargetContext):
        raise TypeError("kernel target must be a TargetContext")
    return {
        "case": context.case,
        "language": context.language,
        "platform": context.platform,
    }


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(_HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

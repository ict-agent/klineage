"""Run local Codex processes and retain their traces."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, get_args

from klineage.constants import (
    MEMORY_DIRECTORY,
    MESSAGE_DIRECTORY,
    SKILL_DIRECTORY,
    STATE_DIRECTORY,
    STATS_DIRECTORY,
    RunKind,
)
from klineage.harness.codex_skills import render_memory
from klineage.harness.process import stop_process
from klineage.logging import LOG_ENV
from klineage.tools import write_agent_docs
from klineage.utils import operation_id

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REASONING_EFFORTS = frozenset(get_args(ReasoningEffort))
_STDERR_SUMMARY_CHARS = 1000
#: Flags both `exec` and `exec resume` accept; order matches their shared usage.
COMMON_EXEC_FLAGS = (
    "--strict-config",
    "--ignore-user-config",
    "--ignore-rules",
    "--skip-git-repo-check",
    "--json",
)
#: Flags only plain `exec` accepts.
PLAIN_EXEC_FLAGS = ("--color", "never")
#: Sandbox flag for plain `exec`; `resume` has no `-s` and uses this instead.
RESUME_SANDBOX_FLAGS = ("--dangerously-bypass-approvals-and-sandbox",)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = PACKAGE_ROOT / "skills"
#: Native skills staged into every workspace. CUDA is the only backend this
#: host can run, so the HIP and AscendC instructions would only add noise.
SKILL_NAMES = ("bench", "cuda")


class CodexRunnerError(RuntimeError):
    """A Codex launch or execution failure."""

    def __init__(self, message: str, *, trace_path: Path | None = None):
        self.trace_path = trace_path
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class CodexRunResult:
    """One response and its process trace."""

    final_message: str
    trace_path: Path


class CodexRunner:
    """Run Codex with the host's filesystem and network access."""

    #: Provider name Codex is configured with; distinct from the host's own so a
    #: run never inherits unrelated settings from the user's config.
    PROVIDER = "klineage"
    #: Key variable named when a caller does not supply its own.
    DEFAULT_KEY_ENV = "AIPING_API_KEY"

    def __init__(
        self,
        work_dir: str | os.PathLike[str],
        *,
        codex_bin: str | os.PathLike[str] = "codex",
        reasoning_effort: ReasoningEffort | None = "xhigh",
        timeout: float | None = None,
        resume_id: str | None = None,
        state_dir: str | os.PathLike[str] = STATE_DIRECTORY,
        api_base_url: str | None = None,
        api_key_env: str | None = None,
        skill_names: Sequence[str] = SKILL_NAMES,
    ):
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if reasoning_effort not in _REASONING_EFFORTS and reasoning_effort is not None:
            raise ValueError("invalid reasoning effort")
        if resume_id is not None and not _RUN_ID_RE.fullmatch(resume_id):
            raise ValueError("resume_id must be a recorded session identifier")
        if api_base_url is not None and not api_base_url.strip():
            raise ValueError("api_base_url must be a non-empty string")
        if api_key_env is not None and not api_key_env.isidentifier():
            raise ValueError("api_key_env must name an environment variable")

        self.work_dir = Path(work_dir).expanduser().resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir = (self.work_dir / Path(state_dir).expanduser()).resolve()
        self.trace_root = self.state_dir / "codex-runs"
        self.trace_root.mkdir(parents=True, exist_ok=True)
        self.stats_dir = self.state_dir / STATS_DIRECTORY.name
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.resume_id = resume_id
        self.api_base_url = api_base_url.strip() if api_base_url else None
        self.api_key_env = api_key_env
        self.codex_program = resolve_executable(codex_bin)

        # Expose packaged instructions through Codex's workspace skill discovery.
        skill_dir = self.work_dir / SKILL_DIRECTORY
        skill_dir.mkdir(parents=True, exist_ok=True)
        message_dir = self.work_dir / MESSAGE_DIRECTORY
        message_dir.parent.mkdir(parents=True, exist_ok=True)
        resources = [(SKILL_ROOT / name, skill_dir / name) for name in skill_names]
        resources.append((PACKAGE_ROOT / "message", message_dir))
        for source, destination in resources:
            _link_resource(source, destination)

    def mount_memory(
        self, memory: Path | None, *, exclude_skills: Sequence[str] = ()
    ) -> Path | None:
        """Stage discoverable native skills; None requires a memory-free workdir."""
        destination = self.work_dir / MEMORY_DIRECTORY
        if memory is None:
            if destination.exists() or destination.is_symlink():
                raise ValueError(
                    f"memory-free execution requires a fresh workdir: {destination}"
                )
            return None

        source = memory.expanduser().resolve(strict=True)
        if not source.is_dir():
            raise NotADirectoryError(f"memory must be a directory: {source}")
        location = destination.parent.resolve() / destination.name
        if location.is_relative_to(source):
            raise ValueError("memory cannot contain its workspace mount")
        files = render_memory(source, exclude_skills=exclude_skills)
        if destination.exists() or destination.is_symlink():
            if _matches_memory(destination, files):
                return destination
            raise CodexRunnerError(f"workspace resource already exists: {destination}")

        # Publish all skills together on the destination filesystem.
        with TemporaryDirectory(dir=destination.parent) as temporary:
            staged = Path(temporary) / destination.name
            staged.mkdir()
            for relative, contents in files.items():
                path = staged / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8")
            staged.rename(destination)
        return destination

    def __call__(
        self,
        prompt: str,
        *,
        run_id: str | None = None,
    ) -> CodexRunResult:
        """Run one prompt and persist its trace."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        resolved_run_id = run_id or operation_id(RunKind.CODEX)
        if not _RUN_ID_RE.fullmatch(resolved_run_id):
            raise ValueError(
                "run_id must start with an alphanumeric character and contain "
                "only letters, digits, '.', '_' or '-' (maximum 128 characters)"
            )

        write_agent_docs(self.work_dir)

        run_dir = self.trace_root / resolved_run_id
        run_dir.mkdir(mode=0o700, exist_ok=False)
        prompt_path = run_dir / "prompt.txt"
        trace_path = run_dir / "trace.jsonl"
        final_message_path = run_dir / "final_message.txt"
        stderr_path = run_dir / "stderr.log"
        prompt_path.write_text(prompt, encoding="utf-8")
        final_message_path.touch()

        command = self.command(final_message_path)
        environment = os.environ.copy()
        python_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(PACKAGE_ROOT.parent), python_path)
            if python_path
            else (str(PACKAGE_ROOT.parent),)
        )
        # Point this run's measurements at its own workspace, so evaluate and
        # profile records land under <work_dir>/.klineage/stats. The path travels
        # by environment because the evaluator worker is a fresh process.
        environment[LOG_ENV] = str(self.stats_dir)

        with (
            trace_path.open("w", encoding="utf-8") as trace_file,
            stderr_path.open("w", encoding="utf-8") as stderr_file,
        ):
            try:
                process = subprocess.Popen(
                    command,
                    cwd=self.work_dir,
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
                process.communicate(prompt, timeout=self.timeout)
            except subprocess.TimeoutExpired as error:
                stop_process(process)
                process.communicate()
                raise CodexRunnerError(
                    f"Codex timed out after {self.timeout}s; trace: {trace_path}",
                    trace_path=trace_path,
                ) from error
            except BaseException:
                # Cancellation must also stop detached descendants.
                stop_process(process)
                process.communicate()
                raise

        if process.returncode != 0:
            stderr = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
            summary = f": {stderr[-_STDERR_SUMMARY_CHARS:]}" if stderr else ""
            raise CodexRunnerError(
                f"Codex exited with exit code {process.returncode}{summary}; "
                f"trace: {trace_path}",
                trace_path=trace_path,
            )

        try:
            final_message = final_message_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise CodexRunnerError(
                f"Codex reply is unreadable; trace: {trace_path}",
                trace_path=trace_path,
            ) from error
        return CodexRunResult(final_message=final_message, trace_path=trace_path)

    def command(self, final_message_path: Path) -> list[str]:
        command = [str(self.codex_program), "-a", "never"]
        if self.reasoning_effort is not None:
            command.extend(
                ("-c", f"model_reasoning_effort={json.dumps(self.reasoning_effort)}")
            )
        if self.api_base_url is not None:
            command.extend(
                _provider_flags(
                    self.PROVIDER,
                    self.api_base_url,
                    self.api_key_env or self.DEFAULT_KEY_ENV,
                )
            )
        command.append("exec")
        # `exec resume` accepts neither `-s` nor `-C`, so it runs with the session's
        # own sandbox and workspace; cwd still comes from Popen.
        if self.resume_id is not None:
            command.extend(("resume", self.resume_id))
            command.extend(RESUME_SANDBOX_FLAGS)
            command.extend(COMMON_EXEC_FLAGS)
            command.extend(("--output-last-message", str(final_message_path), "-"))
            return command
        command.extend(("-s", "danger-full-access"))
        command.extend(PLAIN_EXEC_FLAGS)
        command.extend(COMMON_EXEC_FLAGS)
        command.extend(
            (
                "--output-last-message",
                str(final_message_path),
                "-C",
                str(self.work_dir),
                "-",
            )
        )
        return command


def _provider_flags(name: str, base_url: str, key_env: str) -> list[str]:
    """Pin one OpenAI-compatible provider so the endpoint stays where we point it."""

    return [
        "-c", "model_provider=" + name,
        "-c", f"model_providers.{name}.name={name}",
        "-c", f"model_providers.{name}.base_url={base_url}",
        "-c", f"model_providers.{name}.env_key={key_env}",
        "-c", f"model_providers.{name}.wire_api=responses",
        "-c", f"model_providers.{name}.requires_openai_auth=false",
    ]


def session_id(trace_path: Path) -> str | None:
    """The Codex session recorded by a run, so a retry can continue its thread."""

    try:
        lines = Path(trace_path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "thread.started":
            thread = event.get("thread_id")
            if isinstance(thread, str) and _RUN_ID_RE.fullmatch(thread):
                return thread
    return None


def _matches_memory(directory: Path, files: dict[Path, str]) -> bool:
    if directory.is_symlink() or not directory.is_dir():
        return False
    paths = tuple(directory.rglob("*"))
    if any(path.is_symlink() for path in paths):
        return False
    existing = {
        path.relative_to(directory): path.read_bytes()
        for path in paths
        if path.is_file()
    }
    return existing == {path: text.encode("utf-8") for path, text in files.items()}


def _link_resource(source: Path, destination: Path):
    try:
        destination.symlink_to(source, target_is_directory=True)
    except FileExistsError:
        if destination.is_symlink() and destination.resolve() == source:
            return
        raise CodexRunnerError(f"workspace resource already exists: {destination}")


def resolve_executable(executable: str | os.PathLike[str]) -> Path:
    value = os.fspath(executable)
    if os.sep in value or (os.altsep and os.altsep in value):
        candidate = Path(value).expanduser().absolute()
    else:
        located = shutil.which(value)
        if located is None:
            raise CodexRunnerError(f"Codex executable not found: {value}")
        candidate = Path(located)
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise CodexRunnerError(f"Codex executable is not runnable: {candidate}")
    return candidate.resolve(strict=True)

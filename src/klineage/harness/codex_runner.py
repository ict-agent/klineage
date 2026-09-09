"""Run local Codex processes and retain their traces."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from klineage._utils import operation_id
from klineage.backend import BACKENDS
from klineage.harness.artifacts import stop_process

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REASONING_EFFORTS = frozenset(
    ("none", "minimal", "low", "medium", "high", "xhigh", "max")
)
_STDERR_SUMMARY_CHARS = 1000
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = PACKAGE_ROOT / "skills"
SKILL_NAMES = ("bench", *(backend.skill_name for backend in BACKENDS))
SKILL_DIRECTORY = Path(".agents") / "skills"
MESSAGE_DIRECTORY = Path(".klineage") / "message"


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

    def __init__(
        self,
        work_dir: str | os.PathLike[str],
        *,
        codex_bin: str | os.PathLike[str] = "codex",
        reasoning_effort: ReasoningEffort | None = "xhigh",
        timeout: float | None = None,
        state_dir: str | os.PathLike[str] = ".klineage",
    ):
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if reasoning_effort not in _REASONING_EFFORTS and reasoning_effort is not None:
            raise ValueError("invalid reasoning effort")

        self.work_dir = Path(work_dir).expanduser().resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir = (self.work_dir / Path(state_dir).expanduser()).resolve()
        self.trace_root = self.state_dir / "codex-runs"
        self.trace_root.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.codex_program = resolve_executable(codex_bin, "Codex")

        # Expose packaged instructions through Codex's workspace skill discovery.
        skill_dir = self.work_dir / SKILL_DIRECTORY
        skill_dir.mkdir(parents=True, exist_ok=True)
        message_dir = self.work_dir / MESSAGE_DIRECTORY
        message_dir.parent.mkdir(parents=True, exist_ok=True)
        resources = [(SKILL_ROOT / name, skill_dir / name) for name in SKILL_NAMES]
        resources.append((PACKAGE_ROOT / "message", message_dir))
        for source, destination in resources:
            try:
                destination.symlink_to(source, target_is_directory=True)
            except FileExistsError:
                if destination.is_symlink() and destination.resolve() == source:
                    continue
                raise CodexRunnerError(
                    f"workspace resource already exists: {destination}"
                )

    def __call__(
        self,
        prompt: str,
        *,
        run_id: str | None = None,
    ) -> CodexRunResult:
        """Run one prompt and persist its trace."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        resolved_run_id = run_id or operation_id("codex")
        if not _RUN_ID_RE.fullmatch(resolved_run_id):
            raise ValueError(
                "run_id must start with an alphanumeric character and contain "
                "only letters, digits, '.', '_' or '-' (maximum 128 characters)"
            )

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
        command.extend(
            (
                "exec",
                "-s",
                "danger-full-access",
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
                str(self.work_dir),
                "-",
            )
        )
        return command


def resolve_executable(
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

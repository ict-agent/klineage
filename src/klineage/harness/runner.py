"""Shared agent lifecycle, sandbox and evaluation gateways."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory

from klineage.backend import backends
from klineage.constants import (
    AGENT_FILES,
    MEMORY_DIRECTORY,
    RUN_RECORDS_DIRECTORY,
    SKILL_DIRECTORY,
    STATE_DIRECTORY,
    WORK_DIRECTORY,
)
from klineage.harness.codex_skills import render_memory
from klineage.harness.logging import LOG_ENV, LOG_FILE
from klineage.harness.process import PROCESS_SCOPE_ENV, stop_process
from klineage.harness.sandbox import AgentRuntime, Sandbox, runtime_paths
from klineage.tools import write_agent_docs
from klineage.utils import DirectoryMode, operation_id, safe_directory, safe_file

RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_STDERR_SUMMARY_CHARS = 1000


class RunnerKind(StrEnum):
    CODEX = "codex"
    CLAUDE_CODE = "claude-code"


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BENCH_SKILL = Path(__file__).parent / "resources" / "bench"
PROVIDER_KEY_PLACEHOLDER = "sandbox-provider"
RESPONSE_RESERVE_SECONDS = 60


class RunnerError(RuntimeError):
    """Report an agent launch, timeout, execution or reply failure.

    Attributes:
        trace_path: Recorded JSONL trace, when a run reached trace creation.
    """

    def __init__(self, message: str, *, trace_path: Path | None = None):
        self.trace_path = trace_path
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class RunResult:
    """Retain one completed response and its process trace.

    Attributes:
        final_message: Final text returned by the agent.
        trace_path: JSONL event trace under the runner's state directory.
    """

    final_message: str
    trace_path: Path


class AgentRunner:
    """Share process lifetime and artifact access across agent CLIs."""

    name = "Agent"
    kind = "agent"

    def __init__(
        self, work_dir, *, program, settings, timeout=None, resume_id=None,
        state_dir=STATE_DIRECTORY, read_only=(), evaluation=None, agent=None,
    ):
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if resume_id is not None and not RUN_ID.fullmatch(resume_id):
            raise ValueError("resume_id must be a recorded session identifier")
        self.work_dir = Path(os.path.abspath(Path(work_dir).expanduser()))
        with safe_directory(self.work_dir, DirectoryMode.CREATE):
            pass
        self.state_dir = Path(os.path.abspath(self.work_dir / Path(state_dir).expanduser()))
        self.trace_root = self.state_dir / RUN_RECORDS_DIRECTORY
        with safe_directory(self.trace_root, DirectoryMode.CREATE):
            pass
        self.timeout = timeout
        self.resume_id = resume_id
        self.program = resolve_executable(program)
        self.model = settings["model"]
        self.reasoning_effort = settings["reasoning_effort"]
        self.api_base_url = settings["api_base_url"]
        self.api_key_env = settings["api_key_env"]
        self.api_key = settings["api_key"]
        self.provider_headers = settings["http_headers"]
        self.model_options = settings["options"]
        self.agent = agent or AgentRuntime()
        self.read_only = tuple(Path(path).absolute() for path in read_only)
        self.evaluation = evaluation
        if self.evaluation is None and os.environ.get("KLINEAGE_SERVER"):
            from klineage.server import Client
            self.evaluation = Client()

        skill_dir = self.work_dir / SKILL_DIRECTORY
        with safe_directory(skill_dir, DirectoryMode.CREATE):
            pass
        skill_paths = {"bench": BENCH_SKILL, **{backend.language: backend.skill_root for backend in backends()}}
        for name, source in skill_paths.items():
            _link_resource(source, skill_dir / name)

    def mount_memory(
        self, memory: Path | None, *, exclude_skills: Sequence[str] = ()
    ) -> Path | None:
        """Stage repository skillcards as agent-readable instructions.

        Args:
            memory: Memory repository directory, or `None` to require a workspace
                without previously staged memory.
            exclude_skills: Skill identifiers omitted from the staged memory.

        Returns:
            Staged memory directory, or `None` when memory is disabled.
        """
        destination = self.work_dir / MEMORY_DIRECTORY
        with safe_directory(destination.parent):
            pass
        if memory is None:
            if destination.exists() or destination.is_symlink():
                raise ValueError(
                    f"memory-free execution requires a fresh workdir: {destination}"
                )
            return None

        source = memory.expanduser().resolve(strict=True)
        if not source.is_dir():
            raise NotADirectoryError(f"memory must be a directory: {source}")
        if destination.is_relative_to(source):
            raise ValueError("memory cannot contain its workspace mount")
        files = render_memory(source, exclude_skills=exclude_skills)
        if destination.exists() or destination.is_symlink():
            if _matches_memory(destination, files):
                return destination
            raise RunnerError(f"workspace resource already exists: {destination}")

        # Publish through the opened parent even if its pathname is replaced.
        with safe_directory(destination.parent) as parent:
            temporary = operation_id("memory")
            os.mkdir(temporary, mode=0o700, dir_fd=parent)
            try:
                for relative, contents in files.items():
                    with safe_file(destination.parent / temporary / relative, "w") as stream:
                        stream.write(contents)
                os.rename(temporary, destination.name, src_dir_fd=parent, dst_dir_fd=parent)
            finally:
                with suppress(FileNotFoundError):
                    shutil.rmtree(temporary, dir_fd=parent)
        return destination

    def __call__(
        self,
        prompt: str,
        *,
        run_id: str | None = None,
    ) -> RunResult:
        """Run one prompt and persist its trace, final response and stderr.

        Args:
            prompt: Nonempty text sent to the agent.
            run_id: Unique trace-directory identifier; generated when omitted.

        Returns:
            Final response and the JSONL trace path.

        Raises:
            RunnerError: Launch, timeout, execution or reply reading fails.
            ValueError: The prompt or run identifier is invalid.
        """

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        resolved_run_id = run_id or operation_id(self.kind)
        if not RUN_ID.fullmatch(resolved_run_id):
            raise ValueError(
                "run_id must start with an alphanumeric character and contain "
                "only letters, digits, '.', '_' or '-' (maximum 128 characters)"
            )

        prompt = (
            f"Read `{AGENT_FILES[0]}` before starting. Read relevant guidance in "
            f"`{SKILL_DIRECTORY}/` as needed. Keep scripts, diagnostics, backups, "
            f"logs and other non-artifact files under `{WORK_DIRECTORY}/`.\n\n"
            + prompt
        )
        if self.timeout is not None:
            deadline = datetime.now(UTC) + timedelta(seconds=self.timeout)
            reserve = min(RESPONSE_RESERVE_SECONDS, self.timeout / 2)
            tools_deadline = deadline - timedelta(seconds=reserve)
            prompt += (
                "\n\n# Execution time budget\n"
                f"Deadline (UTC): {deadline.isoformat()}\n"
                f"Finish tool calls by (UTC): {tools_deadline.isoformat()}\n"
                f"This attempt has at most {self.timeout:g} seconds. "
                "Complete evaluations and select the final artifact before the tool deadline. "
                f"Reserve the remaining {reserve:g} seconds for your final response. "
                "Check `date -u` before long tool calls."
            )

        write_agent_docs(self.work_dir)

        run_dir = self.trace_root / resolved_run_id
        with safe_directory(self.trace_root) as parent:
            os.mkdir(resolved_run_id, mode=0o700, dir_fd=parent)
        prompt_path = run_dir / "prompt.txt"
        trace_path = run_dir / "trace.jsonl"
        final_message_path = run_dir / "final_message.txt"
        stderr_path = run_dir / "stderr.log"
        with safe_file(prompt_path, "w") as stream:
            stream.write(prompt)
        with safe_file(run_dir / LOG_FILE, "w"):
            pass

        with (
            safe_file(trace_path, "w") as trace_file,
            safe_file(stderr_path, "w+", errors="replace") as stderr_file,
            safe_file(final_message_path, "w+") as final_file,
            self.execution(final_message_path, run_dir) as (command, environment),
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
                raise RunnerError(
                    f"could not launch {self.name}: {error}; trace: {trace_path}",
                    trace_path=trace_path,
                ) from error

            try:
                process.communicate(prompt, timeout=self.timeout)
            except subprocess.TimeoutExpired as error:
                raise RunnerError(
                    f"{self.name} timed out after {self.timeout}s; trace: {trace_path}",
                    trace_path=trace_path,
                ) from error
            finally:
                # Reclaim detached tools on completion, failure and cancellation.
                stop_process(process, scope=str(run_dir))
                process.communicate()

            if process.returncode != 0:
                stderr_file.seek(0)
                stderr = stderr_file.read().strip()
                summary = f": {stderr[-_STDERR_SUMMARY_CHARS:]}" if stderr else ""
                raise RunnerError(
                    f"{self.name} exited with exit code {process.returncode}{summary}; "
                    f"trace: {trace_path}",
                    trace_path=trace_path,
                )

            try:
                for path, opened in (
                    (trace_path, trace_file), (stderr_path, stderr_file),
                    (final_message_path, final_file),
                ):
                    with safe_file(path) as current:
                        if not os.path.samestat(os.fstat(opened.fileno()), os.fstat(current.fileno())):
                            raise OSError(f"trace file changed during agent execution: {path}")
                final_message = self.read_reply(final_file, trace_path)
                if not final_message.strip():
                    raise OSError(f"{self.name} produced no final response")
            except (OSError, UnicodeDecodeError) as error:
                raise RunnerError(
                    f"{self.name} reply is unreadable; trace: {trace_path}",
                    trace_path=trace_path,
                ) from error
        return RunResult(final_message=final_message, trace_path=trace_path)

    @contextmanager
    def execution(self, final_message_path: Path, run_dir: Path):
        """Expose session state, declared inputs and scoped IPC endpoints."""
        home, state_env = self.session_home()
        with safe_directory(home, DirectoryMode.CREATE):
            pass
        writable = [home]
        if not final_message_path.is_relative_to(self.work_dir):
            writable.extend((final_message_path, run_dir / LOG_FILE))
        environment = {
            **self.agent.environment,
            "PATH": f"{self.agent.python.parent}:{self.agent.environment.get('PATH', '/usr/bin:/bin')}",
            "PYTHONPATH": os.pathsep.join(filter(None, (
                str(PACKAGE_ROOT.parent), self.agent.environment.get("PYTHONPATH"),
            ))),
            **state_env,
            LOG_ENV: str(run_dir),
            PROCESS_SCOPE_ENV: str(run_dir),
        }
        read_only = [
            *runtime_paths(self.agent.python, self.program),
            PACKAGE_ROOT.parent, *self.agent.read_only, *self.read_only,
        ]
        memory = self.work_dir / MEMORY_DIRECTORY
        if memory.is_dir():
            read_only.append(memory)
        catalog = getattr(self, "model_catalog", None)
        if catalog is not None and not Path(catalog).is_relative_to(self.work_dir):
            read_only.append(catalog)

        with ExitStack() as stack:
            ipc = Path(stack.enter_context(TemporaryDirectory(prefix="klineage-ipc-")))
            sockets = []
            if self.evaluation is not None:
                from klineage.server import ClientGateway
                from klineage.server.client import link_problem

                gateway = stack.enter_context(ClientGateway(
                    self.evaluation, ipc / "evaluation.sock", directory=run_dir,
                ))
                sockets.append(gateway.socket_path)
                environment["KLINEAGE_SERVER"] = gateway.endpoint
                for name, value in gateway.info().get("problem_paths", {}).items():
                    link_problem(self.work_dir, name, {"path": value})
                    path = Path(value)
                    if path.is_dir():
                        read_only.append(path.resolve())
                    nodes = [gateway.problem(name).tree]
                    while nodes:
                        node = nodes.pop()
                        nodes.extend(node.children)
                        if node.reference is not None:
                            source = Path(node.reference.path)
                            if source.is_file():
                                read_only.append(source.resolve())

            bridge = None
            original_url = self.api_base_url
            if original_url:
                from klineage.harness.bridge import ProviderBridge
                if self.api_key is None and self.api_key_env is not None:
                    raise ValueError(f"No API key: set {self.api_key_env} on the host")
                bridge = stack.enter_context(ProviderBridge(
                    original_url, self.bridge_key(), ipc / "provider.sock",
                    headers=self.provider_headers, directory=run_dir, wire_api=self.wire_api,
                ))
                self.api_base_url = bridge.url
                environment.update(self.provider_env(bridge.url))
                sockets.append(bridge.socket_path)
            try:
                command = self.command(final_message_path)
            finally:
                self.api_base_url = original_url
            if bridge is not None:
                command = bridge.command(command, python=self.agent.python)

            sandbox = Sandbox(
                self.work_dir, read_only=read_only, writable=writable, sockets=sockets,
                devices=self.agent.device_nodes,
                environment=environment,
            )
            yield sandbox.command(command), sandbox.environment

    def bridge_key(self):
        return self.api_key

    def read_reply(self, final_file, trace_path):
        final_file.seek(0)
        return final_file.read()

    def session_home(self):
        raise NotImplementedError

    def provider_env(self, url):
        raise NotImplementedError

    def command(self, final_message_path):
        raise NotImplementedError


def runner_type(name=None):
    """Select a CLI; defaults to KLINEAGE_RUNNER, then Codex."""
    from .claude_code_runner import ClaudeCodeRunner
    from .codex_runner import CodexRunner

    selected = RunnerKind(name or os.environ.get("KLINEAGE_RUNNER") or RunnerKind.CODEX)
    return {RunnerKind.CODEX: CodexRunner, RunnerKind.CLAUDE_CODE: ClaudeCodeRunner}[selected]


def create_runner(work_dir, *, runner=None, **options):
    """Construct the selected CLI runner with its provider settings."""
    return runner_type(runner)(work_dir, **options)


def session_id(trace_path: Path) -> str | None:
    """Read a Codex or Claude Code session ID for an action retry."""

    try:
        with safe_file(trace_path) as stream:
            lines = stream.read().splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "thread.started":
            thread = event.get("thread_id")
        elif event.get("type") == "system" and event.get("subtype") == "init":
            thread = event.get("session_id")
        else:
            continue
        if isinstance(thread, str) and RUN_ID.fullmatch(thread):
            return thread
    return None


def _matches_memory(directory: Path, files: dict[Path, str]) -> bool:
    if directory.is_symlink() or not directory.is_dir():
        return False
    paths = tuple(directory.rglob("*"))
    if any(path.is_symlink() for path in paths):
        return False
    existing = {}
    try:
        for path in paths:
            if path.is_file():
                with safe_file(path) as stream:
                    existing[path.relative_to(directory)] = stream.read().encode("utf-8")
    except (OSError, UnicodeError):
        return False
    return existing == {path: text.encode("utf-8") for path, text in files.items()}


def _link_resource(source: Path, destination: Path):
    with safe_directory(destination.parent) as parent:
        try:
            os.symlink(source, destination.name, dir_fd=parent)
        except FileExistsError:
            try:
                target = os.readlink(destination.name, dir_fd=parent)
            except OSError:
                target = None
            if target == str(source):
                return
            raise RunnerError(f"workspace resource already exists: {destination}") from None


def resolve_executable(executable: str | os.PathLike[str]) -> Path:
    value = os.fspath(executable)
    if os.sep in value or (os.altsep and os.altsep in value):
        candidate = Path(value).expanduser().absolute()
    else:
        located = shutil.which(value)
        if located is None:
            raise RunnerError(f"Agent executable not found: {value}")
        candidate = Path(located)
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise RunnerError(f"Agent executable is not runnable: {candidate}")
    return candidate.resolve(strict=True)

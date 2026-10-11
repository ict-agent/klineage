"""Run local Codex processes and retain their traces."""

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
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, get_args

from klineage.backend import backends
from klineage.constants import (
    AGENT_FILES,
    MEMORY_DIRECTORY,
    SKILL_DIRECTORY,
    STATE_DIRECTORY,
    WORK_DIRECTORY,
    RunKind,
)
from klineage.harness.codex_skills import render_memory
from klineage.harness.logging import LOG_ENV, LOG_FILE
from klineage.harness.process import PROCESS_SCOPE_ENV, stop_process
from klineage.harness.sandbox import AgentRuntime, Sandbox, runtime_paths
from klineage.serialization import to_json
from klineage.tools import write_agent_docs
from klineage.utils import DirectoryMode, operation_id, safe_directory, safe_file

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
BENCH_SKILL = Path(__file__).parent / "resources" / "bench"
PROVIDER_KEY_PLACEHOLDER = "sandbox-provider"
RESPONSE_RESERVE_SECONDS = 60


class CodexRunnerError(RuntimeError):
    """Report a Codex launch, timeout, execution or reply failure.

    Attributes:
        trace_path: Recorded JSONL trace, when a run reached trace creation.
    """

    def __init__(self, message: str, *, trace_path: Path | None = None):
        self.trace_path = trace_path
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class CodexRunResult:
    """Retain one completed response and its process trace.

    Attributes:
        final_message: Final text returned by the agent.
        trace_path: JSONL event trace under the runner's state directory.
    """

    final_message: str
    trace_path: Path


class CodexRunner:
    """Run Codex in an OS sandbox with scoped model and evaluation gateways.

    Explicit provider keys remain in the host-side bridge. Codex reuses the
    host CODEX_HOME for configuration and sessions. An evaluation client
    exposes its fixed problem set through a separate Unix socket. Device access and
    extra dependencies come from `AgentRuntime`.

    Model settings fall back to `$CODEX_HOME/config.toml` (`~/.codex/config.toml`
    by default). Explicit arguments override `KLINEAGE_*` and file settings.

    Args:
        work_dir: Dedicated writable agent workspace; created if absent.
        codex_bin: Codex executable name or path.
        model: Model name; defaults to `KLINEAGE_MODEL`, then the host config.
        reasoning_effort: Reasoning level; `None` uses the host config.
        timeout: Seconds per invocation, or `None` for no process deadline.
        resume_id: Previously recorded session identifier to resume.
        state_dir: Trace and statistics directory, relative to `work_dir` unless
            absolute.
        api_base_url: Provider endpoint; defaults to `KLINEAGE_API_BASE_URL`, then
            the host provider configuration.
        api_key_env: Host key variable; defaults to `KLINEAGE_API_KEY_ENV`, then
            the configured provider's key variable.
        api_key: Explicit host-side provider key, overriding the key variable.
        read_only: Additional input paths exposed read-only.
        evaluation (klineage.server.Client | None): Client exposing a bound
            problem or its registered set; inferred from `KLINEAGE_SERVER`.
        agent: Agent runtime configuration; defaults to `AgentRuntime()`.
    """

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
        model: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        timeout: float | None = None,
        resume_id: str | None = None,
        state_dir: str | os.PathLike[str] = STATE_DIRECTORY,
        api_base_url: str | None = None,
        api_key_env: str | None = None,
        api_key: str | None = None,
        read_only: Sequence[Path] = (),
        evaluation=None,
        agent: AgentRuntime | None = None,
    ):
        from klineage.harness.provider import model_settings

        settings = model_settings(
            model=model, reasoning_effort=reasoning_effort,
            api_base_url=api_base_url, api_key_env=api_key_env, api_key=api_key,
        )
        model = settings["model"]
        reasoning_effort = settings["reasoning_effort"]
        api_base_url = settings["api_base_url"]
        api_key_env = settings["api_key_env"]
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if reasoning_effort not in _REASONING_EFFORTS and reasoning_effort is not None:
            raise ValueError("invalid reasoning effort")
        if resume_id is not None and not _RUN_ID_RE.fullmatch(resume_id):
            raise ValueError("resume_id must be a recorded session identifier")

        self.work_dir = Path(os.path.abspath(Path(work_dir).expanduser()))
        with safe_directory(self.work_dir, DirectoryMode.CREATE):
            pass
        self.state_dir = Path(os.path.abspath(self.work_dir / Path(state_dir).expanduser()))
        self.trace_root = self.state_dir / "codex-runs"
        with safe_directory(self.trace_root, DirectoryMode.CREATE):
            pass
        self.timeout = timeout
        self.model = model.strip() if model is not None else None
        self.reasoning_effort = reasoning_effort
        self.resume_id = resume_id
        self.api_base_url = api_base_url
        self.api_key_env = api_key_env
        self.api_key = settings["api_key"]
        self.provider_headers = settings["http_headers"]
        self.model_options = settings["options"]
        catalog = self.model_options.get("model_catalog_json")
        self.model_catalog = Path(catalog) if catalog is not None else None
        self.codex_program = resolve_executable(codex_bin)
        self.agent = agent or AgentRuntime()
        self.read_only = tuple(Path(path).absolute() for path in read_only)
        self.evaluation = evaluation
        if self.evaluation is None and os.environ.get("KLINEAGE_SERVER"):
            from klineage.server import Client
            self.evaluation = Client()

        # Expose packaged backend and evaluation guidance for on-demand reading.
        skill_dir = self.work_dir / SKILL_DIRECTORY
        with safe_directory(skill_dir, DirectoryMode.CREATE):
            pass
        skill_paths = {"bench": BENCH_SKILL, **{backend.language: backend.skill_root for backend in backends()}}
        resources = [(source, skill_dir / name) for name, source in skill_paths.items()]
        for source, destination in resources:
            _link_resource(source, destination)

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
            raise CodexRunnerError(f"workspace resource already exists: {destination}")

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
    ) -> CodexRunResult:
        """Run one prompt and persist its trace, final response and stderr.

        Args:
            prompt: Nonempty text sent to the agent.
            run_id: Unique trace-directory identifier; generated when omitted.

        Returns:
            Final response and the JSONL trace path.

        Raises:
            CodexRunnerError: Launch, timeout, execution or reply reading fails.
            ValueError: The prompt or run identifier is invalid.
        """

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        resolved_run_id = run_id or operation_id(RunKind.CODEX)
        if not _RUN_ID_RE.fullmatch(resolved_run_id):
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
                raise CodexRunnerError(
                    f"could not launch Codex: {error}; trace: {trace_path}",
                    trace_path=trace_path,
                ) from error

            try:
                process.communicate(prompt, timeout=self.timeout)
            except subprocess.TimeoutExpired as error:
                raise CodexRunnerError(
                    f"Codex timed out after {self.timeout}s; trace: {trace_path}",
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
                raise CodexRunnerError(
                    f"Codex exited with exit code {process.returncode}{summary}; "
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
                final_file.seek(0)
                final_message = final_file.read()
                if not final_message.strip():
                    raise OSError("Codex produced no final response")
            except (OSError, UnicodeDecodeError) as error:
                raise CodexRunnerError(
                    f"Codex reply is unreadable; trace: {trace_path}",
                    trace_path=trace_path,
                ) from error
        return CodexRunResult(final_message=final_message, trace_path=trace_path)

    @contextmanager
    def execution(self, final_message_path: Path, run_dir: Path):
        """Share Codex state and expose declared inputs and scoped IPC endpoints."""
        home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser().resolve()
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
            "CODEX_HOME": str(home),
            LOG_ENV: str(run_dir),
            PROCESS_SCOPE_ENV: str(run_dir),
        }
        read_only = [
            *runtime_paths(self.agent.python, self.codex_program),
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
                key_env = self.api_key_env or self.DEFAULT_KEY_ENV
                bridge = stack.enter_context(ProviderBridge(
                    original_url, self.api_key, ipc / "provider.sock",
                    headers=self.provider_headers, directory=run_dir,
                ))
                self.api_base_url = bridge.url
                environment[key_env] = PROVIDER_KEY_PLACEHOLDER
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

    def command(self, final_message_path: Path) -> list[str]:
        command = [str(self.codex_program), "-a", "never"]
        for name, value in self.model_options.items():
            command.extend(("-c", f"{name}={to_json(value, indent=None)}"))
        if self.reasoning_effort is not None:
            command.extend(
                ("-c", f"model_reasoning_effort={to_json(self.reasoning_effort, indent=None)}")
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
        if self.resume_id is not None:
            command.extend(("resume", self.resume_id))
        if self.model is not None:
            command.extend(("--model", self.model))

        # `exec resume` accepts neither `-s` nor `-C`, so it runs with the session's
        # own sandbox and workspace; cwd still comes from Popen.
        if self.resume_id is not None:
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
        with safe_file(trace_path) as stream:
            lines = stream.read().splitlines()
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
            raise CodexRunnerError(f"workspace resource already exists: {destination}") from None


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

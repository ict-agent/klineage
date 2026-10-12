"""Run Codex against a Responses-compatible endpoint."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, get_args

from klineage.constants import STATE_DIRECTORY
from klineage.serialization import to_json

from .protocol import WireAPI
from .runner import PROVIDER_KEY_PLACEHOLDER, AgentRunner
from .sandbox import AgentRuntime

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
_REASONING_EFFORTS = frozenset(get_args(ReasoningEffort))
COMMON_EXEC_FLAGS = (
    "--strict-config", "--ignore-user-config", "--ignore-rules",
    "--skip-git-repo-check", "--json",
)
PLAIN_EXEC_FLAGS = ("--color", "never")
RESUME_SANDBOX_FLAGS = ("--dangerously-bypass-approvals-and-sandbox",)


class CodexRunner(AgentRunner):
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
    name = "Codex"
    kind = "codex"
    wire_api = WireAPI.RESPONSES
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
        effort = settings["reasoning_effort"]
        if effort is not None and effort not in _REASONING_EFFORTS:
            raise ValueError("invalid reasoning effort")
        super().__init__(
            work_dir, program=codex_bin, settings=settings, timeout=timeout,
            resume_id=resume_id, state_dir=state_dir, read_only=read_only,
            evaluation=evaluation, agent=agent,
        )
        if self.model is not None:
            self.model = self.model.strip()
        catalog = self.model_options.get("model_catalog_json")
        self.model_catalog = Path(catalog) if catalog is not None else None

    def session_home(self):
        home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex").expanduser().resolve()
        return home, {"CODEX_HOME": str(home)}

    def provider_env(self, url):
        return {self.api_key_env or self.DEFAULT_KEY_ENV: PROVIDER_KEY_PLACEHOLDER}

    def command(self, final_message_path: Path) -> list[str]:
        command = [str(self.program), "-a", "never"]
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


__all__ = ["CodexRunner", "ReasoningEffort"]

"""Run Claude Code against an Anthropic Messages-compatible endpoint."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from klineage.constants import STATE_DIRECTORY
from klineage.utils import safe_file

from .protocol import WireAPI
from .runner import PROVIDER_KEY_PLACEHOLDER, AgentRunner, RunnerError
from .sandbox import AgentRuntime

if TYPE_CHECKING:
    from klineage.server import Client

    from .provider import AnthropicAuth

CLAUDE_EFFORTS = frozenset(("low", "medium", "high", "xhigh", "max"))


class ClaudeCodeRunner(AgentRunner):
    """Run Claude Code in KLineage's Linux sandbox with host-only credentials.

    Uses the native ``claude`` executable in print/stream-json mode. Model and
    endpoint settings resolve from explicit arguments, then KLINEAGE variables,
    then ANTHROPIC variables. No host Claude settings or login are loaded.
    Sessions persist under ``state_dir/claude-home`` for retries; run records
    live under ``state_dir/agent-runs``. Skills and evaluation use the same
    workspace interfaces as CodexRunner.

    Args:
        work_dir: Dedicated writable workspace.
        claude_bin: Native Claude Code executable.
        model: Model ID; defaults to KLINEAGE_MODEL, then ANTHROPIC_MODEL.
        reasoning_effort: Optional low, medium, high, xhigh or max effort.
        timeout: Seconds per attempt; None disables the process deadline.
        resume_id: Claude session UUID from a previous trace.
        state_dir: Persistent state directory relative to work_dir, or absolute.
        api_base_url: Gateway root, optionally ending in /v1.
        api_key_env: Host credential variable; defaults to ANTHROPIC_API_KEY
            or ANTHROPIC_AUTH_TOKEN. Custom variable names default to x-api-key.
        api_key: Explicit host credential, overriding the environment.
        auth_mode: ``api-key`` for x-api-key or ``bearer`` for Authorization.
            Defaults to KLINEAGE_AUTH_MODE, then the selected key variable.
        read_only: Additional input mounts.
        evaluation: Bound evaluation service client.
        agent: Interpreter, dependencies and optional device access.
    """

    name = "Claude Code"
    kind = "claude"
    wire_api = WireAPI.MESSAGES

    def __init__(
        self, work_dir: str | Path, *, claude_bin: str | Path = "claude",
        model: str | None = None,
        reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
        timeout: float | None = None, resume_id: str | None = None,
        state_dir: str | Path = STATE_DIRECTORY,
        api_base_url: str | None = None, api_key_env: str | None = None,
        api_key: str | None = None, auth_mode: AnthropicAuth | str | None = None,
        read_only: Sequence[Path] = (), evaluation: Client | None = None,
        agent: AgentRuntime | None = None,
    ):
        from .provider import anthropic_settings

        settings = anthropic_settings(
            model=model, reasoning_effort=reasoning_effort, api_base_url=api_base_url,
            api_key_env=api_key_env, api_key=api_key, auth_mode=auth_mode,
        )
        if reasoning_effort is not None and reasoning_effort not in CLAUDE_EFFORTS:
            raise ValueError("invalid Claude Code reasoning effort")
        if resume_id is not None:
            UUID(resume_id)
        super().__init__(
            work_dir, program=claude_bin, settings=settings, timeout=timeout,
            resume_id=resume_id, state_dir=state_dir, read_only=read_only,
            evaluation=evaluation, agent=agent,
        )

    def session_home(self):
        home = self.state_dir / "claude-home"
        return home, {"CLAUDE_CONFIG_DIR": str(home)}

    def bridge_key(self):
        # provider_headers already contains the selected Anthropic auth scheme.
        return None

    def provider_env(self, url):
        values = {
            "ANTHROPIC_BASE_URL": url,
            "ANTHROPIC_API_KEY": PROVIDER_KEY_PLACEHOLDER,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_AUTOUPDATER": "1",
            "IS_SANDBOX": "1",
        }
        if self.model:
            values["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = self.model
        return values

    def command(self, final_message_path):
        command = [
            str(self.program), "--bare", "--print", "--verbose",
            "--output-format", "stream-json", "--input-format", "text",
            "--dangerously-skip-permissions", "--setting-sources", "",
            "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
        ]
        if self.model is not None:
            command.extend(("--model", self.model))
        if self.reasoning_effort is not None:
            command.extend(("--effort", self.reasoning_effort))
        if self.resume_id is not None:
            command.extend(("--resume", self.resume_id))
        return command

    def read_reply(self, final_file, trace_path):
        result = None
        with safe_file(trace_path) as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict) and event.get("type") == "result":
                    result = event
        if result is None:
            raise OSError("Claude Code produced no result event")
        if result.get("is_error") or result.get("subtype") != "success":
            raise RunnerError(
                f"Claude Code result failed: {result.get('subtype')}; trace: {trace_path}",
                trace_path=trace_path,
            )
        reply = result.get("result")
        if not isinstance(reply, str) or not reply.strip():
            raise OSError("Claude Code produced no final response")
        final_file.write(reply)
        final_file.flush()
        return reply

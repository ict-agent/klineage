"""Shared CLI options and runtime configuration."""

import argparse
import os
from pathlib import Path

from klineage.constants import MAX_RETRIES, TIMEOUT
from klineage.harness import AgentRuntime
from klineage.server import Client
from klineage.server.client import SERVER_ENV


def add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--timeout", type=int, default=TIMEOUT,
        help="Seconds per agent attempt and evaluation request",
    )
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument(
        "--server", metavar="ENDPOINT", help="Evaluation endpoint; defaults to KLINEAGE_SERVER"
    )
    parser.add_argument(
        "--problem-id", help="Registered task; optional when the server exposes one problem"
    )
    add_agent_options(parser)


def add_agent_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--agent-python", type=Path, help="Python runtime mounted in the agent sandbox"
    )
    parser.add_argument(
        "--agent-device-node",
        type=Path,
        action="append",
        default=[],
        help="Device node exposed to the agent for diagnostics",
    )
    parser.add_argument(
        "--agent-read-only",
        type=Path,
        action="append",
        default=[],
        help="Read-only agent toolchain or dependency path",
    )
    parser.add_argument(
        "--agent-env", action="append", default=[], help="Agent environment NAME=VALUE"
    )


def agent_runtime(values: dict) -> AgentRuntime | None:
    python = values.pop("agent_python", None)
    devices = values.pop("agent_device_node", ())
    read_only = values.pop("agent_read_only", ())
    assignments = values.pop("agent_env", ())
    if not (python or devices or read_only or assignments):
        return None
    environment = {}
    for assignment in assignments:
        name, separator, value = assignment.partition("=")
        if not separator or not name.isidentifier():
            raise ValueError("--agent-env requires NAME=VALUE")
        environment[name] = value
    options = {
        "device_nodes": devices,
        "read_only": read_only,
        "environment": environment,
    }
    if python is not None:
        options["python"] = python
    return AgentRuntime(**options)


def action_arguments(arguments: argparse.Namespace) -> dict:
    values = vars(arguments).copy()
    agent = agent_runtime(values)
    if agent is not None:
        values["agent"] = agent
    endpoint = values.pop("server")
    problem_id = values.pop("problem_id")
    if endpoint is None and problem_id is None and not os.environ.get(SERVER_ENV):
        return values
    values["evaluation"] = Client(
        endpoint=endpoint, problem_id=problem_id, timeout=values["timeout"],
    )
    return values

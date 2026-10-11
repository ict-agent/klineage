"""Run a fresh sandboxed agent against a separate device evaluation service."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from klineage.artifact import Kernel
from klineage.action import Apply
from klineage.artifact import Node, json_node, save_artifact, source_files
from klineage.artifact.problem import problem_reference
from klineage.backend import get_backend
from klineage.cli.common import add_agent_options, agent_runtime
from klineage.harness import AgentRuntime
from klineage.harness.process import STOP_GRACE_SECONDS, stop_process
from klineage.harness.provider import resolve_provider, runner_class, usage_proxy
from klineage.harness.sandbox import runtime_environment
from klineage.serialization import to_json
from klineage.server import Client
from klineage.server.protocol import DEFAULT_TIMEOUT
from klineage.utils import operation_id, write_json

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_BUDGET = 10 * 60
AUDIT_RESERVE = 2 * 60
SERVER_START_TIMEOUT = 60
READY_POLL = 0.1
MODEL = "DeepSeek-V4.1-Flash"
PROVIDER_URL = "https://aiping.cn/api/v1"
PROVIDER_KEY_ENV = "AIPING_API_KEY"


def remaining(deadline: float) -> float:
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("campaign budget exhausted")
    return seconds


@contextmanager
def server_process(args, root: Path, problem: Path, deadline: float):
    ready = root / "server-ready.json"
    environment = runtime_environment() | {
        "KLINEAGE_BACKEND": str(get_backend(args.backend).kind),
        "PYTHONPATH": str(PROJECT / "src"),
    }
    command = [
        str(args.server_python), "-m", "klineage.server",
        "--root", str(root / "server"), "--backend", args.backend,
        "--problem", str(problem),
        "--listen", "http://127.0.0.1:0", "--ready-file", str(ready),
        "--timeout", str(min(DEFAULT_TIMEOUT, remaining(deadline))),
    ]
    for device in args.device_node:
        command.extend(("--device-node", str(device)))
    timer = shutil.which("timeout")
    if timer is None:
        raise RuntimeError("campaign supervision requires GNU timeout")
    command = [
        timer, "--signal=TERM", f"--kill-after={STOP_GRACE_SECONDS}s",
        str(remaining(deadline)), *command,
    ]
    with (root / "server.log").open("w") as log:
        process = subprocess.Popen(
            command, env=environment, cwd=PROJECT, stdout=log, stderr=log,
            start_new_session=True,
        )
        try:
            startup = min(deadline, time.monotonic() + SERVER_START_TIMEOUT)
            while not ready.is_file():
                if process.poll() is not None:
                    raise RuntimeError(f"server exited with {process.returncode}; see {log.name}")
                remaining(startup)
                time.sleep(READY_POLL)
            state = json.loads(ready.read_text())
            client = Client(state["endpoint"])
            client.health()
            yield client, state["pid"]
        finally:
            stop_process(process)


def run(args):
    agent = getattr(args, "agent", None) or AgentRuntime()
    started = time.monotonic()
    started_at = datetime.now(UTC)
    deadline = started + args.budget
    generation_deadline = deadline - AUDIT_RESERVE
    root = args.output.resolve() / operation_id("sandbox-campaign")
    root.mkdir(parents=True, exist_ok=False)
    work = root / "agent"
    state = {
        "pid": os.getpid(), "root": str(root), "model": args.model,
        "budget_seconds": args.budget, "started_at": started_at.isoformat(),
        "deadline": (started_at + timedelta(seconds=args.budget)).isoformat(),
        "generation_deadline": (started_at + timedelta(seconds=args.budget - AUDIT_RESERVE)).isoformat(),
        "agent_python": str(agent.python), "server_python": str(args.server_python),
        "agent_device_nodes": [str(path) for path in agent.device_nodes],
        "fresh_session": True,
    }

    def status(phase, **fields):
        state.update(phase=phase, elapsed_seconds=round(time.monotonic() - started, 3), **fields)
        write_json(root / "status.json", state)
        print(to_json(state, indent=None), flush=True)

    status("prepare")
    hashes = {
        str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((PROJECT / "src").rglob("*"))
        if path.is_file() and path.suffix in (".py", ".md", ".j4", ".toml")
    }
    write_json(root / "source-hashes.json", hashes)

    def stop_on_signal(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop_on_signal)
    try:
        with server_process(args, root, args.problem.resolve(), deadline) as (client, server_pid):
            problem = client.problem()
            info = client.info()
            write_json(root / "server-info.json", info)
            backend = get_backend(problem.language, problem.platform)
            initial = backend.kernel_type(Node.directory(".", (
                json_node("metadata.json", {
                    "name": problem.name,
                    "problem": problem_reference(Path(info["problem_paths"][problem.name]), problem),
                    "validation": None, "compile_flags": [],
                }),
                Node.directory("submission"),
            )))
            provider = resolve_provider(
                args.model, args.base_url, key_env=PROVIDER_KEY_ENV,
            )
            with usage_proxy(provider, root / "usage") as proxied:
                base = runner_class(proxied, "high")

                class BudgetRunner(base):
                    def __call__(self, prompt, *, run_id=None):
                        self.timeout = remaining(generation_deadline)
                        return super().__call__(prompt, run_id=run_id)

                action = Apply(
                    initial, workdir=work, timeout=int(remaining(generation_deadline)),
                    max_retries=1, evaluation=client, agent=agent,
                )
                action.runner = BudgetRunner(
                    work, timeout=remaining(generation_deadline), evaluation=client, agent=agent,
                )
                action.prompt += f"""

This is a fresh, source-free campaign. Use only this task and workspace.
Read workspace/AGENTS.md and the relevant workspace/.agents/skills/ guidance.
Use `{agent.python}` in the configured agent sandbox. Its selected devices and
available tools, such as ncu, may be used directly for performance diagnosis.
Use klineage.server.Client.evaluate for authoritative correctness and timing.
When local diagnostics share a device with the server, run them sequentially;
never overlap local device work with server evaluations.
Implement computation in native {problem.language}; do not delegate the operation
to torch reference functions, vendor libraries or other existing implementations.
The fixed Problem, inputs and measurement policy belong to the server.
Deliver the best accepted artifact and retain its matching service result by
{state['generation_deadline']}. Check `date -u` before optional trials.
For this source-free start, first submit a native implementation with a simple
execution schedule. The client automatically appends complete results under
versions/0, versions/1, ... including failures. Use the remaining time to improve it.
Action uses the largest numbered version; do not save Kernels manually. To finish
with an earlier accepted best job, call client.result(best_job_id, work=Path.cwd())
to append it as the latest version. current_kernel/ remains trial source space.
Keep helpers, diagnostics, source backups, binaries and logs under workspace/.
Read server diagnostics and retain tool output that explains your changes.
"""
                status("generation", server_pid=server_pid, endpoint=client.endpoint)
                result = action.run()
            save_artifact(result, root / "action-result.json")
            kernel = result.part("kernel", Kernel)
            status("audit")
            audit = client.evaluate(kernel, root / "audit", timeout=remaining(deadline))
            if not audit.accepted:
                raise RuntimeError(f"independent audit failed: {audit.error or audit.validation}")
            if source_files(audit.kernel.find("submission")) != source_files(kernel.find("submission")):
                raise ValueError("audit used different sources")
            changed = [path for path, digest in hashes.items() if hashlib.sha256((PROJECT / path).read_bytes()).hexdigest() != digest]
            if changed:
                raise RuntimeError(f"framework changed during campaign: {changed}")
            status("completed", accepted=True, validation=audit.validation, audit_job=audit.job_id)
    except BaseException as error:
        phase = "budget_exhausted" if isinstance(error, TimeoutError) or time.monotonic() >= generation_deadline else "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        status(phase, accepted=False, error=f"{type(error).__name__}: {error}")
        if phase == "error":
            raise
    finally:
        status(state["phase"], finished_at=datetime.now(UTC).isoformat())
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", type=Path, default=PROJECT / "problems/fmha")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument("--server-python", type=Path, required=True)
    parser.add_argument("--device-node", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=PROJECT / "agent-workspace")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base-url", default=PROVIDER_URL)
    add_agent_options(parser)
    args = parser.parse_args()
    args.agent = agent_runtime(vars(args))
    if args.budget <= AUDIT_RESERVE:
        parser.error("budget must exceed the independent audit reserve")
    run(args)


if __name__ == "__main__":
    main()

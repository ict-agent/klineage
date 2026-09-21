#!/usr/bin/env python3
"""Resume a unit's Codex session after its batch slot ended.

A unit stops when the agent declares convergence or when ``--timeout`` fires.
This continues the recorded thread for the rest of its budget::

    scripts/ascend/run.sh resume --kernel kda --setting without_memory

The thread id is read from ``trace.jsonl``, the session is routed through a new
usage proxy onto the same ``events.jsonl``, and ``result.json`` is rewritten
with the cumulative counters.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from klineage.logging import UsageProxy

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TIMEOUT_DEFAULT = 1800
PROMPT = """Continue the same task: keep optimizing the kernel you already have.

- Budget: {minutes} more minutes, then this session is killed. Run `date` before each
  step and keep iterating while more than ~10 minutes remain. Do not stop early to
  report; only gated versions count.
- Gate every change with the fixed command in `AGENTS.md`, and keep the best bundle in
  `submission/`.

End your last message with the final measured latency.
"""


def load_batch():
    """Import the batch runner for its paths, provider config and proxy."""

    spec = importlib.util.spec_from_file_location("ascend_batch", HERE / "batch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def thread_id(trace: Path) -> str:
    for line in trace.read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("{"):
            continue
        record = json.loads(line)
        if record.get("type") == "thread.started":
            return record["thread_id"]
    raise SystemExit(f"no thread id in {trace}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--setting", required=True, choices=("without_memory", "with_memory"))
    parser.add_argument("--timeout", type=int, default=TIMEOUT_DEFAULT, help="seconds to allow")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--run-root", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    options = parse_args(argv)
    batch = load_batch()
    root = (options.run_root or batch.RUN_ROOT).expanduser().resolve()
    out = root / options.kernel / options.setting
    work = out / "work"
    if not work.is_dir():
        raise SystemExit(f"no workspace at {work}")

    session = thread_id(out / "trace.jsonl")
    provider, upstream = batch.provider_settings()
    stats = work / ".klineage" / "stats"
    started = time.monotonic()
    print(f"resume {options.kernel}/{options.setting} session {session} "
          f"for {options.timeout}s", flush=True)

    with UsageProxy(upstream, directory=stats) as proxy:
        command = [
            options.codex_bin, "exec", "resume", session,
            "--json", "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "--output-last-message", str(out / "final_message.txt"),
            "-c", f"model_providers.{provider}.base_url={proxy.url}",
            "-",
        ]
        with (out / "trace.jsonl").open("a", encoding="utf-8") as trace, \
             (out / "stderr.log").open("a", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                command, cwd=work, stdin=subprocess.PIPE, stdout=trace, stderr=stderr,
                text=True, start_new_session=True,
            )
            try:
                process.communicate(PROMPT.format(minutes=options.timeout // 60),
                                    timeout=options.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.communicate()

    batch.package(out, work)
    record = json.loads((out / "result.json").read_text(encoding="utf-8"))
    record.update(batch.events_counts(work / ".klineage" / "stats" / "events.jsonl"))
    record["versions"] = len(list((work / "versions").glob("version*")))
    record["duration_s"] = round(record.get("duration_s", 0) + time.monotonic() - started, 1)
    record["resumed_at"] = datetime.now(UTC).isoformat()
    record["finished_at"] = datetime.now(UTC).isoformat()
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: record.get(key) for key in
                      ("kernel", "setting", "status", "versions", "duration_s")}), flush=True)


if __name__ == "__main__":
    main()

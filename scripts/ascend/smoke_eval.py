"""Smoke-test klineage.evaluate on this Ascend host with a python bundle.

Validates the full gate the agent will use: problem load, NPU inputs, torch
reference, candidate execution, timing, event logging, and kernel snapshot.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    if args.device:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = args.device
    os.environ.setdefault("KLINEAGE_BACKEND", "ascend")
    os.environ.setdefault("ASCEND_ARCH", "Ascend910B1")
    sys.path.insert(0, str(args.repo / "src"))

    from klineage.artifact.kernel import Kernel
    from klineage.artifact.problem import ProblemSpec, load_trace
    from klineage.backend import detect_backend
    from klineage.harness.eval import evaluate
    from klineage.logging import LOG_ENV

    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    os.environ[LOG_ENV] = str(work / ".klineage" / "stats")

    # Tiny GEMM: the torch reference is short and runs on any device.
    module = load_trace(args.repo / "problems" / "definitions" / "gemm.json")
    module.workload["axes"] = {"M": 128, "N": 128, "K": 128}
    backend = detect_backend()
    problem = ProblemSpec(
        name="smoke-gemm",
        definition=module.definition,
        workload=module.workload,
        language=backend.language,
        platform=backend.platform(),
    )
    sources = {
        "config.toml": (
            "[solution]\n"
            'name = "smoke-gemm"\n'
            'definition = "gemm"\n'
            'author = "klineage"\n\n'
            "[build]\n"
            'language = "python"\n'
            'entry_point = "kernel.py::kernel"\n'
            "destination_passing_style = true\n"
        ),
        "solution/kernel.py": (
            "import torch\n\n\n"
            "def kernel(x, weight, output):\n"
            "    torch.matmul(x, weight.transpose(0, 1), out=output)\n"
        ),
    }
    kernel = Kernel("smoke-gemm", problem, sources)
    result = evaluate(kernel, work)
    print(json.dumps(result.to_dict(), indent=2))

    events = work / ".klineage" / "stats" / "events.jsonl"
    versions = work / ".klineage" / "versions"
    names = sorted(path.name for path in versions.iterdir()) if versions.is_dir() else []
    print("events:", events.is_file(), "versions:", names)
    if not result.correctness_passed:
        raise SystemExit("smoke failed: correctness gate did not pass")
    if not events.is_file() or not names:
        raise SystemExit("smoke failed: no event or snapshot recorded")
    print("smoke ok, latency_ms:", result.latency_ms)


if __name__ == "__main__":
    main()

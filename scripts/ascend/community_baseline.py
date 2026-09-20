"""Time a community Ascend implementation of the problem with the harness timer.

Runs inside the container, in the workspace that holds the problem::

    python community_baseline.py --repo <checkout> --problems <dir> [--scratch <dir>]

The candidate is vllm-ascend's triton chunk_kda, called with the gate the
problem definition folds into the recurrence. It is evaluated like any other
kernel, so correctness and latency come from the same protocol as the runs and
the report can quote a baseline that is not the naive torch reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: vllm-ascend's chunked kernel takes the log decay and the sigmoided gain, and
#: normalizes q/k in kernel; the definition supplies raw values.
SOURCE = '''"""vllm-ascend triton chunk_kda on the NPU."""

import torch
from vllm_ascend.ops.triton.kda.kda import chunk_kda


def run(q, k, v, g, beta, scale, a_log, dt_bias, lower_bound, initial_state):
    gate = lower_bound.float() * torch.sigmoid(
        a_log.float().exp()[:, None] * (g.float() + dt_bias.float())
    )
    return chunk_kda(
        q=q,
        k=k,
        v=v,
        g=gate.to(q.dtype),
        beta=beta.sigmoid(),
        scale=float(scale),
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )


def kernel(*args):
    return run(*args)
'''
CONFIG = '''[solution]
name = "{name}-community"
definition = "{name}"
author = "klineage"

[build]
language = "python"
entry_point = "kernel.py::kernel"
'''


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="klineage checkout")
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, default=Path("/tmp/klineage-community"))
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(args.repo / "src"))
    from klineage.artifact.kernel import Kernel
    from klineage.artifact.problem import ProblemSpec
    from klineage.backend import detect_backend
    from klineage.harness.eval import evaluate

    definitions = sorted((args.problems / "definitions").glob("*.json"))
    if len(definitions) != 1:
        raise SystemExit(f"expected one problem definition under {args.problems}")
    definition = json.loads(definitions[0].read_text(encoding="utf-8"))
    name = definition["name"]
    workload_path = args.problems / "workloads" / definitions[0].with_suffix(".jsonl").name
    workload = json.loads(workload_path.read_text(encoding="utf-8").splitlines()[0])["workload"]

    backend = detect_backend()
    problem = ProblemSpec(name=name, definition=definition, workload=workload,
                          language=backend.language, platform=backend.platform())
    sources = {"config.toml": CONFIG.format(name=name), "solution/kernel.py": SOURCE}
    args.scratch.mkdir(parents=True, exist_ok=True)
    result = evaluate(Kernel(name, problem, sources), args.scratch, timeout=args.timeout)
    print(json.dumps({
        "community_ms": result.latency_ms,
        "compile_passed": result.compile_passed,
        "correctness_passed": result.correctness_passed,
        "profile_passed": result.profile_passed,
    }))


if __name__ == "__main__":
    main()

"""Evaluate one kernel workspace on the host NPU (runs inside the container).

Called by ``scripts/ascend/eval.py`` on the Mac; not meant for direct use.
Prints one line of JSON: the ValidationResult, or the platform in --probe mode.
``--baseline`` times the problem's torch reference instead of the candidate:
the task's Baseline column, measured with the same harness timer so the
speedup ratio compares like with like.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

#: This file lives at `<project root>/scripts/ascend/`, so the checkout is two
#: levels up; the SSH caller always runs it from there.
REPO_DEFAULT = Path(__file__).resolve().parents[2]

BASELINE_CONFIG = """[solution]
name = "{name}-baseline"
definition = "{name}"
author = "klineage"

[build]
language = "python"
entry_point = "kernel.py::kernel"
"""

#: The definition's own torch reference, run on the NPU: torch-npu is the
#: community implementation the task names as a baseline.
BASELINE_SOURCE = '''"""{name}: torch reference executed on the NPU."""

{reference}

def kernel(*args):
    return run(*args)
'''


#: The gate evaluates in a worker process, so the launch probe cannot be
#: installed here: it rides in on PYTHONPATH as sitecustomize and reports the
#: count through a file when that worker exits.
#: Both scratch directories live in the container's /tmp: the container runs as
#: root, and anything it writes under the run directory blocks the next rsync.
SCRATCH_ROOT = Path("/tmp")
PROBE_DIR = "klineage-probe"
COUNT_ENV = "KLINEAGE_LAUNCH_COUNT"
PROBE_SOURCE = '''"""Count triton kernel launches in the gate's evaluator worker."""

import atexit
import os
import pathlib

destination = os.environ.get("%s")
if destination:
    from triton.runtime.jit import JITFunction

    original = JITFunction.run
    state = {"count": 0}

    def counted(instance, *args, **kwargs):
        state["count"] += 1
        return original(instance, *args, **kwargs)

    JITFunction.run = counted

    @atexit.register
    def flush():
        pathlib.Path(destination).write_text(str(state["count"]), encoding="utf-8")
''' % COUNT_ENV


def install_probe(work: Path) -> Path:
    """Expose the launch probe to every worker the gate spawns."""

    directory = SCRATCH_ROOT / PROBE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sitecustomize.py").write_text(PROBE_SOURCE, encoding="utf-8")
    count = directory / "launches"
    count.unlink(missing_ok=True)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(directory), os.environ.get("PYTHONPATH")) if value
    )
    os.environ[COUNT_ENV] = str(count)
    return count


def launched_kernels(count: Path) -> int | None:
    """Triton launches the worker made, or None when the probe never reported."""

    if not count.is_file():
        return None
    try:
        return int(count.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def problem_files(problems: Path) -> tuple[dict, dict]:
    """The single problem definition under ``problems/`` and its first workload."""

    definitions = sorted((problems / "definitions").glob("*.json"))
    if len(definitions) != 1:
        raise SystemExit(f"expected one problem definition under {problems}")
    definition = json.loads(definitions[0].read_text(encoding="utf-8"))
    workloads = problems / "workloads" / definitions[0].with_suffix(".jsonl").name
    workload = json.loads(workloads.read_text(encoding="utf-8").splitlines()[0])["workload"]
    return definition, workload


def measure_baseline(args: argparse.Namespace, backend) -> None:
    """Time the problem's torch reference as a Kernel, like any candidate."""

    from klineage.artifact.kernel import Kernel
    from klineage.artifact.problem import ProblemSpec
    from klineage.harness.eval import evaluate

    definition, workload = problem_files(args.problems)
    name = definition["name"]
    problem = ProblemSpec(
        name=name,
        definition=definition,
        workload=workload,
        language=backend.language,
        platform=backend.platform(),
    )
    sources = {
        "config.toml": BASELINE_CONFIG.format(name=name),
        "solution/kernel.py": BASELINE_SOURCE.format(
            name=name, reference=definition["reference"]
        ),
    }
    scratch = args.scratch or args.problems.parent / ".klineage-baseline"
    scratch.mkdir(parents=True, exist_ok=True)
    result = evaluate(Kernel(name, problem, sources), scratch, timeout=args.timeout)
    print(json.dumps({
        "baseline_ms": result.latency_ms,
        "compile_passed": result.compile_passed,
        "correctness_passed": result.correctness_passed,
        "profile_passed": result.profile_passed,
    }))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--probe", action="store_true", help="print the platform and exit")
    parser.add_argument("--baseline", action="store_true",
                        help="time the problem's torch reference instead")
    parser.add_argument("--problems", type=Path, default=Path.cwd(),
                        help="directory holding definitions/ and workloads/")
    parser.add_argument("--scratch", type=Path, help="build directory for --baseline")
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(args.repo / "src"))
    from klineage.backend import detect_backend

    backend = detect_backend()
    if args.probe:
        print(backend.platform())
        return
    if args.baseline:
        measure_baseline(args, backend)
        return
    if args.work is None:
        raise SystemExit("--work is required")

    from klineage.artifact.kernel import load_kernel
    from klineage.harness.eval import evaluate
    from klineage.harness.timing import timing_policy

    work = Path(args.work).resolve()
    count = install_probe(work)
    kernel = load_kernel(work)
    result = evaluate(kernel, work, timeout=args.timeout)
    print(json.dumps(result.to_dict()))
    # The task asks for the benchmark protocol next to the number: record the
    # sampling policy and which timer produced it.
    print(json.dumps({"gate": {
        "triton_launches": launched_kernels(count),
        "timing": {
            "backend": backend.timing_backend,
            "policy": timing_policy(backend).to_dict(),
        },
    }}))


if __name__ == "__main__":
    main()

"""Run the Astra suite, then Luna, with a 12-hour limit per job."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import signal
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

from _process import _compress, _execute, _git, _save_rollout, _thread_id

_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT = _ROOT / "baseline"
_WORK = _ROOT / "agent-workspace" / "baselines"
_UPSTREAM = _ROOT / "agent-workspace" / "upstream"
_BUDGET_SECONDS = 12 * 60 * 60
_FINAL_SECONDS = 5 * 60
_PROBLEMS = ("gemm", "conv2d", "fmha", "gdn", "topk")
_METHODS = ("KDA", "codex", "AKO4ALL")
_MODELS = {"astra": ("gpt-6-astra", "xhigh"), "luna": ("gpt-5.6-luna", "max")}
_REVISIONS = {
    "KDA": ("https://github.com/mit-han-lab/kernel-design-agents.git", "9f5ee5d5cd623c2d69c4ff71c8a5155edcb8c43b"),
    "AKO4ALL": ("https://github.com/TongmingLAIC/AKO4ALL.git", "8d3065a7588c124182fa0fe0b3e589879641e1b7"),
}
_ARCHIVE_EXCLUDES = {".git", "__pycache__", ".venv", "build", ".cache"}
_GPU_MARKERS = ("triton", "__global__", "cutlass", "cuda.tile")


def _write(path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _jobs():
    # Complete every Astra job before starting the Luna experiment.
    for model in _MODELS:
        for method in _METHODS:
            for problem in _PROBLEMS:
                yield method, model, problem


def _prepare(method, problem, work) -> None:
    work.mkdir(parents=True, exist_ok=True)
    reference = work / "reference.py"
    if reference.exists():
        return
    shutil.copyfile(_ROOT / "problems" / problem / "reference.py", reference)
    (work / "solution").mkdir()
    seed = "from reference import torch_ref\n\ndef kernel(**inputs):\n    return torch_ref(**inputs)"
    if problem == "topk":
        seed += ".indices.to(dtype=__import__('torch').int32)"
    (work / "solution" / "kernel.py").write_text(seed + "\n")
    (work / ".gitignore").write_text("__pycache__/\n*.pyc\nbuild/\n*.so\n*.ncu-rep\n")
    _git(work, "init", "-q")
    _git(work, "add", ".")
    _git(work, "commit", "-qm", "Seed the paper workload")
    if method in _REVISIONS:
        upstream = _UPSTREAM / method
        if _git(upstream, "rev-parse", "HEAD") != _REVISIONS[method][1]:
            raise RuntimeError(f"upstream revision differs: {method}")


def _bench_command(problem, work):
    return [
        sys.executable, str(_OUTPUT / "evaluate.py"),
        "--problem", problem, "--reference", str(work / "reference.py"),
    ]


def _prompt(method, model, problem, work, deadline) -> str:
    command = shlex.join(_bench_command(problem, work))
    contract = f"""Generate and optimize a custom GPU kernel for {problem}.
Use { _MODELS[model][0] } at { _MODELS[model][1] } reasoning for this entire run.
Workspace: {work}. Target: the visible NVIDIA H100 PCIe (SM90).
Deadline: {datetime.fromtimestamp(deadline, UTC).isoformat()}.
The total budget is 12 hours, including compilation, experiments and validation.
Continue optimization until the deadline. Reserve the final five minutes for
restoring and validating the fastest correct candidate. Never reset the budget.

Read reference.py. Its default make_inputs() shapes and torch_ref() define the
contract; all inputs are keyword arguments. Write solution/kernel.py exposing
kernel(**inputs). CUDA C++/PTX, Triton and CuTe DSL are allowed. Python may handle
allocation, layout views, compilation and launches; replace the initial PyTorch
compute with custom GPU code. Do not call torch_ref, torch compute operations,
cuBLAS/cuDNN, FlashInfer operators, or another prebuilt operator in the final
candidate. Implement all outputs with the reference's shapes/dtypes, except
Top-K returns only int32 indices, with unique indices and unspecified order.
Do not change the reference, evaluator, shapes, input generation or tolerances.
Do not cache computed outputs, modify inputs, or bypass timing using streams.
Keep all implementation dependencies under solution/ or record package versions.
Do not read other baseline workspaces, kernels or traces. Do not delegate kernel
generation to another model. You may research techniques and official sources.

Validation and performance command (choose a new label every time):
{command} --solution solution --output runs/LABEL.json
Each call saves the exact evaluated sources under runs/LABEL/solution/.
It checks seeds 17, 43 and 101, then times with FlashInfer CUPTI (10 warmups,
50 repeats, 3 trials, cold L2), then tests new values at reused input addresses.
Fixed (rtol, atol): GEMM (.02,.02), Conv2d (.01,.01), FMHA (.01,.005),
GDN (.01,.001); Top-K selected values must match exactly. No dtype relaxation.
Use --mode full for a final verdict including reference timing and speedup.
Use --mode reference --output runs/reference.json to measure the original.
Rank candidates by median kernel latency; only promote correct candidates.
Failed experiments still belong in the record. Do not stop after the first win.

Save notes, benchmark results, profiling evidence and all candidates locally.
Before the deadline copy the fastest passing runs/LABEL/solution/ to solution/,
then run --mode full. Write a concise REPORT.md with the winning measurement,
techniques tried, rejected candidates, dependencies and remaining limitations.
The outer runner saves traces and commits/pushes each finished job to klineage.
Local commits are allowed; do not push or change the outer repository.
"""
    if method == "KDA":
        return contract + f"""
Use Kernel Design Agents from {_UPSTREAM / method}.
Read docs/agent-flow.md and prompts/basic-flow.md there. Follow that workflow:
write docs/draft.md, convert it to docs/plan.md before implementation, then
implement one candidate at a time, validate, record parent links and promotion
decisions in candidates.jsonl and benchmark.csv, and iterate.
Use its skills/KernelWiki/SKILL.md for relevant H100 techniques and
skills/ncu-report-skill/SKILL.md for profiling; this GPU is H100, not B200.
Humanize is optional in this pinned upstream; write the executable plan yourself.
"""
    if method == "AKO4ALL":
        return contract + f"""
Use AKO4ALL: read {_UPSTREAM / method / 'SKILL.md'} and follow its protocol.
The seed kernel is solution/kernel.py; reference and inputs are in reference.py.
Use the custom evaluator command above, not the default KernelBench evaluator.
Bootstrap HINTS.md, ITERATIONS.md and bench-wrapper.sh from that upstream.
Persist this contract in HINTS.md. Generate scripts/bench.sh with the provided
command and a unique output per label. Profile, modify, benchmark, log and commit
each iteration as specified. The 12-hour deadline overrides early stopping rules.
"""
    return contract + "\nUse direct Codex. Do not load KDA, AKO4ALL or their skills.\n"


def _codex_command(model, final, session=None):
    name, effort = _MODELS[model]
    command = ["codex", "exec"]
    if session:
        command.append("resume")
    command += [
        "--ignore-user-config", "--skip-git-repo-check", "--json",
        "-m", name, "-c", f'model_reasoning_effort="{effort}"',
        "-c", 'approval_policy="never"', "-c", 'sandbox_mode="danger-full-access"',
        "-o", str(final),
    ]
    if session:
        command.append(session)
    return command + ["-"]


def _best_candidate(work):
    candidates = []
    for path in (work / "runs").glob("*.json"):
        result = json.loads(path.read_text())
        if result.get("status") != "passed":
            continue
        source = path.with_suffix("") / "solution"
        if not (source / "kernel.py").is_file():
            continue
        sources = result.get("sources", {})
        text = "\n".join(item.read_text() for item in source.rglob("*.py"))
        text += "\n".join(item.read_text() for item in source.rglob("*.cu"))
        if "torch_ref" in text or not any(marker in text for marker in _GPU_MARKERS):
            continue
        if not sources or any(
            hashlib.sha256((source / name).read_bytes()).hexdigest() != digest
            for name, digest in sources.items()
        ):
            raise RuntimeError(f"evaluated sources changed: {source}")
        candidates.append((result["candidate"]["median_ms"], source))
    if not candidates:
        raise RuntimeError("no passing kernel candidate")
    return min(candidates)[1]


def _archive(work, destination) -> None:
    def include(info):
        parts = Path(info.name).parts
        if any(part in _ARCHIVE_EXCLUDES for part in parts):
            return None
        if Path(info.name).suffix in {".so", ".o", ".pyc", ".ncu-rep"}:
            return None
        return info

    with tarfile.open(destination, "w:gz") as archive:
        archive.add(work, arcname="workspace", filter=include)


def _finish(problem, work, output, deadline):
    best = _best_candidate(work)
    destination = output / "kernel"
    shutil.copytree(best, destination, ignore=shutil.ignore_patterns("__pycache__"))
    command = _bench_command(problem, work) + [
        "--solution", str(destination), "--output", str(output / "evaluation.json"),
        "--mode", "full",
    ]
    remaining = deadline - time.time()
    if remaining <= 0:
        raise RuntimeError("budget expired before final verification")
    code = _execute(command, work, output / "evaluation.log", remaining)
    if code:
        raise RuntimeError(f"final evaluation failed: exit {code}")
    return json.loads((output / "evaluation.json").read_text())


def _publish(output, method, model, problem) -> None:
    _git(_ROOT, "add", "--", str(output.relative_to(_ROOT)))
    title = f"Save {method} {model} {problem} experiment"
    _git(_ROOT, "commit", "-m", title)
    _git(_ROOT, "push", "origin", "HEAD")


def _run_job(method, model, problem) -> None:
    output = _OUTPUT / method / model / problem
    work = _WORK / method / model / problem
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "status.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["status"] == "validated":
            return
        raise RuntimeError(f"existing unfinished job requires inspection: {output}")

    _prepare(method, problem, work)
    start = time.time()
    deadline = start + _BUDGET_SECONDS
    state = {
        "method": method, "model": _MODELS[model][0], "effort": _MODELS[model][1],
        "problem": problem, "budget_seconds": _BUDGET_SECONDS,
        "started_at": start, "deadline": deadline, "status": "running",
        "runner_pid": os.getpid(), "framework": _REVISIONS.get(method),
        "repository_commit": _git(_ROOT, "rev-parse", "HEAD"),
    }
    _write(state_path, state)
    prompt = _prompt(method, model, problem, work, deadline)
    (output / "prompt.txt").write_text(prompt)
    session = None
    turn = 0
    try:
        while time.time() < deadline - _FINAL_SECONDS:
            turn += 1
            trace = output / "traces" / f"{turn:04d}.jsonl"
            final = trace.with_suffix(".final.txt")
            command = _codex_command(model, final, session)
            remaining = deadline - _FINAL_SECONDS - time.time()
            code = _execute(command, work, trace, remaining, prompt)
            session = session or _thread_id(trace)
            _compress(trace)
            if session:
                _save_rollout(session, output / "rollout.jsonl.gz")
            state.update(turns=turn, session=session, last_exit=code)
            _write(state_path, state)
            if code and code != -signal.SIGTERM:
                raise RuntimeError(f"Codex failed with exit {code}; inspect traces")
            prompt = (
                "Continue the same experiment. Read the saved task contract and notes. "
                f"The original deadline remains {datetime.fromtimestamp(deadline, UTC).isoformat()}. "
                "Time remains: investigate another optimization, validate and keep the best. "
                "Do not reset the budget or switch models."
            )
        state["evaluation"] = _finish(problem, work, output, deadline)
        state["status"] = "validated"
    except Exception as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        state["elapsed_seconds"] = time.time() - start
        _archive(work, output / "workspace.tar.gz")
        _write(state_path, state)
        _publish(output, method, model, problem)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", nargs=3, metavar=("METHOD", "MODEL", "PROBLEM"))
    args = parser.parse_args()
    os.environ["PYTHONPATH"] = str(_ROOT / "src")
    _WORK.mkdir(parents=True, exist_ok=True)
    with (_WORK / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = [tuple(args.job)] if args.job else _jobs()
        for method, model, problem in jobs:
            if method not in _METHODS or model not in _MODELS or problem not in _PROBLEMS:
                parser.error("unknown method, model or problem")
            print(f"Starting {method}/{model}/{problem}", flush=True)
            _run_job(method, model, problem)


if __name__ == "__main__":
    _main()

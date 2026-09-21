"""Paired profiler measurements: core kernel and all device kernels."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import statistics
from datetime import datetime, timezone

import torch
import torch_npu

from benchmark import (PAPER_SHAPE, PAPER_FILTERS, ROOT, inputs, cpu_reference,
                       compare, inventory, check_inventory, digest, summarize)
from src import baseline, conv2d

CORE_NAME = "aclnnConvolution_Conv2dWithFlag_Conv2D"
CANDIDATE_NAMES = ("_prepare", "_conv_cube", "_crop")


def groups(values, count):
    return [values[i:i + count] for i in range(0, len(values), count)]


def analyze(rows, rounds, calls, warmup):
    block_calls = warmup + calls
    total = rounds * block_calls
    candidate = {name: [float(r["Duration(us)"]) for r in rows if r["Name"] == name]
                 for name in CANDIDATE_NAMES}
    official = [r for r in rows if r["Name"].startswith("aclnn")]
    if any(not r["Name"].startswith("aclnn") and r["Name"] not in CANDIDATE_NAMES for r in rows):
        raise RuntimeError("Unattributed device task in profiler output")
    core = [float(r["Duration(us)"]) for r in official if r["Name"] == CORE_NAME]
    if len(core) != total or any(len(values) != total for values in candidate.values()):
        raise RuntimeError("Profiler omitted or duplicated expected kernel tasks")
    if len(official) % total:
        raise RuntimeError("Baseline device task count is not stable")
    tasks_per_call = len(official) // total
    official_calls = groups(official, tasks_per_call)
    if any(sum(r["Name"] == CORE_NAME for r in call) != 1 for call in official_calls):
        raise RuntimeError("Cannot partition baseline device tasks by invocation")
    official_sum = [sum(float(r["Duration(us)"]) for r in call) for call in official_calls]
    candidate_sum = [sum(parts) for parts in zip(*candidate.values())]
    all_samples = {"baseline_core": core, **candidate,
                   "baseline_all_kernels": official_sum, "candidate_all_kernels": candidate_sum}
    metrics = {}
    for name, values in all_samples.items():
        blocks = [block[warmup:] for block in groups(values, block_calls)]
        means = [statistics.fmean(block) for block in blocks]
        metrics[name] = {**summarize(means), "per_call_us": [v for block in blocks for v in block]}
    speedup = metrics["baseline_core"]["mean_us"] / metrics["_conv_cube"]["mean_us"]
    paired = [a / b for a, b in zip(metrics["baseline_core"]["samples_us"],
                                   metrics["_conv_cube"]["samples_us"])]
    return {"metrics": metrics, "core_speedup": speedup, "paired_core_speedups": paired,
            "all_device_kernels_speedup": metrics["baseline_all_kernels"]["mean_us"]
                / metrics["candidate_all_kernels"]["mean_us"],
            "baseline_tasks_per_call": tasks_per_call}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--physical-device", type=int)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "kernels.json")
    args = parser.parse_args()
    if min(args.rounds, args.calls, args.warmup) <= 0:
        parser.error("Counts must be positive")
    if args.physical_device is None:
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        args.physical_device = int(visible.split(",")[args.device]) if visible else args.device
    before = inventory()
    check_inventory(before, args.physical_device, [])
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    xc, wc = inputs(9182)
    oracle = cpu_reference(xc, wc)
    x, w = xc.to(f"npu:{args.device}"), wc.to(f"npu:{args.device}")
    funcs = {"baseline": lambda: baseline.run(x, w), "candidate": lambda: conv2d.run(x, w)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace_root = args.output.parent / "profiler" / args.output.stem
    if trace_root.exists():
        raise FileExistsError(f"Use a new output name; profiler trace already exists: {trace_root}")
    snapshots = [before]
    checks = {}
    with torch.inference_mode():
        for name, fn in funcs.items():
            out = fn()
            checks[name] = compare(out, oracle)
            for _ in range(args.warmup):
                fn()
        torch.npu.synchronize()
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace_root)),
            record_shapes=True,
        ):
            for round_id in range(args.rounds):
                snapshots.append(inventory())
                check_inventory(snapshots[-1], args.physical_device, [os.getpid()])
                order = list(funcs) if round_id % 2 == 0 else list(reversed(funcs))
                for name in order:
                    # Occupancy queries leave the device idle; restore steady clocks.
                    with torch.profiler.record_function(f"warmup_{round_id}_{name}"):
                        for _ in range(args.warmup):
                            funcs[name]()
                    torch.npu.synchronize()
                    with torch.profiler.record_function(f"round_{round_id}_{name}"):
                        for _ in range(args.calls):
                            funcs[name]()
                    torch.npu.synchronize()
                snapshots.append(inventory())
                check_inventory(snapshots[-1], args.physical_device, [os.getpid()])
                print(f"Profiler round {round_id + 1}/{args.rounds}", flush=True)
    csv_files = list(trace_root.glob("*/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
    if len(csv_files) != 1:
        raise RuntimeError("Expected exactly one parsed device task CSV")
    csv_path = args.output.with_suffix(".csv")
    shutil.copyfile(csv_files[0], csv_path)
    with csv_path.open(newline="") as stream:
        result = analyze(list(csv.DictReader(stream)), args.rounds, args.calls, args.warmup)
    sources = [ROOT / "benchmark_kernels.py", ROOT / "benchmark.py", *sorted((ROOT / "src").glob("*.py"))]
    report = {"status": "pass", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "config": {**vars(args), "output": str(args.output), "shape": PAPER_SHAPE,
                         "filters": PAPER_FILTERS, "dtype": "float16", "cache": "warm; no flush"},
              "environment": {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
                              "device": torch.npu.get_device_name(args.device)},
              "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
              "correctness": checks, "inventories": snapshots, **result,
              "protocol": "Arithmetic mean of all measured calls, via equal-count round means; speedup is ratio of means. Median/min/max describe round means. Alternate order; warm up each implementation immediately before every measured block. Raw CSV includes excluded warmup calls; retain all measured calls.",
              "scope": {"core": "CANN Conv2D vs _conv_cube; both exclude preparation and output reformatting.",
                        "all_kernels": "Sum all device tasks per full ABI invocation: baseline weight/output conversions and CANN internal formats; candidate padding/convolution/crop; excludes gaps/Python.",
                        "input": "Both implementations consume the same NHWC input and flattened HWCF weights and return contiguous NHWC output."}}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"core_speedup": report["core_speedup"]}), flush=True)


if __name__ == "__main__":
    main()

"""Correctness and synchronized host latency for the public Conv2d ABI."""

import argparse
import hashlib
import json
import os
import re
from importlib import metadata
from pathlib import Path
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone

import torch
import torch_npu
import triton

from src import baseline, conv2d

PAPER_SHAPE = conv2d.INPUT_SHAPE
PAPER_FILTERS = conv2d.WEIGHT_SHAPE[1]
ATOL = 1e-2
RTOL = 1e-2
ROOT = Path(__file__).resolve().parent


def inventory():
    proc = subprocess.run(["npu-smi", "info"], text=True, capture_output=True, check=True)
    return proc.stdout


def check_inventory(raw, physical, expected_pids):
    match = re.search(rf"^\|\s+{physical}\s+910\S*\s*\|\s*(\w+)", raw, re.M)
    if not match or match[1] != "OK":
        raise RuntimeError(f"Physical NPU {physical} is absent or unhealthy")
    pids = sorted(int(pid) for pid in re.findall(
        rf"^\|\s+{physical}\s+0\s*\|\s*(\d+)\s*\|", raw, re.M))
    if not pids and not re.search(rf"No running processes found in NPU\s+{physical}\b", raw):
        raise RuntimeError("Cannot determine device occupancy")
    if pids != expected_pids:
        raise RuntimeError(f"NPU occupancy changed: {pids}; expected {expected_pids}")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_reference(x, weight):
    return baseline.run(x.float(), weight.float())


def compare(actual, expected):
    actual = actual.detach().cpu().float()
    expected = expected.detach().cpu().float()
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    delta = (actual - expected).abs()
    return {"max_abs": delta.max().item(), "mean_abs": delta.mean().item(),
            "max_normalized": (delta / (ATOL + RTOL * expected.abs())).max().item()}


def inputs(seed, mode="random"):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(PAPER_SHAPE, generator=generator, dtype=torch.float16)
    weight = torch.randn(conv2d.WEIGHT_SHAPE, generator=generator, dtype=torch.float16)
    if mode == "zero":
        x.zero_()
    if mode == "ones":
        x.fill_(1)
        weight.fill_(1)
    if mode == "impulse":
        x.zero_()
        x[0, 0, 0, 0] = 1
        x[-1, -1, -1, -1] = -1
    return x, weight


def validate(device):
    cases = [("paper_seed_" + str(seed), seed, "random")
             for seed in (0, 17, 2026)]
    cases += [("paper_zero", 5, "zero"),
              ("paper_ones", 11, "ones"),
              ("paper_impulses", 12, "impulse"),
              ("paper_offset", 13, "offset")]
    reports = []
    for name, seed, mode in cases:
        x_cpu, w_cpu = inputs(seed, mode)
        oracle = cpu_reference(x_cpu, w_cpu)
        x, weight = x_cpu.to(device), w_cpu.to(device)
        if mode == "offset":
            x = torch.empty(x_cpu.numel() + 1, device=device, dtype=x_cpu.dtype)[1:].view(PAPER_SHAPE)
            weight = torch.empty(w_cpu.numel() + 1, device=device, dtype=w_cpu.dtype)[1:].view(w_cpu.shape)
            x.copy_(x_cpu)
            weight.copy_(w_cpu)
        ref = baseline.run(x, weight)
        out = conv2d.run(x, weight)
        assert out.shape == conv2d.OUTPUT_SHAPE
        assert out.dtype == torch.float16 and out.is_contiguous()
        assert out.device == x.device
        report = {"case": name, "shape": PAPER_SHAPE, "filters": PAPER_FILTERS,
                  "baseline_vs_cpu_fp32": compare(ref, oracle),
                  "candidate_vs_cpu_fp32": compare(out, oracle),
                  "candidate_vs_baseline": compare(out, ref)}
        for _ in range(5):
            repeated = conv2d.run(x, weight)
            assert torch.equal(repeated, out), "Repeated launch is not bitwise stable"
        reports.append(report)
        print(f"PASS {name}", flush=True)
    return reports


def validate_streams(device):
    pending = []
    for seed in (333, 334):
        xc, wc = inputs(seed)
        oracle = cpu_reference(xc, wc)
        x, weight = xc.to(device), wc.to(device)
        stream = torch.npu.Stream(device=device)
        stream.wait_stream(torch.npu.current_stream(device))
        with torch.npu.stream(stream):
            # A dependent clone must see the custom kernel's output on this stream.
            copied = conv2d.run(x, weight).clone()
        pending.append((seed, stream, copied, oracle, x, weight))
    reports = []
    for seed, stream, copied, oracle, x, weight in pending:
        stream.synchronize()
        reports.append({"seed": seed, "candidate_vs_cpu_fp32": compare(copied, oracle)})
    print("PASS non_default_streams", flush=True)
    return reports


def capture(fn, calls):
    graph = torch.npu.NPUGraph()
    outputs = []
    with torch.npu.graph(graph):
        for _ in range(calls):
            outputs.append(fn())
    torch.npu.synchronize()
    return graph, outputs


def summarize(values):
    return {"median_us": statistics.median(values), "min_us": min(values),
            "max_us": max(values), "samples_us": values}


def time_graph(graph, replays, calls):
    torch.npu.synchronize()
    start = time.perf_counter_ns()
    for _ in range(replays):
        graph.replay()
        torch.npu.synchronize()
    return (time.perf_counter_ns() - start) / (replays * calls * 1000)


def time_eager(fn, iterations):
    torch.npu.synchronize()
    start = time.perf_counter_ns()
    for _ in range(iterations):
        out = fn()
        torch.npu.synchronize()
    elapsed = (time.perf_counter_ns() - start) / (iterations * 1000)
    assert out is not None
    return elapsed


def measure(args, device):
    x_cpu, w_cpu = inputs(1234)
    x, weight = x_cpu.to(device), w_cpu.to(device)
    funcs = {"torch_npu_nhwc": lambda: baseline.run(x, weight),
             "tle_nhwc": lambda: conv2d.run(x, weight)}
    for fn in funcs.values():
        for _ in range(args.warmup):
            fn()
    torch.npu.synchronize()
    graphs = {name: capture(fn, args.graph_calls) for name, fn in funcs.items()}

    # Replay must consume changed tensor values, including changed weights.
    new_x, new_w = inputs(4321)
    x.copy_(new_x)
    weight.copy_(new_w)
    oracle = cpu_reference(new_x, new_w)
    graph_checks = {}
    for name, (graph, outputs) in graphs.items():
        graph.replay()
        torch.npu.synchronize()
        for out in outputs:
            graph_checks[name] = compare(out, oracle)
        for _ in range(args.warmup):
            graph.replay()
            torch.npu.synchronize()
    torch.npu.synchronize()

    samples = {name: [] for name in funcs}
    eager = {name: [] for name in funcs}
    inventories = []
    for repeat in range(args.repeats):
        snapshot = inventory()
        inventories.append(snapshot)
        check_inventory(snapshot, args.physical_device, [os.getpid()])
        order = list(funcs) if repeat % 2 == 0 else list(reversed(funcs))
        for name in order:
            samples[name].append(time_graph(graphs[name][0], args.iterations, args.graph_calls))
        for name in order:
            eager[name].append(time_eager(funcs[name], args.iterations))
        snapshot = inventory()
        inventories.append(snapshot)
        check_inventory(snapshot, args.physical_device, [os.getpid()])
        print(f"Timing round {repeat + 1}/{args.repeats}: "
              + ", ".join(f"{name}={samples[name][-1]:.3f} us" for name in funcs), flush=True)
    graph_results = {name: summarize(values) for name, values in samples.items()}
    eager_results = {name: summarize(values) for name, values in eager.items()}
    return {"graph_wall": graph_results, "eager_wall": eager_results,
            "graph_changed_inputs": graph_checks, "timing_inventories": inventories,
            "speedup_graph": graph_results["torch_npu_nhwc"]["median_us"] / graph_results["tle_nhwc"]["median_us"],
            "speedup_eager": eager_results["torch_npu_nhwc"]["median_us"] / eager_results["tle_nhwc"]["median_us"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--physical-device", type=int)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--graph-calls", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "latest.json")
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.repeats, args.graph_calls) <= 0:
        parser.error("Timing counts must be positive")
    if args.physical_device is None:
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
        args.physical_device = int(visible.split(",")[args.device]) if visible else args.device
    torch.set_num_threads(4)
    before = inventory()
    check_inventory(before, args.physical_device, [])
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    files = [ROOT / "benchmark.py", ROOT / "run_benchmark.sh", *sorted((ROOT / "src").glob("*.py"))]
    report = {"status": "running", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "torch_npu": torch_npu.__version__, "triton": triton.__version__,
                              "triton_file": triton.__file__, "device": torch.npu.get_device_name(device),
                              "flagtree": metadata.version("flagtree"), "pid": os.getpid(),
                              "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                              "all_blocks_parallel": os.environ.get("TRITON_ALL_BLOCKS_PARALLEL"),
                              "allow_internal_format": getattr(torch.npu.config, "allow_internal_format", "unavailable"),
                              "allow_hf32_conv": getattr(torch.npu.conv, "allow_hf32", "unavailable"),
                              "allow_hf32_matmul": getattr(torch.npu.matmul, "allow_hf32", "unavailable")},
              "config": {**vars(args), "output": str(args.output), "atol": ATOL, "rtol": RTOL,
                         "cache": "warm; no explicit cache flush", "shape": PAPER_SHAPE,
                         "timing": "host perf_counter_ns; synchronize after each eager call or graph replay",
                         "filters": PAPER_FILTERS, "dtype": "float16"},
              "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in files},
              "inventory_before": before}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with torch.inference_mode():
            report["correctness"] = validate(device)
            report["streams"] = validate_streams(device)
            report["performance"] = measure(args, device)
        report["status"] = "pass"
    except Exception as exc:
        report["status"] = "fail"
        report["error"] = repr(exc)
        raise
    finally:
        report["inventory_after"] = inventory()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {args.output}", flush=True)


if __name__ == "__main__":
    main()

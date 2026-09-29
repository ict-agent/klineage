#!/usr/bin/env python3
"""GDN Stage 6/7 benchmark on Ascend: vLLM-Ascend chain vs our fused kernel.

Shape: the paper workload in problems/workloads/gdn.jsonl
       (B1 T4096 HQ16 HV48 D128, chunk 64).
Baseline: every stage is a vLLM-Ascend Triton kernel (baseline/fla).
Ours:     the same first five stages with Stage 6/7 replaced by ours/fused.py.

Both outputs are checked against the FLA chunk oracle embedded in
problems/definitions/gdn.json with atol = rtol = 1e-2, the gate the problem
harness uses. Latency is the sum of device kernel durations per iteration
(do_bench_npu profile), median of the active samples after trimming one
minimum and one maximum.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from baseline.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from baseline.fla.chunk_o import chunk_fwd_o
from baseline.fla.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from baseline.fla.cumsum import chunk_local_cumsum
from baseline.fla.solve_tril import solve_tril
from baseline.fla.utils import prepare_chunk_indices, prepare_chunk_offsets
from baseline.fla.wy_fast import recompute_w_u_fwd
from ours import chunk_h_o_fused

SEED = 0
WARMUP_ITERATIONS = 3
ACTIVE_ITERATIONS = 10
TOLERANCE = 1e-2
CHUNK_SIZE = 64
LARGE_BLOCK_T = 608 * 2  # solve_tril block, from baseline/fla/solve_tril.py
REFERENCE_NAME = "gdn_reference"
CHAIN_LABELS = ("vllm", "ours")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--problem-root", type=Path, default=ROOT.parents[2])
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERATIONS)
    parser.add_argument("--active", type=int, default=ACTIVE_ITERATIONS)
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=Path(os.environ.get("GDN_PROFILE_ROOT", "/tmp/gdn-optimization")),
    )
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    return parser.parse_args()


def load_problem(problem_root: Path) -> tuple[dict, dict, Path]:
    problems = problem_root / "problems"
    definition = json.loads((problems / "definitions" / "gdn.json").read_text())
    records = (problems / "workloads" / "gdn.jsonl").read_text().splitlines()
    workload = json.loads(records[0])["workload"]
    return definition, workload, problems


def axis_shape(spec: dict, definition: dict, axes: dict) -> tuple[int, ...]:
    shape = []
    for axis in spec["shape"]:
        if axis in axes:
            shape.append(axes[axis])
            continue
        shape.append(definition["axes"][axis]["value"])
    return tuple(shape)


def stage_inputs(definition, workload, problems, device):
    """Reproduce klineage.artifact.problem.trace_inputs: seed 0, CPU generator."""
    from safetensors import safe_open

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    inputs = {}
    for name, spec in definition["inputs"].items():
        shape = axis_shape(spec, definition, workload["axes"])
        dtype = getattr(torch, spec["dtype"])
        source = workload["inputs"][name]
        if source["type"] == "safetensors":
            with safe_open(
                problems / source["path"], framework="pt", device="cpu"
            ) as tensors:
                value = tensors.get_tensor(source["tensor_key"])
        else:
            value = torch.randn(shape, dtype=dtype, generator=generator)
        inputs[name] = value.to(device)
    return inputs


def build_metadata(cu_seqlens, heads, chunk_size):
    """Prebuild the chunk metadata the vLLM chain would get from its runner.

    Both chains share it, so no chain pays for index construction inside the
    measured region.
    """
    cumsum_block = 1 << (((2**18) // (heads * chunk_size)) - 1).bit_length()
    return {
        "cumsum": prepare_chunk_indices(cu_seqlens, cumsum_block),
        "chunk": prepare_chunk_indices(cu_seqlens, chunk_size),
        "large": prepare_chunk_indices(cu_seqlens, LARGE_BLOCK_T),
        "offsets": prepare_chunk_offsets(cu_seqlens, chunk_size),
    }


def build_chains(inputs, repeats, scale, cu_seqlens, meta):
    """Return the baseline and our pipeline, both over the same inputs."""
    q = inputs["q"].repeat_interleave(repeats, dim=2)
    k = inputs["k"].repeat_interleave(repeats, dim=2)
    v, g, beta = inputs["v"], inputs["g"], inputs["beta"]
    initial_state = inputs["initial_state"]

    # Stages 1-5: cumsum, causal KKT, triangular solve, WY recompute.
    def first_five():
        g_cumsum = chunk_local_cumsum(
            g,
            chunk_size=CHUNK_SIZE,
            cu_seqlens=cu_seqlens,
            block_indices=meta["cumsum"],
            output_dtype=torch.float32,
        )
        a = chunk_scaled_dot_kkt_fwd(
            k=k,
            beta=beta,
            g_cumsum=g_cumsum,
            cu_seqlens=cu_seqlens,
            chunk_indices=meta["chunk"],
            chunk_size=CHUNK_SIZE,
            output_dtype=torch.float32,
        )
        a_inv = solve_tril(
            A=a,
            cu_seqlens=cu_seqlens,
            chunk_indices_large_block=meta["large"],
            chunk_indices_bt=meta["chunk"],
            output_dtype=k.dtype,
        )
        w, u = recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            g_cumsum=g_cumsum,
            A=a_inv,
            cu_seqlens=cu_seqlens,
            chunk_indices=meta["chunk"],
        )
        return g_cumsum, w, u

    def vllm_chain():
        g_cumsum, w, u = first_five()
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g_cumsum,
            initial_state=initial_state,
            output_final_state=True,
            chunk_size=CHUNK_SIZE,
            save_new_value=True,
            cu_seqlens=cu_seqlens,
            chunk_indices=meta["chunk"],
            chunk_offsets=meta["offsets"],
        )
        o = chunk_fwd_o(
            q=q,
            k=k,
            v=v_new,
            h=h,
            g=g_cumsum,
            scale=scale,
            chunk_size=CHUNK_SIZE,
            cu_seqlens=cu_seqlens,
            chunk_offsets=meta["offsets"],
        )
        return o.float(), final_state

    def ours_chain():
        g_cumsum, w, u = first_five()
        o, final_state = chunk_h_o_fused(
            q=q,
            k=k,
            w=w,
            u=u,
            g=g_cumsum,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
            chunk_size=CHUNK_SIZE,
            cu_seqlens=cu_seqlens,
        )
        return o.float(), final_state

    return {"vllm": vllm_chain, "ours": ours_chain}


def chain_outputs(fn):
    with torch.no_grad():
        outputs = fn()
    torch.npu.synchronize()
    return outputs


def time_chain(fn, label: str, args) -> dict:
    from triton.backends.ascend.testing import do_bench_npu

    profile_dir = args.profile_root / label
    profile_dir.mkdir(parents=True, exist_ok=True)
    do_bench_npu(
        fn,
        warmup=args.warmup,
        active=args.active,
        prof_dir=str(profile_dir),
        keep_res=True,
    )

    with next(profile_dir.rglob("kernel_details.csv")).open(newline="") as file:
        rows = [
            row
            for row in csv.DictReader(file)
            if row.get("Type", "").lower() != "reducesum"
        ]
    # do_bench_npu profiles every iteration of both phases with the same grid.
    per_iteration = len(rows) // (args.warmup + args.active)
    active_rows = rows[args.warmup * per_iteration :]

    samples_us = [
        sum(
            float(row["Duration(us)"])
            for row in active_rows[index * per_iteration : (index + 1) * per_iteration]
        )
        for index in range(args.active)
    ]
    by_name_us: dict[str, float] = {}
    for row in active_rows:
        name = row.get("Name") or "unknown"
        by_name_us[name] = by_name_us.get(name, 0.0) + float(row["Duration(us)"]) / args.active

    return {
        "label": label,
        "samples_us": samples_us,
        "median_us": statistics.median(sorted(samples_us)[1:-1]),
        "kernels_per_iteration": per_iteration,
        "by_name_us": by_name_us,
    }


def run_reference(definition: dict, inputs, device):
    module = types.ModuleType(REFERENCE_NAME)
    exec(compile(definition["reference"], f"{REFERENCE_NAME}.py", "exec"), module.__dict__)
    with torch.no_grad():
        outputs = module.run(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["beta"],
            inputs["initial_state"],
            output_final_state=True,
        )
    torch.npu.synchronize()
    return outputs


def compare(actual, expected, actual_label: str, expected_label: str) -> list[dict]:
    checks = []
    for name, got, want in zip(("output", "final_state"), actual, expected):
        got, want = got.float(), want.float()
        abs_err = (got - want).abs()
        rel_err = abs_err / want.abs().clamp_min(1e-6)
        # torch.allclose passes where |got - want| <= atol + rtol * |want|.
        excess = abs_err - (TOLERANCE + TOLERANCE * want.abs())
        checks.append(
            {
                "actual": actual_label,
                "expected": expected_label,
                "tensor": name,
                "max_abs_err": abs_err.max().item(),
                "max_rel_err": rel_err.max().item(),
                "mean_abs_expected": want.abs().mean().item(),
                "violations": int((excess > 0).sum().item()),
                "max_excess": excess.max().item(),
                "allclose": bool(
                    torch.allclose(got, want, rtol=TOLERANCE, atol=TOLERANCE)
                ),
            }
        )
    return checks


def print_report(checks, timings, axes, repeats) -> None:
    shape = " ".join(f"{axis}{value}" for axis, value in axes.items())
    print(f"\nshape: {shape}, q/k heads repeated x{repeats}, chunk {CHUNK_SIZE}")
    print("\ncorrectness (atol = rtol = 1e-2)")
    print(
        f"{'actual':<8}{'expected':<12}{'tensor':<13}{'max_abs':>11}"
        f"{'|want|':>9}{'viol':>9}{'max_excess':>12}  ok"
    )
    for check in checks:
        print(
            f"{check['actual']:<8}{check['expected']:<12}{check['tensor']:<13}"
            f"{check['max_abs_err']:>11.3e}{check['mean_abs_expected']:>9.3f}"
            f"{check['violations']:>9d}{check['max_excess']:>12.2e}  "
            f"{'yes' if check['allclose'] else 'NO'}"
        )

    print("\nlatency (device kernel durations, us)")
    for timing in timings.values():
        samples = " / ".join(f"{value:.1f}" for value in timing["samples_us"])
        print(
            f"{timing['label']:<6}{timing['median_us']:>9.2f}  "
            f"({timing['kernels_per_iteration']} kernels/iter)  {samples}"
        )
    baseline_us = timings[CHAIN_LABELS[0]]["median_us"]
    ours_us = timings[CHAIN_LABELS[1]]["median_us"]
    print(f"\nspeedup vs vLLM-Ascend chain: {baseline_us / ours_us:.3f}x")

    print("\nper-kernel means (us)")
    for timing in timings.values():
        print(f"  {timing['label']}")
        for name, value in sorted(
            timing["by_name_us"].items(), key=lambda item: -item[1]
        ):
            print(f"    {value:>8.2f}  {name}")


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    definition, workload, problems = load_problem(args.problem_root)
    axes = workload["axes"]

    inputs = stage_inputs(definition, workload, problems, device)
    repeats = inputs["v"].shape[2] // inputs["q"].shape[2]
    scale = inputs["q"].shape[-1] ** -0.5
    cu_seqlens = torch.tensor([0, axes["T"]], dtype=torch.int32, device=device)
    meta = build_metadata(cu_seqlens, inputs["v"].shape[2], CHUNK_SIZE)
    chains = build_chains(inputs, repeats, scale, cu_seqlens, meta)

    outputs = {label: chain_outputs(chains[label]) for label in CHAIN_LABELS}
    checks = compare(outputs["ours"], outputs["vllm"], "ours", "vllm")
    if not args.skip_reference:
        reference = run_reference(definition, inputs, device)
        checks += compare(outputs["vllm"], reference, "vllm", "fla")
        checks += compare(outputs["ours"], reference, "ours", "fla")

    timings = {
        label: time_chain(chains[label], label, args) for label in CHAIN_LABELS
    }
    print_report(checks, timings, axes, repeats)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "axes": axes,
                    "repeats": repeats,
                    "chunk_size": CHUNK_SIZE,
                    "tolerance": TOLERANCE,
                    "checks": checks,
                    "timings": timings,
                },
                indent=2,
            )
        )
    return 0 if all(check["allclose"] for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

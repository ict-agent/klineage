"""Compare the saved GEMM with direct and copied contiguous outputs."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from enum import StrEnum
import hashlib
from importlib import metadata
import json
from pathlib import Path
import statistics
import subprocess

import torch

import evaluate as evaluator
from _process import _gpu_slot
from klineage.harness.timing import FlashInferCuptiTimer, TimingPolicy

_ROOT = Path(__file__).resolve().parents[1]
_JOB = _ROOT / "baseline/KDA/astra/gemm"
_PROBLEM = "gemm"
_CORRECTNESS_SEEDS = (17, 43, 101)
_TIMING_SEED = 211
_MUTATION_SEED = 307
_GPU_FIELDS = "name,uuid,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw"


class _Case(StrEnum):
    REFERENCE = "reference"
    PADDED = "padded"
    DIRECT = "direct_contiguous"
    COPY = "padded_then_contiguous"


def _invoke(case, candidate, oracle, inputs):
    if case == _Case.REFERENCE:
        return oracle(**inputs)
    if case == _Case.PADDED:
        return candidate(**inputs)
    if case == _Case.COPY:
        return candidate(**inputs).contiguous()

    x, weight = inputs["x"], inputs["weight"]
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    return candidate(**inputs, out=out)


def _check(case, actual, expected):
    rtol, atol = evaluator._TOLERANCES[_PROBLEM]
    torch.testing.assert_close(
        actual, expected, rtol=rtol, atol=atol,
        check_stride=case != _Case.PADDED,
    )
    if case != _Case.PADDED and not actual.is_contiguous():
        raise AssertionError(f"{case} output is not contiguous")


def _verify(candidate, oracle, inputs):
    original = {name: value.clone() for name, value in inputs.items()}
    expected = oracle(**original)
    records = {}
    for case in _Case:
        actual = _invoke(case, candidate, oracle, inputs)
        torch.cuda.synchronize()
        _check(case, actual, expected)
        for name, value in inputs.items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        records[case] = {
            "status": "passed", "shape": list(actual.shape),
            "dtype": str(actual.dtype), "stride": list(actual.stride()),
            "contiguous": actual.is_contiguous(),
            "max_abs_error": float((actual.float() - expected.float()).abs().max()),
        }
    return records


def _orders():
    # Balance measurement position and reverse the order to expose drift.
    cases = list(_Case)
    for order in (cases, cases[::-1]):
        for offset in range(len(order)):
            yield order[offset:] + order[:offset]


def _gpu_state():
    return subprocess.check_output([
        "nvidia-smi", f"--query-gpu={_GPU_FIELDS}", "--format=csv,noheader",
    ], text=True).strip()


def _save(output, result):
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)


def _summarize(rounds):
    summary = {}
    for case in _Case:
        medians = [row["results"][case]["median_ms"] for row in rounds]
        ratios = [row["results"][_Case.REFERENCE]["median_ms"] / value
                  for row, value in zip(rounds, medians, strict=True)]
        summary[case] = {
            "median_ms": statistics.median(medians),
            "round_medians_ms": medians,
            "round_range_ms": [min(medians), max(medians)],
            "median_paired_speedup": statistics.median(ratios),
        }
    reference_ms = summary[_Case.REFERENCE]["median_ms"]
    for row in summary.values():
        row["speedup"] = reference_ms / row["median_ms"]
        row["latency_reduction_pct"] = 100 * (1 - row["median_ms"] / reference_ms)
    return summary


def _run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = evaluator._snapshot(args.solution, output / "solution")
    reference_path = output / "reference.py"
    reference_path.write_bytes(args.reference.read_bytes())
    reference = evaluator._load(reference_path, "layout_reference")
    candidate = evaluator._load(output / "solution/kernel.py", "layout_candidate").kernel
    policy = TimingPolicy()
    timer = FlashInferCuptiTimer(policy)
    result = {
        "status": "running", "started_at": datetime.now(UTC).isoformat(),
        "sources": sources,
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "gpu_before": _gpu_state(), "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "flashinfer": metadata.version("flashinfer-python"),
        },
        "policy": policy.to_dict(),
        "precision": {"tf32": False, "bf16_reduced_precision_reduction": False},
        "timing_scope": "CUPTI first GPU activity start to last end per call; "
                        "COPY includes GEMM and contiguous copy; no CUDA graph; "
                        "host setup and allocation excluded",
        "correctness": {}, "rounds": [],
    }
    result_path = output / "results.json"
    _save(result_path, result)
    try:
        for seed in _CORRECTNESS_SEEDS:
            result["correctness"][str(seed)] = _verify(
                candidate, reference.torch_ref, reference.make_inputs(seed=seed),
            )
        inputs = reference.make_inputs(seed=_TIMING_SEED)
        result["correctness"][str(_TIMING_SEED)] = _verify(
            candidate, reference.torch_ref, inputs,
        )
        _save(result_path, result)
        for order in _orders():
            measured = {}
            for case in order:
                measured[case] = timer.measure(
                    lambda: _invoke(case, candidate, reference.torch_ref, inputs),
                ).to_dict()
                print(f"Round {len(result['rounds']) + 1} {case}: "
                      f"{measured[case]['median_ms'] * 1000:.3f} us", flush=True)
            result["rounds"].append({"order": order, "results": measured})
            _save(result_path, result)

        # Reuse the timed addresses with new values to reject cached outputs.
        pointers = {name: value.data_ptr() for name, value in inputs.items()}
        fresh = reference.make_inputs(seed=_MUTATION_SEED)
        for name, value in inputs.items():
            value.copy_(fresh[name])
        result["correctness"]["reused_addresses"] = _verify(
            candidate, reference.torch_ref, inputs,
        )
        assert pointers == {name: value.data_ptr() for name, value in inputs.items()}
        result["summary"] = _summarize(result["rounds"])
        result["environment"]["gpu_after"] = _gpu_state()
        result["status"] = "passed"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["finished_at"] = datetime.now(UTC).isoformat()
        _save(result_path, result)


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution", type=Path, default=_JOB / "kernel")
    parser.add_argument("--reference", type=Path, default=_ROOT / "problems/gemm/reference.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    with _gpu_slot(), torch.inference_mode():
        _run(args)


if __name__ == "__main__":
    _main()

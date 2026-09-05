"""Evaluate baseline kernels against the unchanged paper workloads."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import traceback
from enum import StrEnum
from pathlib import Path

import torch

from klineage.harness.timing import FlashInferCuptiTimer, TimingPolicy

_SEEDS = (17, 43, 101)
_TIMING_SEED = 211
_MUTATION_SEED = 307
_TOLERANCES = {
    "gemm": (0.02, 0.02),
    "conv2d": (0.01, 0.01),
    "fmha": (0.01, 0.005),
    "gdn": (0.01, 0.001),
    "topk": (0.0, 0.0),
}
_SOURCE_SUFFIXES = {".py", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc", ".c", ".json"}


class _Mode(StrEnum):
    CANDIDATE = "candidate"
    FULL = "full"
    REFERENCE = "reference"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _snapshot(source: Path, destination: Path) -> dict[str, str]:
    # Evaluate a frozen copy so each measurement identifies its exact sources.
    destination.mkdir(parents=True)
    hashes = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"source symlink: {path}")
        if not path.is_file() or path.suffix not in _SOURCE_SUFFIXES:
            continue
        relative = path.relative_to(source)
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    return hashes


def _check(problem, actual, expected, inputs) -> None:
    if problem == "topk":
        _check_topk(actual, expected, inputs["values"])
        return

    actuals = actual if isinstance(actual, tuple) else (actual,)
    expecteds = expected if isinstance(expected, tuple) else (expected,)
    if len(actuals) != len(expecteds):
        raise AssertionError("output count differs")

    rtol, atol = _TOLERANCES[problem]
    for value, golden in zip(actuals, expecteds, strict=True):
        torch.testing.assert_close(value, golden, rtol=rtol, atol=atol)


def _check_topk(indices, expected, values) -> None:
    if not isinstance(indices, torch.Tensor) or indices.dtype != torch.int32:
        raise AssertionError("Top-K must return int32 indices")
    if indices.shape != expected.indices.shape or indices.device != values.device:
        raise AssertionError("Top-K output shape or device differs")

    indices = indices.long()
    if bool(((indices < 0) | (indices >= values.shape[-1])).any()):
        raise AssertionError("Top-K index outside input")
    ordered = indices.sort(dim=-1).values
    if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
        raise AssertionError("Top-K repeats an index")

    selected = values.gather(-1, indices).sort(dim=-1).values
    golden = expected.values.sort(dim=-1).values
    torch.testing.assert_close(selected, golden, rtol=0, atol=0)


def _verify(problem, candidate, reference, inputs) -> None:
    # Separate inputs detect mutation and avoid contaminating the oracle.
    original = {name: value.clone() for name, value in inputs.items()}
    expected = reference(**original)
    actual = candidate(**inputs)
    torch.cuda.synchronize()
    for name, value in inputs.items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    _check(problem, actual, expected, original)


def _measure(candidate, inputs):
    timer = FlashInferCuptiTimer(TimingPolicy())
    return timer.measure(lambda: candidate(**inputs)).to_dict()


def _evaluate(args) -> dict:
    reference = _load(args.reference, "reference")
    oracle = reference.torch_ref
    result = {
        "problem": args.problem,
        "mode": args.mode,
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "seeds": list(_SEEDS),
        "tolerances": _TOLERANCES[args.problem],
    }
    if args.mode == _Mode.REFERENCE:
        result["reference"] = _measure(oracle, reference.make_inputs(seed=_TIMING_SEED))
        result["status"] = "reference_only"
        return result

    frozen = args.output.with_suffix("") / "solution"
    result["sources"] = _snapshot(args.solution, frozen)
    sys.path.insert(0, str(frozen))
    candidate = _load(frozen / "kernel.py", "kernel").kernel
    for seed in _SEEDS:
        _verify(args.problem, candidate, oracle, reference.make_inputs(seed=seed))

    inputs = reference.make_inputs(seed=_TIMING_SEED)
    _verify(args.problem, candidate, oracle, inputs)
    result["candidate"] = _measure(candidate, inputs)

    # Reuse addresses with new values to reject identity-based output caches.
    fresh = reference.make_inputs(seed=_MUTATION_SEED)
    for name, value in inputs.items():
        value.copy_(fresh[name])
    _verify(args.problem, candidate, oracle, inputs)
    if args.mode == _Mode.FULL:
        result["reference"] = _measure(oracle, fresh)
        result["speedup"] = (
            result["reference"]["median_ms"] / result["candidate"]["median_ms"]
        )
    result["status"] = "passed"
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=_TOLERANCES, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--solution", type=Path, default=Path("solution"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", type=_Mode, choices=list(_Mode), default=_Mode.CANDIDATE)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() or args.output.with_suffix("").exists():
        parser.error("use a new output path for each evaluation")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    try:
        with torch.inference_mode():
            result = _evaluate(args)
    except Exception as error:
        result = {"status": "failed", "error": str(error), "traceback": traceback.format_exc()}

    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    print(f"CORRECT={result['status'] == 'passed'}")
    if "candidate" in result:
        print(f"RUNTIME={result['candidate']['median_ms']}")
    if "speedup" in result:
        print(f"SPEEDUP={result['speedup']}")
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    _main()

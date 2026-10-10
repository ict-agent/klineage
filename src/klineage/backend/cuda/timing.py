"""CUDA latency measurement through FlashInfer and CUPTI."""

from __future__ import annotations

import importlib
import re
import statistics
from dataclasses import asdict
from importlib import metadata

from klineage.backend.timing import TimingPolicy, TimingResult, validated_samples

NAME = "cupti"
DEFAULT_POLICY = TimingPolicy()
MINIMUM_CUPTI_MAJOR = 13


def bench(fn, *, args=(), policy=None) -> TimingResult:
    """Measure CUDA kernel activity through FlashInfer's CUPTI timer."""
    if not callable(fn):
        raise TypeError("fn must be callable")
    policy = policy or DEFAULT_POLICY
    version = require_cupti()
    benchmark = load_flashinfer_benchmark()
    trials = []
    for _ in range(policy.trials):
        measured = benchmark(
            fn=fn,
            dry_run_iters=policy.warmup,
            repeat_iters=policy.repeat,
            cold_l2_cache=policy.cold_l2,
            use_cuda_graph=False,
            input_args=tuple(args),
            input_kwargs={},
        )
        samples = validated_samples(measured)
        if len(samples) != policy.repeat:
            raise ValueError("CUPTI sample count differs from the trial policy")
        trials.append(samples)
    return TimingResult(
        samples_ms=tuple(trials),
        median_ms=float(statistics.median(map(statistics.median, trials))),
        backend_used=NAME,
        details={"policy": asdict(policy), "cupti_version": version},
    )


def load_flashinfer_benchmark():
    # Keep FlashInfer optional for users who only manipulate kernel metadata.
    from flashinfer.testing.utils import bench_gpu_time_with_cupti

    return bench_gpu_time_with_cupti


def probe_cupti() -> str:
    # Mirror the imports used by FlashInfer itself so a successful check means
    # its implementation will take the CUPTI branch rather than silently fall
    # back to CUDA events.
    module = importlib.import_module("cupti")
    if not hasattr(module, "cupti"):
        importlib.import_module("cupti.cupti")
    return metadata.version("cupti-python")


def require_cupti() -> str:
    try:
        return validate_cupti(probe_cupti())
    except Exception as exc:
        raise RuntimeError(
            "strict CUPTI timing requested, but the backend probe failed: "
            f"{exception_text(exc)}"
        ) from exc


def validate_cupti(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("CUPTI version must be a non-empty string")
    version = value.strip()
    if version_major(version) < MINIMUM_CUPTI_MAJOR:
        raise ValueError(
            f"CUPTI {version} is older than required version {MINIMUM_CUPTI_MAJOR}"
        )
    return version


def version_major(version: str) -> int:
    match = re.match(r"\s*(\d+)", version)
    if match is None:
        raise ValueError(f"cannot determine CUPTI major version from {version!r}")
    return int(match.group(1))


def exception_text(exc: Exception) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def check_timing(details, policy: TimingPolicy) -> None:
    validate_cupti(details.get("cupti_version"))


__all__ = ["NAME", "DEFAULT_POLICY", "bench", "check_timing"]

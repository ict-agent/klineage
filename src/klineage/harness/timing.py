"""GPU timing contracts and a strict FlashInfer CUPTI implementation."""

from __future__ import annotations

import importlib
import math
import re
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Protocol

_BenchmarkCallable = Callable[..., Sequence[float]]
_BackendProbe = Callable[[], str]


@dataclass(frozen=True, slots=True)
class TimingPolicy:
    """Configuration shared by every kernel in a performance comparison."""

    warmup: int = 10
    repeat: int = 50
    cold_l2: bool = True
    trials: int = 3

    def __post_init__(self) -> None:
        for name in ("warmup", "repeat", "trials"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if not isinstance(self.cold_l2, bool):
            raise TypeError("cold_l2 must be a bool")

    def to_dict(self) -> dict[str, Any]:
        return {
            "warmup": self.warmup,
            "repeat": self.repeat,
            "cold_l2": self.cold_l2,
            "trials": self.trials,
        }


@dataclass(frozen=True, slots=True)
class TimingResult:
    """Raw and reduced latency measurements, in milliseconds."""

    samples_ms: tuple[tuple[float, ...], ...]
    median_ms: float
    backend_used: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        samples = tuple(
            tuple(float(item) for item in trial) for trial in self.samples_ms
        )
        median = float(self.median_ms)
        if not samples or any(not trial for trial in samples):
            raise ValueError("samples_ms must contain at least one non-empty trial")
        for value in (median, *(item for trial in samples for item in trial)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("timing values must be finite and greater than zero")
        if not isinstance(self.backend_used, str) or not self.backend_used:
            raise ValueError("backend_used must be a non-empty string")
        object.__setattr__(self, "samples_ms", samples)
        object.__setattr__(self, "median_ms", median)
        object.__setattr__(self, "details", dict(self.details))

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples_ms": [list(trial) for trial in self.samples_ms],
            "median_ms": self.median_ms,
            "backend_used": self.backend_used,
            "details": dict(self.details),
        }


class KernelTimer(Protocol):
    """Measure one callable without including input construction in the timing."""

    @property
    def policy(self) -> TimingPolicy: ...

    def measure(
        self,
        fn: Callable[..., Any],
        *,
        args: tuple[Any, ...] = (),
    ) -> TimingResult: ...


class FlashInferCuptiTimer:
    """Measure kernel latency with FlashInfer's CUPTI activity timer.

    ``benchmark`` and ``backend_probe`` are injection points for deterministic
    tests. A probe returns a CUPTI version string such as ``"13.0.0"``.
    """

    def __init__(
        self,
        policy: TimingPolicy | None = None,
        *,
        benchmark: _BenchmarkCallable | None = None,
        backend_probe: _BackendProbe | None = None,
    ) -> None:
        self._policy = policy or TimingPolicy()
        self._benchmark = benchmark
        self._backend_probe = backend_probe or _probe_cupti

    @property
    def policy(self) -> TimingPolicy:
        return self._policy

    def measure(
        self,
        fn: Callable[..., Any],
        *,
        args: tuple[Any, ...] = (),
    ) -> TimingResult:
        if not callable(fn):
            raise TypeError("fn must be callable")
        call_args = tuple(args)

        version = _require_cupti(self._backend_probe)

        benchmark = self._benchmark or _load_flashinfer_benchmark()
        trial_samples: list[tuple[float, ...]] = []
        trial_medians: list[float] = []
        for _ in range(self.policy.trials):
            measured = benchmark(
                fn=fn,
                dry_run_iters=self.policy.warmup,
                repeat_iters=self.policy.repeat,
                cold_l2_cache=self.policy.cold_l2,
                use_cuda_graph=False,
                input_args=call_args,
                input_kwargs={},
            )
            samples = _validated_samples(measured)
            trial_samples.append(samples)
            trial_medians.append(float(statistics.median(samples)))

        details: dict[str, Any] = {
            "policy": self.policy.to_dict(),
            "cupti_version": version,
        }
        return TimingResult(
            samples_ms=tuple(trial_samples),
            median_ms=float(statistics.median(trial_medians)),
            backend_used="cupti",
            details=details,
        )


def _load_flashinfer_benchmark() -> _BenchmarkCallable:
    # Keep FlashInfer optional for users who only manipulate lineage metadata.
    from flashinfer.testing import bench_gpu_time_with_cupti

    return bench_gpu_time_with_cupti


def _probe_cupti() -> str:
    # Mirror the imports used by FlashInfer itself so a successful check means
    # its implementation will take the CUPTI branch rather than silently fall
    # back to CUDA events.
    module = importlib.import_module("cupti")
    if not hasattr(module, "cupti"):
        importlib.import_module("cupti.cupti")
    return metadata.version("cupti-python")


def _require_cupti(probe: _BackendProbe) -> str:
    try:
        value = probe()
        if not isinstance(value, str) or not value.strip():
            raise TypeError("CUPTI probe must return a version string")
        version = value.strip()
        if _version_major(version) < 13:
            raise RuntimeError(f"CUPTI {version} is older than required version 13")
        return version
    except Exception as exc:
        raise RuntimeError(
            "strict CUPTI timing requested, but the backend probe failed: "
            f"{_exception_text(exc)}"
        ) from exc


def _version_major(version: str) -> int:
    match = re.match(r"\s*(\d+)", version)
    if match is None:
        raise ValueError(f"cannot determine CUPTI major version from {version!r}")
    return int(match.group(1))


def _validated_samples(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("benchmark must return a sequence of millisecond samples")
    samples = tuple(float(value) for value in values)
    if not samples:
        raise ValueError("benchmark returned no timing samples")
    for value in samples:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("benchmark samples must be finite and greater than zero")
    return samples


def _exception_text(exc: Exception) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


__all__ = [
    "FlashInferCuptiTimer",
    "KernelTimer",
    "TimingPolicy",
    "TimingResult",
]

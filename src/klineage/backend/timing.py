"""Timing data and automatic backend dispatch."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from klineage.artifact.kernel import Kernel

EVENT_SCOPE = "current_stream_interval"


@dataclass(frozen=True, slots=True)
class TimingPolicy:
    """Measurement settings shared by kernels in a performance comparison.

    Counts must be positive integers. Backends may reject unsupported settings:
    CUDA defaults to cold L2 with CUPTI, while HIP and Ascend event timing use
    ``cold_l2=False`` and reject cold-cache requests.

    Attributes:
        warmup: Untimed calls before sampling each trial.
        repeat: Measured samples per trial.
        cold_l2: Request the timing provider's cold-L2 measurement mode.
        trials: Independent sample groups used for the latency reduction.
    """

    warmup: int = 10
    repeat: int = 50
    cold_l2: bool = True
    trials: int = 3

    def __post_init__(self):
        for name in ("warmup", "repeat", "trials"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if not isinstance(self.cold_l2, bool):
            raise TypeError("cold_l2 must be a bool")


@dataclass(frozen=True, slots=True)
class TimingResult:
    """Raw and reduced latency measurements, in milliseconds.

    Construction requires nonempty sample groups and finite positive timings.
    It normalizes values but does not recompute or validate the supplied median
    against the samples; the evaluation harness checks measurement evidence.

    Attributes:
        samples_ms: Trials containing individual millisecond samples.
        median_ms: Reduced latency; built-in timers use the median of trial
            medians rather than the median of all pooled samples.
        backend_used: Measurement provider identifier, such as ``cupti``.
        details: Provider evidence, including policy and version or stream scope.
    """

    samples_ms: tuple[tuple[float, ...], ...]
    median_ms: float
    backend_used: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
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


def bench(
    fn: Kernel, *, args: tuple[Any, ...] = (), policy: TimingPolicy | None = None
) -> TimingResult:
    """Detect a backend, check the kernel target, and measure its latency.

    This performs timing only; use the evaluation service for correctness and
    acceptance. CUDA uses CUPTI kernel-activity samples. HIP and Ascend measure
    current-stream event intervals without a cold-L2 guarantee.

    Args:
        fn: Kernel already built into a callable, with a resolvable problem.
        args: Positional inputs already allocated on the target device.
        policy: Measurement settings; None uses the detected backend's default.

    Returns:
        Backend samples and timing evidence, with latency in milliseconds.

    Raises:
        ValueError: The detected backend differs from the problem's target,
            or the timing provider rejects the policy or samples.
        RuntimeError: No supported backend is available or timing cannot run.
    """
    from klineage.artifact.problem import load_problem

    from .registry import detect_backend, get_backend

    backend = detect_backend()
    problem = load_problem(fn)
    target = get_backend(problem.language, problem.platform)
    if backend.kind != target.kind:
        raise ValueError("detected backend differs from the kernel target")
    return backend.bench(fn, args=args, policy=policy)


def validated_samples(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("benchmark must return a sequence of millisecond samples")
    samples = tuple(float(value) for value in values)
    if not samples:
        raise ValueError("benchmark returned no timing samples")
    for value in samples:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("benchmark samples must be finite and greater than zero")
    return samples


def check_event_timing(details, policy):
    if details.get("scope") != EVENT_SCOPE:
        raise ValueError("device-event timing requires current-stream evidence")
    if policy.cold_l2:
        raise ValueError("device-event timing cannot claim cold_l2 evidence")

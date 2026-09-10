"""Device timing, performance comparisons, and CUDA counter profiling."""

from __future__ import annotations

import csv
import importlib
import io
import json
import math
import re
import statistics
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from klineage.backend import Backend
    from klineage.harness.profiling import ProfileOptions


_BenchmarkCallable = Callable[..., Sequence[float]]
_BackendProbe = Callable[[], str]
_MINIMUM_RATIO = 0.99
_CUPTI_BACKEND = "cupti"
_MINIMUM_CUPTI_MAJOR = 13
_EVENT_BACKENDS = frozenset({"hip-events", "npu-events"})
_EVENT_SCOPE = "current_stream_interval"
_ROUNDING_TOLERANCE = 1e-9
GATE_LOG_PREFIX = "Performance gate: "
NCU = "/usr/local/cuda/bin/ncu"
PROFILE_RANGE = "klineage_profile"
_REGEX = "regex:"
_ID = "ID"
_KERNEL = "Kernel Name"
_METRIC = "Metric Name"
_FIELDS = {
    "kernel": _KERNEL,
    "launch_id": _ID,
    "section": "Section Name",
    "metric": _METRIC,
    "unit": "Metric Unit",
    "value": "Metric Value",
}
_RAW_META = frozenset(
    (
        _ID,
        "Process ID",
        "Process Name",
        "Host Name",
        _KERNEL,
        "Context",
        "Stream",
        "Block Size",
        "Grid Size",
        "Device",
        "CC",
    )
)
_ERRORS = {
    "ERR_NVGPUCTRPERM": "NCU ERR_NVGPUCTRPERM: GPU counter permission denied",
    "No kernels were profiled": "NCU: No kernels were profiled",
}
_LOG_PREFIXES = ("==PROF==", "==WARNING==", "==ERROR==")
_TOOL_TIMEOUT_SECONDS = 10


@dataclass(frozen=True, slots=True)
class TimingPolicy:
    """Configuration shared by every kernel in a performance comparison."""

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
    ):
        self.policy = policy or TimingPolicy()
        self.benchmark = benchmark
        self.backend_probe = backend_probe or probe_cupti

    def measure(
        self,
        fn: Callable[..., Any],
        *,
        args: tuple[Any, ...] = (),
    ) -> TimingResult:
        if not callable(fn):
            raise TypeError("fn must be callable")
        call_args = tuple(args)

        version = require_cupti(self.backend_probe)

        benchmark = self.benchmark or load_flashinfer_benchmark()
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
            samples = validated_samples(measured)
            if len(samples) != self.policy.repeat:
                raise ValueError("CUPTI sample count differs from the trial policy")
            trial_samples.append(samples)
            trial_medians.append(float(statistics.median(samples)))

        details: dict[str, Any] = {
            "policy": self.policy.to_dict(),
            "cupti_version": version,
        }
        return TimingResult(
            samples_ms=tuple(trial_samples),
            median_ms=float(statistics.median(trial_medians)),
            backend_used=_CUPTI_BACKEND,
            details=details,
        )


class DeviceEventTimer:
    """Measure a HIP/NPU stream interval, including gaps between device work."""

    def __init__(self, backend: Backend, policy: TimingPolicy | None = None):
        if backend.timing_backend not in _EVENT_BACKENDS:
            raise ValueError("CUDA requires strict CUPTI timing")
        self.backend = backend
        self.policy = policy or TimingPolicy(cold_l2=False)
        if self.policy.cold_l2:
            raise ValueError("cold_l2 is unsupported by HIP/NPU event timing")

    def measure(self, fn: Callable[..., Any], *, args: tuple[Any, ...] = ()):
        if not callable(fn):
            raise TypeError("fn must be callable")
        runtime = self.backend.runtime()
        torch = self.backend.torch()
        device = next(
            (value.device for value in args if hasattr(value, "device")), None
        )
        if device is None:
            device = self.backend.device()
        trials = []
        with runtime.device(device), torch.inference_mode():
            stream = runtime.current_stream(device)
            for _ in range(self.policy.trials):
                for _ in range(self.policy.warmup):
                    fn(*args)
                runtime.synchronize(device)

                # Allocate events before recording; time only the current stream.
                events = [
                    (
                        runtime.Event(enable_timing=True),
                        runtime.Event(enable_timing=True),
                    )
                    for _ in range(self.policy.repeat)
                ]
                samples = []
                for start, end in events:
                    start.record(stream)
                    output = fn(*args)
                    end.record(stream)
                    end.synchronize()
                    samples.append(start.elapsed_time(end))
                    del output
                trials.append(validated_samples(samples))

        return TimingResult(
            samples_ms=tuple(trials),
            median_ms=float(statistics.median(map(statistics.median, trials))),
            backend_used=self.backend.timing_backend,
            details={
                "policy": self.policy.to_dict(),
                "scope": _EVENT_SCOPE,
                "device_type": self.backend.device_type,
            },
        )


def timing_policy(backend: Backend) -> TimingPolicy:
    return TimingPolicy(cold_l2=backend.timing_backend == _CUPTI_BACKEND)


def make_timer(backend: Backend, policy: TimingPolicy) -> KernelTimer:
    if backend.timing_backend == _CUPTI_BACKEND:
        return FlashInferCuptiTimer(policy)
    return DeviceEventTimer(backend, policy)


def load_flashinfer_benchmark() -> _BenchmarkCallable:
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


def require_cupti(probe: _BackendProbe) -> str:
    try:
        return validate_cupti(probe())
    except Exception as exc:
        raise RuntimeError(
            "strict CUPTI timing requested, but the backend probe failed: "
            f"{exception_text(exc)}"
        ) from exc


def validate_cupti(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError("CUPTI version must be a non-empty string")
    version = value.strip()
    if version_major(version) < _MINIMUM_CUPTI_MAJOR:
        raise ValueError(
            f"CUPTI {version} is older than required version {_MINIMUM_CUPTI_MAJOR}"
        )
    return version


def version_major(version: str) -> int:
    match = re.match(r"\s*(\d+)", version)
    if match is None:
        raise ValueError(f"cannot determine CUPTI major version from {version!r}")
    return int(match.group(1))


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


def exception_text(exc: Exception) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def verify_performance(
    candidate: TimingResult,
    reference: TimingResult,
    minimum_ratio: float = _MINIMUM_RATIO,
) -> bool:
    """Compare paired trials from the same timing backend and policy."""

    if not math.isfinite(minimum_ratio) or minimum_ratio <= 0:
        raise ValueError("minimum_ratio must be finite and positive")
    record: dict[str, Any] = {"minimum_ratio": minimum_ratio}
    passed = False
    try:
        record.update(candidate=candidate.to_dict(), reference=reference.to_dict())
        current, policy = timing_evidence(record["candidate"], candidate.median_ms)
        baseline, baseline_policy = timing_evidence(
            record["reference"], reference.median_ms
        )
        if current.backend_used != baseline.backend_used:
            raise ValueError("candidate and reference timing backends differ")
        if policy != baseline_policy:
            raise ValueError("candidate and reference timing policies differ")
        ratios = [
            statistics.median(old) / statistics.median(new)
            for new, old in zip(current.samples_ms, baseline.samples_ms, strict=True)
        ]
        overall = reference.median_ms / candidate.median_ms
        record.update(overall_ratio=overall, trial_ratios=ratios)
        failures = [
            f"trial {index}"
            for index, ratio in enumerate(ratios, start=1)
            if ratio < minimum_ratio
        ]
        if overall < minimum_ratio:
            failures.insert(0, "overall")
        passed = not failures
        record["reason"] = (
            "passed"
            if passed
            else f"ratio below {minimum_ratio}: {', '.join(failures)}"
        )
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        record["reason"] = f"{type(error).__name__}: {error}"

    # The worker retains stderr; validation keeps its existing schema.
    record["passed"] = passed
    print(GATE_LOG_PREFIX + json.dumps(record, default=str), file=sys.stderr)
    return passed


def timing_evidence(
    value: Mapping[str, Any],
    latency: float | None,
) -> tuple[TimingResult, TimingPolicy]:
    details = value["details"]
    backend = value["backend_used"]
    if backend == _CUPTI_BACKEND:
        validate_cupti(details.get("cupti_version"))
    elif backend in _EVENT_BACKENDS:
        if details.get("scope") != _EVENT_SCOPE:
            raise ValueError("device-event timing requires current-stream evidence")
    else:
        raise ValueError("unsupported timing backend")
    policy = TimingPolicy(**details["policy"])
    if backend in _EVENT_BACKENDS and policy.cold_l2:
        raise ValueError("device-event timing cannot claim cold_l2 evidence")
    timing = TimingResult(
        samples_ms=value["samples_ms"],
        median_ms=value["median_ms"],
        backend_used=value["backend_used"],
        details=details,
    )
    if len(timing.samples_ms) != policy.trials or any(
        len(trial) != policy.repeat for trial in timing.samples_ms
    ):
        raise ValueError("timing sample count differs from the trial policy")
    median = statistics.median(statistics.median(trial) for trial in timing.samples_ms)
    if latency is None or any(
        not math.isclose(median, recorded, rel_tol=_ROUNDING_TOLERANCE)
        for recorded in (latency, timing.median_ms)
    ):
        raise ValueError("recorded latency differs from timing samples")
    return timing, policy


def ncu_command(
    options: ProfileOptions,
    raw_path: Path,
    report_path: Path,
) -> tuple[str, ...]:
    # The worker brackets only target launches; nested NVTX ranges remain included.
    command = [
        NCU,
        "--target-processes",
        "application-only",
        "--replay-mode",
        "kernel",
        "--nvtx",
        "--nvtx-include",
        PROFILE_RANGE + "/",
        "--set",
        options.set,
        "--page",
        "raw",
        "--csv",
        "--print-units",
        "base",
        "--log-file",
        str(raw_path),
        "--export",
        str(report_path),
        "--force-overwrite",
        "--kernel-name-base",
        "demangled",
    ]
    for section in options.sections:
        command.extend(("--section", section))

    if options.kernel_filter:
        pattern = options.kernel_filter
        command.extend(
            (
                "--kernel-name",
                pattern if pattern.startswith(_REGEX) else _REGEX + pattern,
            )
        )

    return tuple(command)


def parse_metrics(text: str) -> list[dict[str, str]]:
    for marker, message in _ERRORS.items():
        if marker in text:
            raise ValueError(message)

    metrics = []
    header = []
    units = {}

    # Parse complete CSV records so quoted newlines survive intervening profiler logs.
    for row in csv.reader(io.StringIO(text, newline="")):
        if not row or row[0].startswith(_LOG_PREFIXES):
            continue
        if _ID in row and _KERNEL in row:
            header, units = row, {}
            continue
        if not header or len(row) != len(header):
            continue

        values = dict(zip(header, row))
        if _METRIC in values:
            if not all(name in values for name in _FIELDS.values()):
                continue
            metric = {name: values[column] for name, column in _FIELDS.items()}
            if all(
                metric[name].strip()
                for name in ("kernel", "launch_id", "metric", "value")
            ):
                metrics.append(metric)
            continue

        # Raw CSV stores units above launch rows and provides no section mapping.
        if not values[_ID]:
            units = values
            continue
        if not values[_KERNEL].strip():
            continue
        for name, value in values.items():
            if name in _RAW_META or not name.strip() or not value.strip():
                continue
            metrics.append(
                {
                    "kernel": values[_KERNEL],
                    "launch_id": values[_ID],
                    "section": "",
                    "metric": name,
                    "unit": units.get(name, ""),
                    "value": value,
                }
            )

    if not metrics:
        raise ValueError("NCU report contains no metrics")
    return metrics


def profile_launch(fn: Callable[..., Any], *, args: tuple[Any, ...], device):
    """Collect one target invocation; leave compilation and warmup outside NVTX."""
    import torch

    version = subprocess.check_output(
        (NCU, "--version"),
        text=True,
        timeout=_TOOL_TIMEOUT_SECONDS,
    ).strip()
    with torch.cuda.device(device), torch.inference_mode():
        properties = torch.cuda.get_device_properties(device)
        fn(*args)
        torch.cuda.synchronize(device)
        with torch.cuda.nvtx.range(PROFILE_RANGE):
            output = fn(*args)
            torch.cuda.synchronize(device)
        del output
    return {
        "tool_version": version,
        "device": {
            "name": properties.name,
            "uuid": str(properties.uuid),
            "capability": f"sm{properties.major}{properties.minor}",
        },
    }


__all__ = [
    "DeviceEventTimer",
    "FlashInferCuptiTimer",
    "KernelTimer",
    "TimingPolicy",
    "TimingResult",
    "make_timer",
    "timing_policy",
    "verify_performance",
]

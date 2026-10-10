"""Backend-independent timing evidence and performance comparisons."""

from __future__ import annotations

import math
import statistics
import sys
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from klineage.backend import get_timing_backend
from klineage.backend.timing import TimingPolicy, TimingResult
from klineage.serialization import to_json

_MINIMUM_RATIO = 0.99


_ROUNDING_TOLERANCE = 1e-9


GATE_LOG_PREFIX = "Performance gate: "


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
        record.update(candidate=asdict(candidate), reference=asdict(reference))
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
    print(GATE_LOG_PREFIX + to_json(record, indent=None), file=sys.stderr)
    return passed


def timing_evidence(
    value: Mapping[str, Any],
    latency: float | None,
) -> tuple[TimingResult, TimingPolicy]:
    details = value["details"]
    backend = value["backend_used"]
    policy = TimingPolicy(**details["policy"])
    get_timing_backend(backend).timing.check_timing(details, policy)
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




__all__ = ["TimingPolicy", "TimingResult", "timing_evidence", "verify_performance"]

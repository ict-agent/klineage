"""Source-backed semantic audit and conservative CUPTI performance gates."""

from __future__ import annotations

import hashlib
import math
import re
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from klineage.errors import StructuredOutputError
from klineage.harness.eval import ValidationResult
from klineage.harness.timing import TimingPolicy, TimingResult
from klineage.kernel import Kernel
from klineage.prompts import render_prompt

_MINIMUM_RATIO = 0.99
_CUPTI_BACKEND = "cupti"
_ROUNDING_TOLERANCE = 1e-9
_MECHANISMS = frozenset(("compute", "tiling", "pipeline", "layout", "scheduling"))
_PRESERVED = frozenset(("preserved", "not_applicable"))
_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)


def verify_performance(
    validation: ValidationResult,
    minimum_ratio: float = _MINIMUM_RATIO,
) -> ValidationResult:
    """Require matching CUPTI policies and the threshold in every trial."""

    if not math.isfinite(minimum_ratio) or minimum_ratio <= 0:
        raise ValueError("minimum_ratio must be finite and positive")
    evidence: dict[str, Any] = {"minimum_ratio": minimum_ratio, "passed": False}
    try:
        timing = validation.details["timing"]
        candidate, policy = _timing(timing["candidate"], validation.latency_ms)
        reference, reference_policy = _timing(
            timing["reference"], validation.reference_latency_ms,
        )
        if policy != reference_policy:
            raise ValueError("candidate and reference timing policies differ")
        ratios = [
            statistics.median(baseline) / statistics.median(generated)
            for generated, baseline in zip(
                candidate.samples_ms, reference.samples_ms, strict=True,
            )
        ]
        ratio = reference.median_ms / candidate.median_ms
        evidence.update(
            backend=_CUPTI_BACKEND, relative_performance=ratio, trial_ratios=ratios,
            passed=validation.accepted and min(ratio, *ratios) >= minimum_ratio,
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        evidence["error"] = str(exc)

    return replace(
        validation,
        profile_passed=validation.profile_passed and evidence["passed"],
        details={**validation.details, "performance_verifier": evidence},
    )


def check_evidence(
    evidence: Sequence[Mapping[str, Any]],
    sources: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Validate exact line quotations and bind them to source hashes."""

    if (
        not isinstance(evidence, Sequence)
        or isinstance(evidence, (str, bytes))
        or not evidence
    ):
        raise ValueError("evidence must contain source citations")
    checked = []
    for citation in evidence:
        if not isinstance(citation, Mapping):
            raise ValueError("each citation must be an object")
        path, start, end = (citation.get(key) for key in ("path", "start", "end"))
        if not isinstance(path, str) or path not in sources:
            raise ValueError("citation path is not an allowed source")
        if type(start) is not int or type(end) is not int:
            raise ValueError("citation lines must be integers")
        source = sources[path]
        lines = source.splitlines()
        if not 1 <= start <= end <= len(lines):
            raise ValueError("citation lines are outside the source")
        quote = "\n".join(lines[start - 1:end])
        if not quote.strip() or citation.get("quote") != quote:
            raise ValueError("citation quote does not match the source lines")
        checked.append({
            "path": path, "start": start, "end": end, "quote": quote,
            "sha256": hashlib.sha256(source.encode()).hexdigest(),
        })
    return checked


def verify_mechanism(
    ask: Callable[..., Mapping[str, Any]],
    expert: Kernel,
    candidate: Kernel,
    repository: Path,
) -> dict[str, Any]:
    """Audit mechanisms independently; unknown or ungrounded findings fail."""

    result = {
        "method": "independent_source_semantic_audit",
        "formal_proof": False,
        "expert_fingerprint": expert.fingerprint,
        "candidate_fingerprint": candidate.fingerprint,
        "passed": False,
        "checks": None,
    }
    try:
        report = ask("verify-mechanism", render_prompt("verify_mechanism"), {
            "repository": str(repository),
            "expert_path": str(expert.artifact_path) if expert.artifact_path else None,
            "expert": expert.source,
            "candidate": candidate.source,
            "problem": expert.problem.to_dict() if expert.problem else None,
            "kernel_abi": expert.abi.to_dict() if expert.abi else None,
            "target": expert.context.to_dict(),
        })
        if not isinstance(report, Mapping):
            raise ValueError("mechanism audit must be an object")
        standalone = report.get("standalone")
        result["standalone"] = standalone
        if not isinstance(standalone, Mapping) or standalone.get("status") != "standalone":
            raise ValueError("candidate execution is delegated or unknown")
        reason = standalone.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("standalone execution requires an explanation")
        result["standalone"] = {
            **standalone,
            "candidate": check_evidence(
                standalone.get("candidate"), {"candidate": candidate.source},
            ),
        }
        checks = report.get("checks")
        result["checks"] = checks
        if not isinstance(checks, list) or len(checks) != len(_MECHANISMS):
            raise ValueError("audit must cover all five mechanisms")
        if any(not isinstance(check, Mapping) for check in checks):
            raise ValueError("mechanism checks must be objects")
        if {check.get("mechanism") for check in checks} != _MECHANISMS:
            raise ValueError("audit must cover each mechanism exactly once")
        sources = _expert_sources(checks, repository, expert.source)
        checked = []
        expert_paths = set()
        for check in checks:
            if check.get("status") not in _PRESERVED:
                raise ValueError(f"mechanism {check['mechanism']} is changed or unknown")
            if check["mechanism"] == "compute" and check["status"] != "preserved":
                raise ValueError("the compute mechanism must be preserved")
            if not isinstance(check.get("reason"), str) or not check["reason"].strip():
                raise ValueError("mechanism checks require a source-based explanation")
            baseline = check_evidence(check.get("expert"), sources)
            generated = check_evidence(
                check.get("candidate"), {"candidate": candidate.source},
            )
            expert_paths.update(item["path"] for item in baseline)
            checked.append({**check, "expert": baseline, "candidate": generated})
        if "expert" not in expert_paths:
            raise ValueError("audit must cite the specified expert instance")
        if _uses_library(expert, repository) and expert_paths == {"expert"}:
            raise ValueError("library expert requires implementation source evidence")
        result.update(passed=True, checks=checked)
    except (KeyError, TypeError, ValueError, OSError, StructuredOutputError) as exc:
        result["error"] = str(exc)
    return result


def _timing(
    value: Mapping[str, Any],
    latency: float | None,
) -> tuple[TimingResult, TimingPolicy]:
    details = value["details"]
    if value["backend_used"] != _CUPTI_BACKEND or not details.get("cupti_version"):
        raise ValueError("performance fidelity requires CUPTI timing evidence")
    policy = TimingPolicy(**details["policy"])
    timing = TimingResult(
        samples_ms=value["samples_ms"], median_ms=value["median_ms"],
        backend_used=value["backend_used"], details=details,
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
        raise ValueError("recorded latency differs from CUPTI samples")
    return timing, policy


def _expert_sources(
    checks: Sequence[Mapping[str, Any]],
    repository: Path,
    source: str,
) -> dict[str, str]:
    sources = {"expert": source}
    root = repository.resolve(strict=True)
    for check in checks:
        citations = check.get("expert")
        if not isinstance(citations, list):
            raise ValueError("expert evidence must be a citation array")
        for citation in citations:
            if not isinstance(citation, Mapping):
                raise ValueError("expert citation must be an object")
            name = citation.get("path")
            if name in sources:
                continue
            if not isinstance(name, str):
                raise ValueError("expert citation requires a repository path")
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("expert citation must remain inside the repository")
            path = (root / path).resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("expert citation must name a repository source file")
            sources[name] = path.read_text(encoding="utf-8")
    return sources


def _uses_library(expert: Kernel, repository: Path) -> bool:
    roots = (repository, repository / "include", repository / "tools/util/include")
    if expert.artifact_path:
        roots += (expert.artifact_path.parent,)
    for name in _INCLUDE_RE.findall(expert.source):
        if any((root / name).is_file() for root in roots):
            return True
    return False


__all__ = ["check_evidence", "verify_mechanism", "verify_performance"]

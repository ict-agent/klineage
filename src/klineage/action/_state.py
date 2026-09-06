"""Observe code mechanisms before advancing a skill plan."""

from collections.abc import Sequence
from dataclasses import replace

from klineage.action._sandbox import _Sandbox
from klineage.errors import StructuredOutputError
from klineage.harness.fidelity import EvidenceMode, check_evidence
from klineage.kernel import Feature, Kernel
from klineage.prompts import render_prompt


def _observe(
    sandbox: _Sandbox, kernel: Kernel, features: Sequence[Feature],
) -> Kernel:
    wanted = tuple(dict.fromkeys(features))
    if not wanted:
        return replace(kernel, features=())

    sources = kernel.source_files or {"kernel.cu": kernel.source}
    citation_limit, line_limit = EvidenceMode.RANGES.value
    prompt = render_prompt(
        "observe_features", citation_limit=citation_limit, line_limit=line_limit,
    )
    response = sandbox._ask("observe-features", prompt, {
        "source_files": sources,
        "source_dir": sandbox._snapshot(sources),
        "target": kernel.context.to_dict(),
        "features": [item.to_dict() for item in wanted],
    })
    observed, evidence = [], []
    try:
        checks = response["checks"]
        if not isinstance(checks, list) or len(checks) != len(wanted):
            raise ValueError("one check per requested feature is required")
        seen = set()
        for check in checks:
            feature = Feature.from_dict(check["feature"])
            if feature not in wanted or feature in seen:
                raise ValueError("unexpected or repeated feature")
            seen.add(feature)
            status = check["status"]
            if status not in ("present", "absent", "unknown"):
                raise ValueError("invalid feature status")
            if status == "unknown":
                raise ValueError(f"unresolved feature: {feature.locus}/{feature.name}")
            reason = check.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("feature checks require a source-based explanation")
            citations = check_evidence(check["evidence"], sources, EvidenceMode.RANGES)
            if status == "present":
                observed.append(feature)
            evidence.append({**check, "evidence": citations})
    except (KeyError, TypeError, ValueError) as exc:
        raise StructuredOutputError(f"invalid feature audit: {exc}") from exc

    validation = kernel.validation
    if validation is not None:
        validation = replace(validation, details={
            **validation.details,
            "feature_verifier": {"method": "source_semantic_audit", "checks": evidence},
        })
    return replace(kernel, features=tuple(observed), validation=validation)

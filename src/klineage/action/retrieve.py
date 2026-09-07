"""Rank eligible lineage paths using the current kernel's measured profile."""

from collections.abc import Mapping, Sequence
from typing import Any

from klineage.action._sandbox import _Kind, _next, _Sandbox
from klineage.action.profile import _profile
from klineage.errors import ActionError
from klineage.kernel import Kernel
from klineage.memory.lineage import Lineage
from klineage.memory.paths import RetrievalPlan, retrieve_paths
from klineage.memory.skillcard import SkillAdmission

_INSTRUCTIONS = """Rank the supplied executable paths for the current kernel.
Use current.profile, its available measurements, and the workload contract to
match bottlenecks to SkillCard intent, carrier, requirements, effects, and risks.
Distinguish measured signals from hypotheses. Source-case latency and fidelity
are evidence; do not promise transferred gains or multiply per-card speedups.
The path lists identify eligible edges; skills preserve the full source memory.
Rank whole paths only: do not change, combine, invent, or omit paths or reorder
their skills. Return exactly {"plan_ids": ["plan-00", ...], "reasons":
{"plan-00": "profile evidence and relevant skill mechanism", ...}}.
Include every supplied plan_id exactly once and a nonempty reason for each.
"""


def retrieve(
    current_kernel: Kernel, lineages: Sequence[Lineage], *,
    skill_admission: SkillAdmission = SkillAdmission.OFF,
) -> tuple[RetrievalPlan, ...]:
    """Profile a kernel and rank its compatible, unchanged memory paths."""

    if not isinstance(current_kernel, Kernel):
        raise TypeError("retrieve requires a Kernel")
    with _next(_Kind.RETRIEVE, current_kernel) as sandbox:
        return _retrieve(sandbox, current_kernel, lineages, skill_admission)


def _retrieve(
    sandbox: _Sandbox, current: Kernel, lineages: Sequence[Lineage],
    admission: SkillAdmission,
) -> tuple[RetrievalPlan, ...]:
    plans = retrieve_paths(lineages, current, skill_admission=admission)
    if not plans:
        return ()

    current = _profile(sandbox, current)
    candidates = {f"plan-{index:02d}": plan for index, plan in enumerate(plans)}
    response = sandbox._ask("retrieve", _INSTRUCTIONS, {
        "current": current._prompt_input(),
        "plans": [_candidate(key, plan) for key, plan in candidates.items()],
    })
    return _ordered(response, candidates)


def _candidate(key: str, plan: RetrievalPlan) -> dict[str, Any]:
    lineage = plan.lineage
    terminal = lineage.states[-1]
    end = lineage.transitions.index(plan.skills[-1].evidence[0], plan.entry_index) + 1
    return {
        "plan_id": key, "entry_index": plan.entry_index,
        "context": terminal.context.to_dict(),
        "problem": terminal.problem.to_dict() if terminal.problem else None,
        "abi": terminal.abi.to_dict() if terminal.abi else None,
        "entry": _state(lineage.states[plan.entry_index]),
        "terminal": _state(terminal),
        "path_end": _state(lineage.states[end]),
        "skills": [card.to_dict() for card in lineage.skills],
        "path": [{"skill_id": card.skill_id,
                  "before": card.evidence[0].before_fingerprint,
                  "after": card.evidence[0].after_fingerprint} for card in plan.skills],
        "termination": lineage.termination.value, "reason": lineage.reason,
    }


def _state(kernel: Kernel) -> dict[str, Any]:
    validation = kernel.validation
    return {
        "fingerprint": kernel.fingerprint,
        "features": [feature.to_dict() for feature in kernel.features],
        "accepted": validation.accepted if validation else False,
        "latency_ms": validation.latency_ms if validation else None,
        "reference_latency_ms": validation.reference_latency_ms if validation else None,
        "fidelity": validation.details.get("performance_verifier") if validation else None,
    }


def _ordered(
    response: Mapping[str, Any], candidates: Mapping[str, RetrievalPlan],
) -> tuple[RetrievalPlan, ...]:
    message = "retrieval must return every plan once with a nonempty reason"
    if not isinstance(response, Mapping) or set(response) != {"plan_ids", "reasons"}:
        raise ActionError(message)
    order, reasons = response["plan_ids"], response["reasons"]
    if (not isinstance(order, list) or not all(isinstance(key, str) for key in order)
            or len(order) != len(candidates) or set(order) != set(candidates)):
        raise ActionError(message)
    if (not isinstance(reasons, Mapping) or set(reasons) != set(candidates)
            or any(not isinstance(reason, str) or not reason.strip() for reason in reasons.values())):
        raise ActionError(message)
    return tuple(candidates[key] for key in order)


__all__ = ["retrieve"]

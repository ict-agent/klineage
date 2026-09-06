"""Retrieve, materialize, and validate optimization paths."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import replace

from klineage.action._contract import _contract
from klineage.action._sandbox import _Kind, _new, _next, _Sandbox
from klineage.action._state import _observe
from klineage.action.code_gen import _materialize, _seed, _skill_sequence
from klineage.errors import ActionError, ValidationGateError
from klineage.harness.fidelity import verify_performance
from klineage.kernel import Feature, Kernel
from klineage.memory.lineage import Lineage
from klineage.memory.paths import retrieve_paths
from klineage.memory.skillcard import SkillAdmission, SkillCard

_MAX_STEPS = 16
_NO_REGRESSION = 1.0
_HARDWARE_LOCUS = "hardware"


def apply(
    current_kernel: Kernel | str | os.PathLike[str],
    skills: Sequence[SkillCard] | Sequence[Lineage],
    *,
    skill_admission: SkillAdmission = SkillAdmission.OFF,
) -> Kernel:
    """Apply selected cards or search lineages for an existing/new case."""

    if not isinstance(skill_admission, SkillAdmission):
        raise TypeError("skill_admission must be a SkillAdmission")
    if isinstance(skills, (str, bytes)) or not isinstance(skills, Sequence) or not skills:
        raise ValueError("apply requires a nonempty card or lineage sequence")
    paths = all(isinstance(item, Lineage) for item in skills)
    if not paths and not all(isinstance(item, SkillCard) for item in skills):
        raise TypeError("provide only SkillCards or only Lineages")
    if isinstance(current_kernel, Kernel):
        with _next(_Kind.APPLY, current_kernel) as sandbox:
            run = _run_paths if paths else _apply
            return run(sandbox, current_kernel, skills, skill_admission)
    if not paths:
        raise TypeError("a new case requires lineages")

    with _new(_Kind.APPLY, current_kernel) as sandbox:
        problem, abi, context = _contract(sandbox._problem, sandbox._inspect())
        sandbox._save(problem, abi, context)
        plans = retrieve_paths(
            skills, context, problem=problem, abi=abi, skill_admission=skill_admission,
        )
        if not plans:
            raise ActionError("no compatible lineage for the new case")
        failure = None
        for index, plan in enumerate(plans):
            try:
                baseline = _seed(sandbox, problem, abi, context, plan,
                                 output=f"baseline-{index:02d}.cu")
                return _run_paths(
                    sandbox, baseline, skills, skill_admission,
                    required=plan.lineage.states[plan.entry_index].features,
                )
            except ActionError as exc:
                failure = exc
        raise ActionError("no retrieved lineage produced a valid baseline") from failure


def _apply(
    sandbox: _Sandbox, current_kernel: Kernel, skills: Sequence[SkillCard],
    skill_admission: SkillAdmission,
) -> Kernel:
    cards = _skill_sequence(skills)
    if not isinstance(skill_admission, SkillAdmission):
        raise TypeError("skill_admission must be a SkillAdmission")
    if skill_admission is SkillAdmission.ON and any(not card.admitted for card in cards):
        raise ActionError("cannot apply unadmitted SkillCards")
    if any(card.provides for card in cards):
        current_kernel = _observe(sandbox, current_kernel, (
            *current_kernel.features, *_features(cards),
        ))
    candidate = _step(sandbox, current_kernel, cards, "submission")
    validation = verify_performance(candidate.validation)
    candidate = candidate.with_validation(validation)
    if not validation.accepted:
        raise ValidationGateError("applied kernel failed performance fidelity", kernel=candidate)
    return candidate


def _step(
    sandbox: _Sandbox, current: Kernel, cards: Sequence[SkillCard], output: str,
    *, vocabulary: Sequence[Feature] = (),
) -> Kernel:
    available = set(current.features)
    available.update(Feature(_HARDWARE_LOCUS, name) for name in current.context.capabilities)
    for card in cards:
        if not set(card.requires).issubset(available) or set(card.conflicts) & available:
            raise ActionError("skill requirements or conflicts reject the current state")
        available.update(card.provides)

    candidate = _materialize(sandbox, current, cards, output=output)
    candidate = candidate.with_validation(sandbox._evaluate(candidate, reference=current))
    if not candidate.validation.accepted:
        raise ValidationGateError("generated kernel failed validation", kernel=candidate)

    # Requested effects become state only after a separate source audit.
    wanted = tuple(sorted(item for item in available if item.locus != _HARDWARE_LOCUS))
    candidate = _observe(sandbox, candidate, (*wanted, *vocabulary))
    if not set(wanted).issubset(candidate.features):
        candidate = candidate.with_validation(replace(candidate.validation, profile_passed=False))
        raise ValidationGateError("generated kernel did not preserve the planned mechanisms",
                                  kernel=candidate)
    return replace(candidate, context=current.context.with_actions(
        card.action_category for card in cards
    ))


def _run_paths(
    sandbox: _Sandbox, current: Kernel, lineages: Sequence[Lineage],
    skill_admission: SkillAdmission,
    *, required: Sequence[Feature] = (),
) -> Kernel:
    current = current.with_validation(sandbox._evaluate(current, reference=None))
    if not current.validation.accepted:
        raise ValidationGateError("baseline failed validation", kernel=current)
    current = _observe(sandbox, current, (
        *required, *current.features,
        *_features(card for lineage in lineages for card in lineage.skills),
    ))
    if not set(required).issubset(current.features):
        raise ValidationGateError("baseline is missing the retrieved entry mechanisms",
                                  kernel=current)
    plans = retrieve_paths(lineages, current, skill_admission=skill_admission)
    best, attempts = current, 0
    history = []
    for plan in plans:
        candidate = current
        while attempts < _MAX_STEPS:
            pending = retrieve_paths((plan.lineage,), candidate, skill_admission=skill_admission)
            if not pending:
                break
            attempts += 1
            card = pending[0].skills[0]
            try:
                candidate = _step(
                    sandbox, candidate, pending[0].skills[:1], f"step-{attempts:02d}",
                    vocabulary=_features(plan.lineage.skills),
                )
            except ActionError as error:
                history.append({"skill_id": card.skill_id, "error": str(error)})
                break
            history.append({"skill_id": card.skill_id, "kernel": candidate.fingerprint,
                            "latency_ms": candidate.validation.latency_ms})

            # Keep preparation steps; replace the best only after remeasurement.
            if candidate.validation.latency_ms > best.validation.latency_ms:
                continue
            checked = verify_performance(
                sandbox._evaluate(candidate, reference=best), _NO_REGRESSION,
            )
            if checked.accepted:
                best = candidate.with_validation(replace(
                    checked, details={**candidate.validation.details, **checked.details},
                ))
    return best.with_validation(replace(best.validation, details={
        **best.validation.details,
        "apply_search": {"attempts": history, "step_limit": _MAX_STEPS},
    }))


def _features(cards: Iterable[SkillCard]) -> tuple[Feature, ...]:
    return tuple(dict.fromkeys(
        feature for card in cards for name in ("requires", "provides", "conflicts")
        for feature in getattr(card, name) if feature.locus != _HARDWARE_LOCUS
    ))

"""Decompose an expert kernel into a verified lineage."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from klineage._utils import nonempty, signature, stable_id
from klineage.action._sandbox import _Kind, _next, _Sandbox
from klineage.action._state import _observe
from klineage.action.code_gen import _materialize
from klineage.errors import StructuredOutputError, ValidationGateError
from klineage.harness.artifacts import (
    is_cuda_language,
    require_pure_cuda,
)
from klineage.harness.eval import EffectVerifier, ValidationResult
from klineage.harness.fidelity import verify_performance
from klineage.harness.structured import response_strings
from klineage.kernel import Feature, Kernel
from klineage.memory.lineage import Lineage, Termination
from klineage.memory.skillcard import (
    Scope,
    SkillAdmission,
    SkillCard,
    TransitionEvidence,
    VerificationTrial,
)
from klineage.prompts import render_prompt


def decompose(
    expert_kernel: Kernel,
    *,
    max_steps: int = 16,
    max_rejections: int = 4,
    roundtrip_cases: Sequence[Kernel] = (),
    effect_verifier: EffectVerifier | None = None,
    skill_admission: SkillAdmission = SkillAdmission.OFF,
) -> Lineage:
    """Recover a validated lineage in one action sandbox."""

    with _next(_Kind.DECOMPOSE, expert_kernel) as sandbox:
        return _decompose(
            sandbox,
            expert_kernel,
            max_steps=max_steps,
            max_rejections=max_rejections,
            roundtrip_cases=roundtrip_cases,
            effect_verifier=effect_verifier,
            skill_admission=skill_admission,
        )


def _decompose(
    sandbox: _Sandbox,
    expert_kernel: Kernel,
    *,
    max_steps: int,
    max_rejections: int,
    roundtrip_cases: Sequence[Kernel],
    effect_verifier: EffectVerifier | None,
    skill_admission: SkillAdmission,
) -> Lineage:

    if max_steps <= 0 or max_rejections < 0:
        raise ValueError("max_steps must be positive and max_rejections non-negative")
    if not isinstance(skill_admission, SkillAdmission):
        raise TypeError("skill_admission must be a SkillAdmission")
    if skill_admission is SkillAdmission.ON and not roundtrip_cases:
        raise ValueError("skill admission requires at least one roundtrip case")
    if roundtrip_cases and skill_admission is SkillAdmission.OFF:
        raise ValueError("roundtrip_cases require skill admission")
    if skill_admission is SkillAdmission.ON and effect_verifier is None:
        raise ValueError(
            "skill admission requires an explicit effect_verifier"
        )
    expert_kernel = expert_kernel.with_validation(
        sandbox._evaluate(expert_kernel, reference=None)
    )
    if not expert_kernel.validation.accepted:
        raise ValidationGateError(
            "expert kernel does not pass the validation gate",
            kernel=expert_kernel,
        )

    suffix = ".cu" if is_cuda_language(expert_kernel.context.language) else ".txt"
    current = expert_kernel
    backward_transitions: list[TransitionEvidence] = []
    backward_states = [expert_kernel]
    rejected_by_category: dict[str, list[str]] = defaultdict(list)
    rejection_count = 0
    termination, reason = Termination.STEP_LIMIT, "decomposition step budget exhausted"

    for step in range(max_steps):
        predecessor_name = f"step-{step:02d}-predecessor{suffix}"
        payload = {
            "current": current._prompt_input(),
            "accepted_actions": [
                transition.action_category
                for transition in backward_transitions
            ],
            "rejected_attempts": dict(rejected_by_category),
            "output_path": sandbox._file(predecessor_name),
        }
        instructions = render_prompt("decompose", step_number=step + 1)
        proposal = sandbox._ask(
            "deoptimize",
            instructions,
            payload,
        )
        done = proposal.get("done")
        if not isinstance(done, bool):
            raise StructuredOutputError("deoptimization 'done' is required and boolean")
        if done:
            reason = nonempty(proposal.get("reason"), "termination reason")
            termination = Termination.COMPLETE
            if current.features:
                termination = Termination.UNKNOWN
                reason += "; remaining mechanisms: " + ", ".join(
                    f"{item.locus}/{item.name}" for item in current.features
                )
            break

        category = nonempty(proposal["action_category"], "action category")
        locus = nonempty(proposal["locus"], "locus")
        backward_edit = nonempty(proposal["backward_edit"], "backward edit")
        predecessor_source, predecessor_artifact = sandbox._take_file(
            predecessor_name,
            "deoptimized source",
        )
        if is_cuda_language(current.context.language):
            require_pure_cuda(predecessor_source, repository=sandbox._repo)
        predecessor = Kernel(
            name=f"{current.name}.deopt{len(backward_transitions) + 1}",
            source=predecessor_source,
            context=current.context,
            problem=current.problem,
            abi=current.abi,
            artifact_path=predecessor_artifact,
        )
        predecessor_validation = sandbox._evaluate(
            predecessor,
            reference=expert_kernel,
        )
        predecessor = predecessor.with_validation(predecessor_validation)
        if predecessor_validation.accepted:
            try:
                audited, predecessor = _audit_transition(
                    sandbox, current, predecessor, proposal, locus,
                )
                predecessor_validation = predecessor.validation
            except (StructuredOutputError, ValidationGateError) as error:
                predecessor_validation = _audit_failure(predecessor_validation, error)
        if not predecessor_validation.accepted:
            rejection_count += 1
            rejected_by_category[category].append(
                _validation_summary(predecessor_validation)
            )
            if rejection_count >= max_rejections:
                termination = Termination.REJECTION_LIMIT
                reason = "decomposition rejection budget exhausted"
                break
            continue

        roundtrip_name = f"step-{step:02d}-roundtrip{suffix}"
        forward_payload = {
            "simpler_kernel": predecessor._prompt_input(),
            "target_latency_ms": current.validation.latency_ms,
            "action_category": category,
            "locus": locus,
            "backward_edit": backward_edit,
            "before_features": [item.to_dict() for item in predecessor.features],
            "after_features": [item.to_dict() for item in audited.features],
            "output_path": sandbox._file(roundtrip_name),
        }
        forward_instructions = render_prompt("decompose_rederive")
        forward = sandbox._ask(
            "rederive-forward",
            forward_instructions,
            forward_payload,
        )
        roundtrip_source, roundtrip_artifact = sandbox._take_file(
            roundtrip_name,
            "forward source",
        )
        if is_cuda_language(current.context.language):
            require_pure_cuda(roundtrip_source, repository=sandbox._repo)
        roundtrip = Kernel(
            name=f"{predecessor.name}.roundtrip",
            source=roundtrip_source,
            context=current.context,
            problem=current.problem,
            abi=current.abi,
            artifact_path=roundtrip_artifact,
        )
        roundtrip_validation = verify_performance(
            sandbox._evaluate(roundtrip, reference=current)
        )
        if roundtrip_validation.accepted:
            try:
                roundtrip = _observe(
                    sandbox, roundtrip.with_validation(roundtrip_validation),
                    audited.features,
                )
                if set(roundtrip.features) != set(audited.features):
                    raise ValidationGateError("forward roundtrip did not restore mechanisms")
                roundtrip_validation = roundtrip.validation
            except (StructuredOutputError, ValidationGateError) as error:
                roundtrip_validation = _audit_failure(roundtrip_validation, error)
        if not roundtrip_validation.accepted:
            rejection_count += 1
            rejected_by_category[category].append(
                "forward roundtrip failed: " + _validation_summary(roundtrip_validation)
            )
            if rejection_count >= max_rejections:
                termination = Termination.REJECTION_LIMIT
                reason = "decomposition rejection budget exhausted"
                break
            continue

        forward_edit = nonempty(forward["forward_edit"], "forward edit")
        evidence = TransitionEvidence(
            action_category=category,
            locus=locus,
            before_fingerprint=predecessor.fingerprint,
            after_fingerprint=current.fingerprint,
            backward_edit=backward_edit,
            forward_edit=forward_edit,
            predecessor_validation=predecessor_validation,
            roundtrip_validation=roundtrip_validation,
            rejected_evidence=tuple(rejected_by_category.get(category, ())),
        )
        backward_transitions.append(evidence)
        backward_states[-1] = audited
        backward_states.append(predecessor)
        current = predecessor

    forward_transitions = tuple(reversed(backward_transitions))
    states = tuple(reversed(backward_states))
    skills = _lift_transitions(
        forward_transitions,
        states=states,
        sandbox=sandbox,
    )
    if skill_admission is SkillAdmission.ON:
        assert effect_verifier is not None
        skills = _admit_skills(
            skills,
            states=states,
            cases=roundtrip_cases,
            effect_verifier=effect_verifier,
        )

    return Lineage(
        states=states,
        transitions=forward_transitions,
        skills=skills,
        termination=termination,
        reason=reason,
    )


def _features(response: Mapping[str, Any], name: str) -> tuple[Feature, ...]:
    try:
        values = response[name]
        if not isinstance(values, list):
            raise TypeError("expected an array")
        if any(not isinstance(item, Mapping) for item in values):
            raise TypeError("features must be objects")
        return tuple(dict.fromkeys(Feature.from_dict(item) for item in values))
    except (KeyError, TypeError, ValueError) as error:
        raise StructuredOutputError(f"invalid {name}: {error}") from error


def _audit_transition(
    sandbox: _Sandbox,
    current: Kernel,
    predecessor: Kernel,
    proposal: Mapping[str, Any],
    locus: str,
) -> tuple[Kernel, Kernel]:
    before = set(_features(proposal, "before_features"))
    after = set(_features(proposal, "after_features"))
    removed = after - before
    if not before < after or len(removed) != 1 or next(iter(removed)).locus != locus:
        raise StructuredOutputError("decomposition must remove one mechanism at its locus")

    # Check retained features too; a claimed edit cannot erase unrelated state.
    wanted = tuple(dict.fromkeys((*current.features, *sorted(after))))
    current = _observe(sandbox, current, wanted)
    predecessor = _observe(sandbox, predecessor, wanted)
    if set(current.features) != set(wanted):
        raise ValidationGateError("current kernel lacks declared mechanisms")
    if set(predecessor.features) != set(wanted) - removed:
        raise ValidationGateError("predecessor did not remove exactly the declared mechanism")
    return current, predecessor


def _audit_failure(validation: ValidationResult, error: Exception) -> ValidationResult:
    return replace(validation, profile_passed=False, details={
        **validation.details, "feature_verifier": {"error": str(error)},
    })


def _lift_transitions(
    transitions: Sequence[TransitionEvidence],
    *,
    states: Sequence[Kernel],
    sandbox: _Sandbox,
) -> tuple[SkillCard, ...]:
    grouped = defaultdict(list)
    for index, evidence in enumerate(transitions):
        key = (
            evidence.action_category, signature(evidence.locus),
            tuple(sorted(states[index].features)),
            tuple(sorted(states[index + 1].features)),
        )
        grouped[key].append(index)

    cards: list[SkillCard] = []
    for (category, *_), group in grouped.items():
        evidence = tuple(transitions[index] for index in group)
        payload = {
            "action_category": category,
            # Every transition shares the lineage's operator contract.
            "context": states[0].context.to_dict(),
            "problem": states[0].problem.to_dict() if states[0].problem else None,
            "abi": states[0].abi.to_dict() if states[0].abi else None,
            "transitions": [
                {
                    "before": states[index].source_files or states[index].source,
                    "after": states[index + 1].source_files or states[index + 1].source,
                    "before_features": [feature.to_dict() for feature in states[index].features],
                    "after_features": [feature.to_dict() for feature in states[index + 1].features],
                    "locus": item.locus,
                    "backward_edit": item.backward_edit,
                    "forward_edit": item.forward_edit,
                    "before_latency_ms": item.predecessor_validation.latency_ms,
                    "target_latency_ms": item.roundtrip_validation.reference_latency_ms,
                    "roundtrip_latency_ms": item.roundtrip_validation.latency_ms,
                    "rejected_evidence": list(item.rejected_evidence),
                }
                for index, item in zip(group, evidence, strict=True)
            ],
        }
        instructions = render_prompt(
            "decompose_lift",
            transition_count=len(group),
        )
        response = sandbox._ask(
            "lift-skill",
            instructions,
            payload,
        )
        context = states[0].context
        scope = replace(
            _scope_from_response(response.get("scope")), cases=(context.case,),
            languages=(context.language,), platforms=(context.platform,),
        )
        features = {name: _features(response, name)
                    for name in ("requires", "provides", "conflicts")}
        for index in group:
            before = set(states[index].features)
            added = set(states[index + 1].features) - before
            if (not features["provides"] or set(features["provides"]) != added
                    or not set(features["requires"]).issubset(before)
                    or set(features["conflicts"]) & before):
                raise StructuredOutputError("skill feature contract lacks transition evidence")
        intent = nonempty(response["intent"], "intent")
        skill_id = stable_id(
            "skill",
            category,
            intent,
            *(f"{item.before_fingerprint}:{item.after_fingerprint}" for item in evidence),
        )
        cards.append(
            SkillCard(
                skill_id=skill_id,
                intent=intent,
                anchor=nonempty(response["anchor"], "anchor"),
                carrier=nonempty(response["carrier"], "carrier"),
                preconditions=response_strings(response, "preconditions"),
                effects=response_strings(response, "effects"),
                evidence=evidence,
                risks=response_strings(response, "risks"),
                scope=scope,
                **features,
            )
        )

    return tuple(cards)


def _admit_skills(
    skills: Sequence[SkillCard],
    *,
    states: Sequence[Kernel],
    cases: Sequence[Kernel],
    effect_verifier: EffectVerifier,
) -> tuple[SkillCard, ...]:
    lineage_contexts = {state.context for state in states}
    trials = [list(card.verification_log) for card in skills]
    for case in cases:
        # Held-out kernels retain their own trusted problem and evaluator.
        with _next(_Kind.APPLY, case) as sandbox:
            baseline = case.with_validation(
                sandbox._evaluate(case, reference=None)
            )
            if not baseline.validation.accepted:
                raise ValidationGateError(
                    "held-out baseline does not pass the validation gate",
                    kernel=baseline,
                )

            for index, card in enumerate(skills):
                present = set(baseline.features)
                if (not set(card.requires).issubset(present)
                        or set(card.conflicts) & present):
                    continue
                candidate = _materialize(
                    sandbox,
                    baseline,
                    (card,),
                    output=f"admission-{index:02d}",
                )
                validation = sandbox._evaluate(candidate, reference=baseline)
                candidate = candidate.with_validation(validation)
                if validation.accepted and card.provides:
                    try:
                        candidate = _observe(
                            sandbox, candidate, (*baseline.features, *card.provides),
                        )
                        if not (present | set(card.provides)).issubset(candidate.features):
                            raise ValidationGateError("skill effects absent")
                        validation = candidate.validation
                    except (StructuredOutputError, ValidationGateError) as error:
                        validation = _audit_failure(validation, error)
                    candidate = candidate.with_validation(validation)
                trial = VerificationTrial(
                    target=baseline.context,
                    validation=validation,
                    expected_effect_observed=(
                        validation.accepted and effect_verifier(card, baseline, candidate)
                    ),
                    held_out=baseline.context not in lineage_contexts,
                )
                trials[index].append(trial)
                if validation.accepted and trial.expected_effect_observed:
                    baseline = candidate

    return tuple(
        replace(card, verification_log=tuple(log))
        for card, log in zip(skills, trials, strict=True)
    )


def _scope_from_response(value: Any) -> Scope:
    if not isinstance(value, Mapping):
        raise StructuredOutputError("scope must be an object")
    try:
        return Scope(
            cases=response_strings(value, "cases"),
            languages=response_strings(value, "languages"),
            platforms=response_strings(value, "platforms"),
            prior_actions=response_strings(value, "prior_actions"),
        )
    except ValueError as error:
        raise StructuredOutputError(str(error)) from error


def _validation_summary(validation: ValidationResult) -> str:
    return json.dumps(validation.to_dict(), ensure_ascii=False, sort_keys=True)

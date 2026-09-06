"""Retrieve executable paths from concrete lineage edges."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
import re

from klineage.contract import ABIValue, KernelABI, ProblemSpec
from klineage.kernel import Feature, Kernel, TargetContext
from klineage.memory.lineage import Lineage
from klineage.memory.skillcard import SkillAdmission, SkillCard

_WILDCARD = "*"
_HARDWARE_LOCUS = "hardware"
_LAYOUT_FIELDS = ("layout", "contiguous")
_STRIDE_FIELD = "stride"


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    """One source path; each card carries only its concrete edge evidence."""

    lineage: Lineage
    entry_index: int
    skills: tuple[SkillCard, ...]


def retrieve_paths(
    lineages: Sequence[Lineage],
    target: Kernel | TargetContext,
    *,
    problem: ProblemSpec | None = None,
    abi: KernelABI | None = None,
    skill_admission: SkillAdmission = SkillAdmission.OFF,
) -> tuple[RetrievalPlan, ...]:
    """Rank compatible paths; a source-free target can retrieve a first plan.

    Kernel features are verified state. A source-free plan instead requests
    the source entry's features as a scaffold; generate and verify it first.
    Planned effects must be verified after every generated step.
    """

    if not isinstance(target, (Kernel, TargetContext)):
        raise TypeError("target must be a Kernel or TargetContext")
    if not isinstance(skill_admission, SkillAdmission):
        raise TypeError("skill_admission must be a SkillAdmission")
    context = target.context if isinstance(target, Kernel) else target
    features = set(target.features) if isinstance(target, Kernel) else set()
    if isinstance(target, Kernel):
        problem, abi = target.problem, target.abi
    features.update(Feature(_HARDWARE_LOCUS, name) for name in context.capabilities)

    ranked = []
    for lineage in lineages:
        if not isinstance(lineage, Lineage):
            raise TypeError("lineages must contain Lineage objects")
        rank = _contract_rank(lineage.states[-1], context, problem, abi)
        if rank is None:
            continue

        # Match evidence, never the order of abstract cards or action history.
        cards = tuple(next((
            replace(card, evidence=(edge,)) for card in lineage.skills
            if edge in card.evidence and card.provides
            and _scope_match(card, context)
            and (skill_admission is SkillAdmission.OFF or card.admitted)
        ), None) for edge in lineage.transitions)
        paths = []
        for start in range(len(cards)):
            initial = set(features)
            if isinstance(target, TargetContext):
                initial.update(feature for feature in lineage.states[start].features
                               if feature.locus != _HARDWARE_LOCUS)
            paths.append(_path(lineage, cards, initial, start))
        plan = max(
            (path for path in paths if path.skills),
            key=lambda path: (len(path.skills), path.entry_index),
            default=None,
        )
        if plan is not None:
            ranked.append(((*rank, _gain(plan)), plan))

    return tuple(plan for _, plan in sorted(ranked, key=lambda item: item[0], reverse=True))


def _path(
    lineage: Lineage,
    cards: tuple[SkillCard | None, ...],
    initial: set[Feature],
    start: int,
) -> RetrievalPlan:
    features = set(initial)
    selected = []
    entry = start
    for index in range(start, len(cards)):
        card = cards[index]
        if card is None:
            break
        edge = lineage.transitions[index]
        if not edge.predecessor_validation.accepted or not edge.roundtrip_validation.accepted:
            break
        if set(card.provides).issubset(features):
            if not selected:
                entry = index + 1
            continue
        if set(card.conflicts) & features or not set(card.requires).issubset(features):
            break

        selected.append(card)
        features.update(card.provides)
    return RetrievalPlan(lineage, entry, tuple(selected))


def _scope_match(card: SkillCard, target: TargetContext) -> bool:
    scope = card.scope
    return all(
        value in allowed or _WILDCARD in allowed
        for value, allowed in (
            (target.case, scope.cases),
            (target.language, scope.languages),
            (target.platform, scope.platforms),
        )
    )


def _contract_rank(
    source: Kernel,
    target: TargetContext,
    problem: ProblemSpec | None,
    abi: KernelABI | None,
) -> tuple[float, float] | None:
    # Architecture support is explicit; newer GPUs are not assumed supersets.
    if any(getattr(source.context, name) != getattr(target, name)
           for name in ("case", "language", "platform")):
        return None
    shape_rank = _abi_rank(source.abi, abi)
    if shape_rank is None:
        return None

    semantic_rank = 0.0
    if source.problem is not None and problem is not None:
        source_words = set(re.findall(r"\w+", source.problem.statement.lower()))
        target_words = set(re.findall(r"\w+", problem.statement.lower()))
        words = source_words | target_words
        semantic_rank = len(source_words & target_words) / len(words) if words else 0.0
    return semantic_rank, shape_rank


def _abi_rank(source: KernelABI | None, target: KernelABI | None) -> float | None:
    if source is None or target is None:
        return 0.0
    if len(source.inputs) != len(target.inputs) or len(source.outputs) != len(target.outputs):
        return None

    similarities = []
    for before, after in zip((*source.inputs, *source.outputs), (*target.inputs, *target.outputs)):
        if before.dtype and after.dtype and before.dtype != after.dtype:
            return None
        if len(before.shape) != len(after.shape):
            return None
        if any(key in before.constraints and key in after.constraints
               and before.constraints[key] != after.constraints[key]
               for key in _LAYOUT_FIELDS):
            return None
        if not _stride_match(before, after):
            return None
        for left, right in zip(before.shape, after.shape):
            if isinstance(left, int) and isinstance(right, int) and min(left, right) > 0:
                similarities.append(min(left, right) / max(left, right))
    return sum(similarities) / len(similarities) if similarities else 0.0


def _stride_match(source: ABIValue, target: ABIValue) -> bool:
    if all(_STRIDE_FIELD not in value.constraints for value in (source, target)):
        return True
    strides = [value.constraints.get(_STRIDE_FIELD) for value in (source, target)]
    for value, stride in zip((source, target), strides):
        if not isinstance(stride, list) or len(stride) != len(value.shape):
            return False
        if any(type(item) is not int or item <= 0 for item in stride):
            return False
    if source.shape == target.shape and strides[0] == strides[1]:
        return True

    # Compare memory-axis order across shapes; ambiguous strides need exact reuse.
    if any(len(set(stride)) != len(stride) for stride in strides):
        return False
    return sorted(range(len(strides[0])), key=strides[0].__getitem__) == sorted(
        range(len(strides[1])), key=strides[1].__getitem__,
    )


def _gain(plan: RetrievalPlan) -> float:
    """Use a measured span, never multiply independent card speedups."""

    before = plan.lineage.states[plan.entry_index].validation
    after_id = plan.skills[-1].evidence[0].after_fingerprint
    after = next(state.validation for state in plan.lineage.states
                 if state.fingerprint == after_id)
    if before is None or after is None or not before.accepted or not after.accepted:
        return 0.0
    if before.latency_ms is None or not after.latency_ms:
        return 0.0
    return before.latency_ms / after.latency_ms


__all__ = ["RetrievalPlan", "retrieve_paths"]

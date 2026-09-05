"""Select applicable skills in memory order."""

from collections.abc import Sequence

from klineage.kernel import TargetContext
from klineage.memory.skillcard import SkillAdmission, SkillCard

_WILDCARD = "*"


def retrieve(
    skills: Sequence[SkillCard],
    target: TargetContext,
    *,
    skill_admission: SkillAdmission = SkillAdmission.OFF,
) -> tuple[SkillCard, ...]:
    """Filter scope and prerequisites without changing application order."""

    if not isinstance(target, TargetContext):
        raise TypeError("target must be a TargetContext")
    if not isinstance(skill_admission, SkillAdmission):
        raise TypeError("skill_admission must be a SkillAdmission")
    if isinstance(skills, (str, bytes)) or not isinstance(skills, Sequence):
        raise TypeError("skills must be a sequence of SkillCards")

    selected: dict[str, SkillCard] = {}
    actions = set(target.prior_actions)
    for card in skills:
        if not isinstance(card, SkillCard):
            raise TypeError("skills contains a non-SkillCard value")
        if skill_admission is SkillAdmission.ON and not card.admitted:
            continue

        scope = card.scope
        dimensions = (
            (target.case, scope.cases),
            (target.language, scope.languages),
            (target.platform, scope.platforms),
        )
        if any(
            value not in allowed and _WILDCARD not in allowed
            for value, allowed in dimensions
        ):
            continue
        if not set(scope.prior_actions).issubset(actions):
            continue

        # Stable IDs remove repeated memory entries without reordering skills.
        previous = selected.get(card.skill_id)
        if previous is not None and previous != card:
            raise ValueError(f"conflicting duplicate skill_id {card.skill_id!r}")
        selected[card.skill_id] = card

    return tuple(selected.values())


__all__ = ["retrieve"]

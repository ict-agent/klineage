"""Filter skills by scope in memory order."""

from collections.abc import Sequence

from klineage.artifact.kernel import Kernel
from klineage.contract import ProblemSpec
from klineage.memory.skillcard import SkillCard
from klineage.utils import string_tuple

WILDCARD = "*"


def retrieve(
    skills: Sequence[SkillCard],
    target: Kernel | ProblemSpec,
    *,
    exclude_skills: Sequence[str] = (),
) -> tuple[SkillCard, ...]:
    """Filter by scope; the caller checks source prerequisites."""

    if not isinstance(target, (Kernel, ProblemSpec)):
        raise TypeError("target must be a Kernel or ProblemSpec")
    if isinstance(skills, (str, bytes)) or not isinstance(skills, Sequence):
        raise TypeError("skills must be a sequence of SkillCards")
    if isinstance(exclude_skills, (str, bytes)) or not isinstance(
        exclude_skills, Sequence
    ):
        raise TypeError("exclude_skills must be a sequence of strings")

    excluded = set(string_tuple(exclude_skills, "exclude_skills"))
    selected: dict[str, SkillCard] = {}
    problem = target.problem if isinstance(target, Kernel) else target
    for card in skills:
        if not isinstance(card, SkillCard):
            raise TypeError("skills contains a non-SkillCard value")
        if card.skill_id in excluded:
            continue

        scope = card.scope
        dimensions = (
            (problem.definition["op_type"], scope.cases),
            (problem.language, scope.languages),
            (problem.platform, scope.platforms),
        )
        if any(
            value not in allowed and WILDCARD not in allowed
            for value, allowed in dimensions
        ):
            continue

        # Stable IDs remove repeated memory entries without reordering skills.
        previous = selected.get(card.skill_id)
        if previous is not None and previous != card:
            raise ValueError(f"conflicting duplicate skill_id {card.skill_id!r}")
        selected[card.skill_id] = card

    return tuple(selected.values())


__all__ = ["retrieve"]

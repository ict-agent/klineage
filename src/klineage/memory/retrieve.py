"""Filter skills by scope in memory order."""

from collections.abc import Sequence
from pathlib import Path

from klineage.artifact.kernel import Kernel
from klineage.artifact.problem import Problem, load_problem
from klineage.constants import SKILL_FILE
from klineage.memory.skillcard import SkillCard
from klineage.memory.storage import load_skill
from klineage.utils import string_tuple

WILDCARD = "*"


def retrieve(
    skills: Sequence[SkillCard | Path] | Path,
    target: Kernel | Problem,
    *,
    exclude_skills: Sequence[str] = (),
) -> tuple[SkillCard, ...]:
    """Filter cards by operator, language, platform and excluded IDs.

    Accept cards, SKILL.md paths, or a directory searched in sorted path order.
    Matching duplicates collapse without reordering. The caller checks source
    prerequisites; retrieval does not rank techniques or predict performance.
    """

    if not isinstance(target, (Kernel, Problem)):
        raise TypeError("target must be a Kernel or Problem")
    if isinstance(skills, Path):
        path = skills.expanduser()
        skills = sorted(path.rglob(SKILL_FILE)) if path.is_dir() else (path,)
    if isinstance(skills, (str, bytes)) or not isinstance(skills, Sequence):
        raise TypeError("skills must be a Path or a sequence of SkillCards/Paths")
    if isinstance(exclude_skills, (str, bytes)) or not isinstance(
        exclude_skills, Sequence
    ):
        raise TypeError("exclude_skills must be a sequence of strings")

    excluded = set(string_tuple(exclude_skills, "exclude_skills"))
    selected: dict[str, SkillCard] = {}
    problem = load_problem(target) if isinstance(target, Kernel) else target
    for card in skills:
        if isinstance(card, Path):
            card = load_skill(card)
        if not isinstance(card, SkillCard):
            raise TypeError("skills entries must be SkillCards or SKILL.md Paths")
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

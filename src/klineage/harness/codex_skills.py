"""Present stored SkillCards as discoverable Codex skills."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

import yaml

from klineage.constants import (
    CODEX_SKILL_DESCRIPTION_LIMIT,
    CODEX_SKILL_HASH_LENGTH,
    CODEX_SKILL_NAME_LIMIT,
    MEMORY_DIR,
    SKILL_FILE,
)
from klineage.memory import load_skill
from klineage.utils import string_tuple


def render_memory(
    memory: Path, *, exclude_skills: Sequence[str] = ()
) -> dict[Path, str]:
    """Render native entrypoints and unchanged recipes without modifying memory."""
    excluded = set(string_tuple(exclude_skills, "exclude_skills"))
    files = {}
    cards = {}
    for path in sorted(memory.rglob(SKILL_FILE)):
        card = load_skill(path)
        if card.skill_id in excluded:
            continue
        previous = cards.get(card.skill_id)
        if previous is not None:
            if previous != card:
                raise ValueError(f"conflicting duplicate skill_id {card.skill_id!r}")
            continue
        cards[card.skill_id] = card

        # Names stay readable without colliding after normalization or truncation.
        digest = hashlib.sha256(card.skill_id.encode()).hexdigest()[
            :CODEX_SKILL_HASH_LENGTH
        ]
        stem = re.sub(r"[^a-z0-9]+", "-", card.skill_id.lower()).strip("-") or "skill"
        prefix = MEMORY_DIR + "-"
        width = CODEX_SKILL_NAME_LIMIT - len(prefix) - len(digest) - 1
        name = f"{prefix}{stem[:width].rstrip('-')}-{digest}"
        scope = card.scope
        description = (
            f"{card.intent} Use for {', '.join(scope.cases)} kernels "
            f"({', '.join(scope.languages)}; {', '.join(scope.platforms)})."
        )
        description = description.replace("<", " less than ").replace(
            ">", " greater than "
        )
        # Codex reads name and description for discovery; the four card fields
        # ride along under metadata so a native skill stays a complete SkillCard.
        header = yaml.safe_dump(
            {
                "name": name,
                "description": description[:CODEX_SKILL_DESCRIPTION_LIMIT].strip(),
                "metadata": card.to_metadata(),
            },
            sort_keys=False,
            allow_unicode=True,
        )
        # The recipe is the skill body: one read delivers the whole technique, so
        # no separate reference file is needed. Codex's own frontmatter carries the
        # discovery description, keeping the four card fields out of the header.
        files[Path(name) / SKILL_FILE] = (
            f"---\n{header}---\n\n{card.body}\n"
        )
    return files

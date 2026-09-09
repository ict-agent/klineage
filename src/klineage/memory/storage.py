"""Markdown SkillCards and flat JSON skill memories."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from pathlib import Path

import yaml

from klineage.memory.skillcard import SkillCard

SKILL_FILE = "SKILL.md"
SKILL_MARKDOWN = re.compile(r"---\n(.*?)\n---\n\n(.*)", re.DOTALL)


def save_skill(card: SkillCard, path: str | os.PathLike[str]) -> Path:
    """Write one SkillCard as Markdown."""

    header = yaml.safe_dump(card.to_metadata(), sort_keys=False, allow_unicode=True)
    return _write(path, f"---\n{header}---\n\n{card.body}\n")


def load_skill(path: str | os.PathLike[str]) -> SkillCard:
    """Restore a SkillCard from YAML metadata and Markdown instructions."""

    text = Path(path).expanduser().read_text(encoding="utf-8")
    match = SKILL_MARKDOWN.fullmatch(text)
    if match is None:
        raise ValueError("SKILL.md requires YAML frontmatter and Markdown sections")
    try:
        header = yaml.safe_load(match[1])
    except yaml.YAMLError as error:
        raise ValueError("invalid SKILL.md YAML") from error
    if not isinstance(header, dict):
        raise TypeError("SKILL.md frontmatter must be a mapping")
    if set(header) != {"skill_id", "intent", "preconditions", "scope"}:
        raise ValueError("SKILL.md frontmatter requires four metadata fields")
    return SkillCard.from_dict({**header, "body": match[2]})


def save_memory(
    skills: Sequence[SkillCard],
    path: str | os.PathLike[str],
) -> Path:
    """Atomically persist a flat, de-duplicated SkillCard sequence."""

    cards = unique_cards(skills)
    payload = json.dumps(
        [card.to_dict() for card in cards], ensure_ascii=False, indent=2
    )
    return _write(path, payload + "\n")


def load_memory(path: str | os.PathLike[str]) -> tuple[SkillCard, ...]:
    """Load a flat SkillCard memory written by :func:`save_memory`."""

    values = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise TypeError("skill memory file must contain a JSON array")
    cards: list[SkillCard] = []
    for value in values:
        if not isinstance(value, dict):
            raise TypeError("skill-memory contains a non-object card")
        cards.append(SkillCard.from_dict(value))
    return unique_cards(cards)


def unique_cards(skills: Sequence[SkillCard]) -> tuple[SkillCard, ...]:
    if isinstance(skills, (str, bytes)) or not isinstance(skills, Sequence):
        raise TypeError("skills must be a sequence of SkillCards")
    by_id: dict[str, SkillCard] = {}
    for card in skills:
        if not isinstance(card, SkillCard):
            raise TypeError("skills contains a non-SkillCard value")
        previous = by_id.get(card.skill_id)
        if previous is not None and previous != card:
            raise ValueError(f"conflicting duplicate skill_id {card.skill_id!r}")
        by_id[card.skill_id] = card
    return tuple(by_id.values())


def _write(path: str | os.PathLike[str], text: str) -> Path:
    destination = Path(path).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, destination)
    return destination


__all__ = ["load_memory", "load_skill", "save_memory", "save_skill"]

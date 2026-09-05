"""Plain JSON persistence for lineages and flat SkillCard memories."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

from klineage.memory.lineage import Lineage
from klineage.memory.skillcard import SkillCard


def save_lineage(
    lineage: Lineage,
    path: str | os.PathLike[str],
) -> Path:
    """Atomically persist one full lineage."""

    return _write(path, lineage.to_dict())


def load_lineage(path: str | os.PathLike[str]) -> Lineage:
    """Load a lineage written by :func:`save_lineage`."""

    value = _read(path)
    if not isinstance(value, dict):
        raise TypeError("lineage file must contain a JSON object")
    return Lineage.from_dict(value)


def save_memory(
    skills: Sequence[SkillCard],
    path: str | os.PathLike[str],
) -> Path:
    """Atomically persist a flat, de-duplicated SkillCard sequence."""

    cards = _unique_cards(skills)
    return _write(path, [card.to_dict() for card in cards])


def load_memory(path: str | os.PathLike[str]) -> tuple[SkillCard, ...]:
    """Load a flat SkillCard memory written by :func:`save_memory`."""

    values = _read(path)
    if not isinstance(values, list):
        raise TypeError("skill memory file must contain a JSON array")
    cards: list[SkillCard] = []
    for value in values:
        if not isinstance(value, dict):
            raise TypeError("skill-memory contains a non-object card")
        cards.append(SkillCard.from_dict(value))
    return _unique_cards(cards)


def _unique_cards(skills: Sequence[SkillCard]) -> tuple[SkillCard, ...]:
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


def _write(path: str | os.PathLike[str], payload: object) -> Path:
    destination = Path(path).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def _read(path: str | os.PathLike[str]) -> object:
    source = Path(path).expanduser()
    value = json.loads(source.read_text(encoding="utf-8"))
    return value


__all__ = ["load_lineage", "load_memory", "save_lineage", "save_memory"]

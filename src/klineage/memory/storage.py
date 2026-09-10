"""Persist SkillCards as YAML metadata and Markdown instructions."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from klineage.constants import SKILL_FILE
from klineage.memory.skillcard import SkillCard

SKILL_MARKDOWN = re.compile(r"---\n(.*?)\n---\n\n(.*)", re.DOTALL)


def save_skill(card: SkillCard, path: str | os.PathLike[str]) -> Path:
    """Write four-field YAML frontmatter and the card's independent Markdown body.

    Create parent directories, atomically replace path, and return its absolute
    path. Keep SKILL.md outside submission/ and Kernel.source_files.
    """

    header = yaml.safe_dump(card.to_metadata(), sort_keys=False, allow_unicode=True)
    destination = Path(path).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(f"---\n{header}---\n\n{card.body}\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def load_skill(path: str | os.PathLike[str]) -> SkillCard:
    """Read a SKILL.md file with four metadata fields and an independent Markdown body.

    Require skill_id, intent, preconditions, and scope in YAML frontmatter.
    Return a SkillCard; malformed metadata or an empty body raises.
    """

    text = Path(path).expanduser().read_text(encoding="utf-8")
    match = SKILL_MARKDOWN.fullmatch(text)
    if match is None:
        raise ValueError(
            f"{SKILL_FILE} requires YAML frontmatter and Markdown sections"
        )
    try:
        header = yaml.safe_load(match[1])
    except yaml.YAMLError as error:
        raise ValueError(f"invalid {SKILL_FILE} YAML") from error
    if not isinstance(header, dict):
        raise TypeError(f"{SKILL_FILE} frontmatter must be a mapping")
    if set(header) != {"skill_id", "intent", "preconditions", "scope"}:
        raise ValueError(f"{SKILL_FILE} frontmatter requires four metadata fields")
    return SkillCard.from_dict({**header, "body": match[2]})


__all__ = ["load_skill", "save_skill"]

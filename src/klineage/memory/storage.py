"""Persist SkillCards as YAML metadata and Markdown instructions."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

from klineage.serialization import raw_text
from klineage.artifact.tree import ArtifactKind, Node
from klineage.constants import SKILL_FILE
from klineage.memory.skillcard import SkillCard


def save_skill(card: SkillCard, path: str | os.PathLike[str]) -> Path:
    """Write four-field YAML frontmatter and the card's independent Markdown body.

    Create parent directories, atomically replace path, and return its absolute
    path. Keep SKILL.md outside the Kernel submission subtree.
    """

    destination = Path(path).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as directory:
        temporary = Path(directory) / destination.name
        temporary.write_text(skill_text(card), encoding="utf-8")
        os.replace(temporary, destination)
    return destination


def load_skill(path: str | os.PathLike[str]) -> SkillCard:
    """Read a SKILL.md file with four metadata fields and an independent Markdown body.

    Require skill_id, intent, preconditions, and scope in YAML frontmatter.
    Return a SkillCard; malformed metadata or an empty body raises.
    """

    return parse_skill(Path(path).expanduser().read_text(encoding="utf-8"))


def skill_text(card: SkillCard) -> str:
    return raw_text(card.node(SKILL_FILE))


def parse_skill(text: str) -> SkillCard:
    return SkillCard.from_tree(
        ArtifactKind.SKILLCARD, Node.directory(".", (Node.file(SKILL_FILE, text),))
    )


__all__ = ["load_skill", "save_skill"]

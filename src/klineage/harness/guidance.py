"""Workspace instructions and skill memory for both agent runners."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

import yaml

from klineage.constants import (
    AGENT_DOCS_END,
    AGENT_DOCS_START,
    AGENT_FILES,
    MEMORY_DIR,
    SKILL_DESCRIPTION_LIMIT,
    SKILL_FILE,
    SKILL_HASH_LENGTH,
    SKILL_NAME_LIMIT,
)
from klineage.memory import load_skill
from klineage.utils import (
    DirectoryMode,
    operation_id,
    safe_directory,
    safe_file,
    string_tuple,
)


def write_agent_docs(workdir: Path):
    """Render workspace/ instructions, preserving user-authored content."""

    from klineage.prompts import render_prompt

    block = "\n\n".join((AGENT_DOCS_START, render_prompt("agents"), AGENT_DOCS_END))
    updates = []
    for name in AGENT_FILES:
        path = workdir / name
        if path.is_symlink():
            raise ValueError(f"agent instructions must not be a symlink: {path}")
        mode = None
        try:
            with safe_file(path) as stream:
                original = stream.read()
                mode = stat.S_IMODE(os.fstat(stream.fileno()).st_mode)
        except FileNotFoundError:
            original = ""
        if name != AGENT_FILES[0] and not original.strip():
            continue
        updated = replace_block(original, block)
        if updated != original:
            updates.append((path, updated, mode))

    # An existing override takes precedence over AGENTS.md; update both.
    for path, updated, mode in updates:
        with safe_directory(path.parent, DirectoryMode.CREATE) as directory:
            temporary = operation_id("agent-docs") + ".tmp"
            try:
                with safe_file(path.parent / temporary, "w") as stream:
                    stream.write(updated)
                    if mode is not None:
                        os.fchmod(stream.fileno(), mode)
                os.replace(temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory)
            finally:
                with suppress(FileNotFoundError):
                    os.unlink(temporary, dir_fd=directory)


def replace_block(original: str, block: str) -> str:
    starts, ends = original.count(AGENT_DOCS_START), original.count(AGENT_DOCS_END)
    if starts == ends == 0:
        separator = "" if not original else "\n" if original.endswith("\n") else "\n\n"
        return original + separator + block + "\n"
    if starts != 1 or ends != 1:
        raise ValueError("malformed generated agent instructions")
    start, end = original.index(AGENT_DOCS_START), original.index(AGENT_DOCS_END)
    if end < start:
        raise ValueError("malformed generated agent instructions")
    return original[:start] + block + original[end + len(AGENT_DOCS_END) :]


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
            :SKILL_HASH_LENGTH
        ]
        stem = re.sub(r"[^a-z0-9]+", "-", card.skill_id.lower()).strip("-") or "skill"
        prefix = MEMORY_DIR + "-"
        width = SKILL_NAME_LIMIT - len(prefix) - len(digest) - 1
        name = f"{prefix}{stem[:width].rstrip('-')}-{digest}"
        scope = card.scope
        description = (
            f"{card.intent} Use for {', '.join(scope.cases)} kernels "
            f"({', '.join(scope.languages)}; {', '.join(scope.platforms)})."
        )
        description = description.replace("<", " less than ").replace(
            ">", " greater than "
        )
        # Agents read name and description for discovery; the four card fields
        # ride along under metadata so a native skill stays a complete SkillCard.
        header = yaml.safe_dump(
            {
                "name": name,
                "description": description[:SKILL_DESCRIPTION_LIMIT].strip(),
                "metadata": card.to_metadata(),
            },
            sort_keys=False,
            allow_unicode=True,
        )
        # The recipe is the skill body: one read delivers the whole technique, so
        # no separate reference file is needed. The entrypoint frontmatter carries the
        # discovery description, keeping the four card fields out of the header.
        files[Path(name) / SKILL_FILE] = (
            f"---\n{header}---\n\n{card.body}\n"
        )
    return files

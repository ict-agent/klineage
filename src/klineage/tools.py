"""Workspace guidance and skill retrieval."""

from __future__ import annotations

import os
import stat
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from klineage.constants import (
    AGENT_DOCS_END,
    AGENT_DOCS_START,
    AGENT_FILES,
    SKILL_FILE,
)
from klineage.utils import DirectoryMode, operation_id, safe_directory, safe_file

if TYPE_CHECKING:
    from klineage.artifact.kernel import Kernel
    from klineage.memory import SkillCard


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


def retrieve(
    kernel: Kernel,
    memory: Sequence[SkillCard | Path] | Path,
    *,
    exclude_skills: Sequence[str] = (),
) -> tuple[SkillCard, ...]:
    """Return skills matching the Kernel's operator, language, and platform.

    Memory accepts ordered SkillCards/SKILL.md paths, one card path, or a directory
    searched recursively for SKILL.md in sorted path order. Empty memory returns
    (). Exclude skill IDs with exclude_skills; matching duplicates collapse.
    Check each card's prerequisites against the source before applying it.
    Rank candidates using available profile evidence; this function only filters
    scope and exclusions, without checking prerequisites or ranking skills.
    """
    from klineage.artifact.kernel import Kernel
    from klineage.memory import SkillCard, load_skill
    from klineage.memory import retrieve as select_skills

    if not isinstance(kernel, Kernel):
        raise TypeError("kernel must be a Kernel")
    if isinstance(memory, Path):
        path = memory.expanduser()
        memory = sorted(path.rglob(SKILL_FILE)) if path.is_dir() else (path,)
    if isinstance(memory, (str, bytes)) or not isinstance(memory, Sequence):
        raise TypeError("memory must be a Path or a sequence of SkillCards/Paths")

    cards = []
    for item in memory:
        if not isinstance(item, (SkillCard, Path)):
            raise TypeError("memory entries must be SkillCards or SKILL.md Paths")
        cards.append(item if isinstance(item, SkillCard) else load_skill(item))
    return select_skills(cards, kernel, exclude_skills=exclude_skills)


__all__ = [
    "retrieve",
    "write_agent_docs",
]

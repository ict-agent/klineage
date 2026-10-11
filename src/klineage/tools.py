"""Agent functions and their generated workspace guidance."""

from __future__ import annotations

import importlib
import inspect
import os
import stat
import sys
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from klineage.constants import (
    AGENT_FILES,
    AGENT_MODULES,
    AGENTS_TEMPLATE,
    FUNCTION_END,
    FUNCTION_START,
    SKILL_FILE,
)
from klineage.utils import DirectoryMode, operation_id, safe_directory, safe_file

if TYPE_CHECKING:
    from klineage.artifact.kernel import Kernel
    from klineage.memory import SkillCard

_FUNCTIONS: dict[str, Callable[..., Any]] = {}


def agent_function[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Register a documented function; place below @classmethod for factories."""

    if not inspect.isfunction(function):
        raise TypeError("agent_function requires a Python function")
    if (
        not all(part.isidentifier() for part in function.__qualname__.split("."))
        or function.__module__ == "__main__"
    ):
        raise ValueError("agent functions must belong to an importable module or class")
    if not inspect.getdoc(function):
        raise ValueError("agent functions require a docstring")
    _FUNCTIONS[f"{api_module(function)}.{function.__qualname__}"] = function
    return function


def api_module(function) -> str:
    """Expose documented functions through package APIs."""
    module = function.__module__
    if module.startswith(("klineage.artifact.", "klineage.backend.", "klineage.harness.", "klineage.server.")):
        return module.rpartition(".")[0]
    return module


def function_docs(*, client=None) -> str:
    """Load built-in functions and render the registered API catalog."""
    for module in AGENT_MODULES:
        importlib.import_module(module)
    sections = [
        FUNCTION_START,
        AGENTS_TEMPLATE.rstrip(),
        "## KLineage Python functions",
        "Import and call these functions from Python. Follow their documented contracts.",
    ]
    for name, function in sorted(_FUNCTIONS.items()):
        if client is not None and not name.startswith("klineage.server.Client."):
            continue
        # Resolve descriptors after class creation so factories omit the bound cls.
        parts = function.__qualname__.split(".")
        module = api_module(function)
        target: Any = sys.modules[module]
        for part in parts:
            target = getattr(target, part)
        signature = str(inspect.signature(target))
        sections.append(
            f"### {name}\n\n"
            f"`from {module} import {parts[0]}`\n\n"
            f"Signature: `{function.__qualname__}{signature}`\n\n"
            f"{inspect.getdoc(target)}"
        )
    return "\n\n".join((*sections, FUNCTION_END))


def write_agent_docs(workdir: Path, *, client=None):
    """Write workspace/ instructions, preserving user-authored content."""

    block = function_docs(client=client)
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
        updated = _replace_block(original, block)
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


def _replace_block(original: str, block: str) -> str:
    starts, ends = original.count(FUNCTION_START), original.count(FUNCTION_END)
    if starts == ends == 0:
        separator = "" if not original else "\n" if original.endswith("\n") else "\n\n"
        return original + separator + block + "\n"
    if starts != 1 or ends != 1:
        raise ValueError("malformed generated agent-function block")
    start, end = original.index(FUNCTION_START), original.index(FUNCTION_END)
    if end < start:
        raise ValueError("malformed generated agent-function block")
    return original[:start] + block + original[end + len(FUNCTION_END) :]


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
    "agent_function",
    "function_docs",
    "retrieve",
    "write_agent_docs",
]

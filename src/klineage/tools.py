"""Agent functions and their generated workspace guidance."""

from __future__ import annotations

import importlib
import inspect
import os
import stat
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

from klineage.constants import (
    AGENT_FILES,
    AGENT_MODULES,
    FUNCTION_END,
    FUNCTION_START,
    SKILL_FILE,
)

if TYPE_CHECKING:
    from klineage.artifact.kernel import Kernel
    from klineage.harness.profiling import ProfileOptions
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
    _FUNCTIONS[f"{function.__module__}.{function.__qualname__}"] = function
    return function


def function_docs() -> str:
    """Load built-in functions and render the registered API catalog."""
    for module in AGENT_MODULES:
        importlib.import_module(module)

    sections = [
        FUNCTION_START,
        "## KLineage Python functions",
        "Import and call these functions from Python. Follow their documented contracts.",
    ]
    for name, function in sorted(_FUNCTIONS.items()):
        # Resolve descriptors after class creation so factories omit the bound cls.
        parts = function.__qualname__.split(".")
        target: Any = sys.modules[function.__module__]
        for part in parts:
            target = getattr(target, part)
        signature = str(inspect.signature(target))
        sections.append(
            f"### {name}\n\n"
            f"`from {function.__module__} import {parts[0]}`\n\n"
            f"Signature: `{function.__qualname__}{signature}`\n\n"
            f"{inspect.getdoc(target)}"
        )
    return "\n\n".join((*sections, FUNCTION_END))


def write_agent_docs(workdir: Path):
    """Refresh the generated block, preserving other workspace instructions."""

    block = function_docs()
    updates = []
    for name in AGENT_FILES:
        path = workdir / name
        if path.is_symlink():
            raise ValueError(f"agent instructions must not be a symlink: {path}")
        original = path.read_bytes().decode("utf-8") if path.exists() else ""
        if name != AGENT_FILES[0] and not original.strip():
            continue
        updated = _replace_block(original, block)
        if updated != original:
            updates.append((path, updated))

    # An existing override takes precedence over AGENTS.md; update both.
    for path, updated in updates:
        temporary = None
        try:
            with NamedTemporaryFile(
                dir=workdir, mode="w", encoding="utf-8", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(updated)
            if path.exists():
                temporary.chmod(stat.S_IMODE(path.stat().st_mode))
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


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


@agent_function
def profile(
    kernel: Kernel,
    workdir: Path | None = None,
    *,
    options: ProfileOptions | None = None,
) -> dict[str, Any]:
    """Capture CUDA NCU metrics for a Kernel's frozen sources.

    Write build artifacts and evaluation evidence under workdir (default: cwd).
    Use klineage.harness.profiling.ProfileOptions for custom options.
    Return metrics, device metadata, and report paths. Hygon/Ascend counter
    profiling is unsupported. Inspect the source and metrics to diagnose
    bottlenecks; use klineage.harness.evaluate for correctness and latency.
    """
    from klineage.artifact.kernel import Kernel
    from klineage.harness.eval import capture
    from klineage.harness.profiling import ProfileOptions

    if not isinstance(kernel, Kernel):
        raise TypeError("kernel must be a Kernel")
    work = Path.cwd() if workdir is None else workdir.expanduser().resolve()
    return capture(kernel, work, options or ProfileOptions())


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
    "profile",
    "retrieve",
    "write_agent_docs",
]

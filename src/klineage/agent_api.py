"""Publish decorated Python functions in workspace instructions."""

from __future__ import annotations

import inspect
import os
import stat
import sys
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

FUNCTION_START = "<!-- klineage:agent-functions:start -->"
FUNCTION_END = "<!-- klineage:agent-functions:end -->"
AGENT_FILES = ("AGENTS.md", "AGENTS.override.md")
_FUNCTIONS: dict[str, Callable[..., Any]] = {}


def agent_function[F: Callable[..., Any]](function: F) -> F:
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
    sections = [
        FUNCTION_START,
        "## KLineage Python functions",
        "Import and call these functions from Python. Follow their documented contracts.",
    ]
    for name, function in sorted(_FUNCTIONS.items()):
        # Resolve descriptors after class creation so factories omit the bound cls.
        target = sys.modules[function.__module__]
        for part in function.__qualname__.split("."):
            target = getattr(target, part)
        signature = str(inspect.signature(target))
        imported = function.__qualname__.split(".", 1)[0]
        sections.append(
            f"### {name}\n\n"
            f"`from {function.__module__} import {imported}`\n\n"
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
        if name != AGENT_FILES[0] and not path.exists() and not path.is_symlink():
            continue
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


__all__ = ["agent_function"]

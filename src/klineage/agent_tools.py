"""Agent-facing Python functions and artifact types."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from klineage.agent_api import agent_function
from klineage.backend import detect_backend, get_backend
from klineage.constants import SKILL_FILE
from klineage.harness import capture, evaluate, inspect_problem
from klineage.harness.artifacts import load_kernel, read_source_tree, save_kernel
from klineage.kernel import Kernel
from klineage.memory import SkillCard, load_skill, save_skill
from klineage.memory import retrieve as select_skills
from klineage.profiling import ProfileOptions
from klineage.repository import stage_repository


@agent_function
def profile(
    kernel: Kernel,
    workdir: Path | None = None,
    *,
    options: ProfileOptions | None = None,
) -> dict[str, Any]:
    """Capture CUDA NCU metrics for a Kernel's frozen sources.

    Write build artifacts and evaluation evidence under workdir (default: cwd).
    Return metrics, device metadata, and report paths. Hygon/Ascend counter
    profiling is unsupported. Inspect the source and metrics to diagnose
    bottlenecks; use klineage.harness.evaluate for correctness and latency.
    """
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
    "Kernel",
    "SkillCard",
    "detect_backend",
    "evaluate",
    "get_backend",
    "inspect_problem",
    "load_kernel",
    "load_skill",
    "profile",
    "read_source_tree",
    "retrieve",
    "save_kernel",
    "save_skill",
    "stage_repository",
]

"""Run the kernel workflow through fixed artifact directories."""

import os
from collections.abc import Iterator
from pathlib import Path

from klineage._utils import new_workdir, operation_id
from klineage.action.action import MAX_RETRIES, TIMEOUT
from klineage.action.apply import Apply
from klineage.action.decompose import Decompose
from klineage.action.init import Init
from klineage.action.profile import Profile
from klineage.action.retrieve import Retrieve
from klineage.errors import StructuredOutputError
from klineage.harness.artifacts import load_kernel
from klineage.kernel import Kernel
from klineage.memory.skillcard import SkillCard
from klineage.memory.storage import SKILL_FILE, load_skill, save_skill

MAX_DECOMPOSE_STEPS = 15
MAX_APPLY_STEPS = 15


def workflow(
    problem: str | os.PathLike[str],
    repo: str | os.PathLike[str],
    expert_kernel: str | os.PathLike[str],
    max_decompose_step: int = MAX_DECOMPOSE_STEPS,
    enable_verifier: bool = True,
    *,
    max_apply_step: int = MAX_APPLY_STEPS,
    workdir: Path | None = None,
    timeout: int = TIMEOUT,
    max_retries: int = MAX_RETRIES,
) -> Kernel:
    if type(max_decompose_step) is not int or max_decompose_step <= 0:
        raise ValueError("max_decompose_step must be a positive integer")
    if type(max_apply_step) is not int or max_apply_step < 0:
        raise ValueError("max_apply_step must be a nonnegative integer")

    # A fresh root prevents previous outputs from entering this run.
    workdir = Path(workdir or new_workdir("workflow")).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=False)

    skill_paths = []
    for current_dir, card in _decompose(
        problem,
        repo,
        expert_kernel,
        max_decompose_step,
        enable_verifier,
        workdir=workdir,
        timeout=timeout,
        max_retries=max_retries,
    ):
        if card is not None:
            skill_paths.append(current_dir / SKILL_FILE)

    # Start retrieval with measurements of the final predecessor.
    profile_dir = workdir / "profile" / "0"
    Profile(
        current_dir,
        workdir=profile_dir,
        enable_verifier=enable_verifier,
        timeout=timeout,
        max_retries=max_retries,
    ).run()
    current_dir = profile_dir
    applied_skills = []

    for step in range(max_apply_step):
        retrieve_dir = workdir / "retrieve" / str(step)
        Retrieve(
            current_dir,
            skill_paths,
            exclude_skills=applied_skills,
            workdir=retrieve_dir,
            enable_verifier=enable_verifier,
            timeout=timeout,
            max_retries=max_retries,
        ).run()
        selected_path = retrieve_dir / SKILL_FILE
        if not selected_path.is_file():
            break
        selected = load_skill(selected_path)

        apply_dir = workdir / "apply" / str(step)
        Apply(
            current_dir,
            selected_path,
            workdir=apply_dir,
            enable_verifier=enable_verifier,
            timeout=timeout,
            max_retries=max_retries,
        ).run()

        # Recompute bottlenecks before selecting another skill.
        profile_dir = workdir / "profile" / str(step + 1)
        Profile(
            apply_dir,
            workdir=profile_dir,
            enable_verifier=enable_verifier,
            timeout=timeout,
            max_retries=max_retries,
        ).run()
        current_dir = profile_dir
        applied_skills.append(selected.skill_id)

    return load_kernel(current_dir)


def init_memory(
    problem: str | os.PathLike[str],
    repo: str | os.PathLike[str],
    expert_kernel: str | os.PathLike[str],
    max_decompose_step: int = MAX_DECOMPOSE_STEPS,
    enable_verifier: bool = False,
    *,
    workdir: Path | None = None,
    timeout: int = TIMEOUT,
    max_retries: int = MAX_RETRIES,
    memory_dir: str | os.PathLike[str],
) -> tuple[Path, ...]:
    """Extract skills into memory_dir and return their paths in removal order."""

    if type(max_decompose_step) is not int or max_decompose_step <= 0:
        raise ValueError("max_decompose_step must be a positive integer")

    workdir = Path(workdir or new_workdir("init-memory")).expanduser().resolve()
    memory_dir = Path(memory_dir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=False)
    memory_dir.mkdir(parents=True, exist_ok=True)

    skill_paths = []
    for _, card in _decompose(
        problem,
        repo,
        expert_kernel,
        max_decompose_step,
        enable_verifier,
        workdir=workdir,
        timeout=timeout,
        max_retries=max_retries,
    ):
        if card is None:
            continue

        # Save each passed step immediately; keep repeated IDs in separate directories.
        destination = memory_dir / operation_id(card.skill_id) / SKILL_FILE
        skill_paths.append(save_skill(card, destination))

    return tuple(skill_paths)


def _decompose(
    problem: str | os.PathLike[str],
    repo: str | os.PathLike[str],
    expert_kernel: str | os.PathLike[str],
    max_decompose_step: int,
    enable_verifier: bool,
    *,
    workdir: Path,
    timeout: int,
    max_retries: int,
) -> Iterator[tuple[Path, SkillCard | None]]:
    init_dir = workdir / "init"
    Init(
        problem,
        repo,
        expert_kernel,
        workdir=init_dir,
        enable_verifier=enable_verifier,
        timeout=timeout,
        max_retries=max_retries,
    ).run()

    # Yield only after the action and its optional verifier have succeeded.
    current_dir = init_dir
    before = load_kernel(current_dir)
    for step in range(max_decompose_step):
        decompose_dir = workdir / "decompose" / str(step)
        Decompose(
            current_dir,
            workdir=decompose_dir,
            enable_verifier=enable_verifier,
            timeout=timeout,
            max_retries=max_retries,
        ).run()
        after = load_kernel(decompose_dir)
        if before.problem != after.problem:
            raise StructuredOutputError("Decompose changed the problem")

        skill_path = decompose_dir / SKILL_FILE
        if not skill_path.is_file():
            if before.fingerprint != after.fingerprint or before.name != after.name:
                raise StructuredOutputError(
                    "Decompose changed a kernel without SKILL.md"
                )
            yield decompose_dir, None
            return

        if before.fingerprint == after.fingerprint:
            raise StructuredOutputError(
                "Decompose emitted SKILL.md for an unchanged kernel"
            )

        yield decompose_dir, load_skill(skill_path)
        current_dir = decompose_dir
        before = after


__all__ = ["init_memory", "workflow"]

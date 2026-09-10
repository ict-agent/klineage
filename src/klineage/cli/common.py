"""Shared CLI arguments and decomposition handoffs."""

import argparse
import os
from collections.abc import Iterator
from pathlib import Path

from klineage.action.decompose import Decompose
from klineage.action.init import Init
from klineage.artifact.kernel import Kernel
from klineage.constants import (
    MAX_DECOMPOSE_STEPS,
    MAX_RETRIES,
    SKILL_FILE,
    SUBMISSION_DIRECTORY,
    TIMEOUT,
    RunKind,
)
from klineage.contract import ValueRole
from klineage.errors import StructuredOutputError
from klineage.harness.artifacts import load_kernel
from klineage.memory.skillcard import SkillCard
from klineage.memory.storage import load_skill


def argument_parser(description: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--expert-kernel", required=True)
    parser.add_argument("--max-decompose-step", type=int, default=MAX_DECOMPOSE_STEPS)
    parser.add_argument("--workdir", type=Path, help="Fresh run directory")
    add_run_options(parser)
    return parser


def add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--timeout", type=int, default=TIMEOUT, help="Seconds per action"
    )
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument(
        "--verifier",
        dest="enable_verifier",
        action=argparse.BooleanOptionalAction,
        help="Verify each action before accepting its output",
    )


def decompose_steps(
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
    init_dir = workdir / RunKind.INIT
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
        decompose_dir = workdir / RunKind.DECOMPOSE / str(step)
        Decompose(
            current_dir,
            workdir=decompose_dir,
            enable_verifier=enable_verifier,
            timeout=timeout,
            max_retries=max_retries,
        ).run()
        after, card = read_step(before, decompose_dir, RunKind.DECOMPOSE)
        yield decompose_dir, card
        if card is None:
            return

        current_dir = decompose_dir
        before = after


def read_step(
    before: Kernel,
    workdir: Path,
    stage: RunKind,
) -> tuple[Kernel, SkillCard | None]:
    """Validate the problem, ABI, and action-specific card contract."""

    label = stage.value.capitalize()
    after = load_kernel(workdir)
    if before.problem != after.problem:
        raise StructuredOutputError(f"{label} changed the problem")
    if any(
        before.problem.values(role) != after.problem.values(role) for role in ValueRole
    ):
        raise StructuredOutputError(f"{label} changed the ABI")

    skill_path = workdir / SKILL_FILE
    has_card = skill_path.exists() or skill_path.is_symlink()
    if before.fingerprint == after.fingerprint:
        if has_card:
            raise StructuredOutputError(
                f"{label} emitted {SKILL_FILE} for an unchanged kernel"
            )
        if before.name != after.name or (
            stage is RunKind.APPLY and before.validation != after.validation
        ):
            raise StructuredOutputError(
                f"{label} changed an unchanged kernel's metadata"
            )
        submission = workdir / SUBMISSION_DIRECTORY
        if submission.exists() or submission.is_symlink():
            raise StructuredOutputError(
                f"{label} terminal result contains {SUBMISSION_DIRECTORY}"
            )
        return after, None

    if stage is RunKind.APPLY:
        if has_card:
            raise StructuredOutputError(f"{label} must not emit {SKILL_FILE}")
        return after, None

    if not skill_path.is_file():
        raise StructuredOutputError(f"{label} changed a kernel without {SKILL_FILE}")
    return after, load_skill(skill_path)

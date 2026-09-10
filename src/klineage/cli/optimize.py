"""Optimize an existing kernel independently or using a SkillCard directory."""

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from klineage.action.apply import Apply
from klineage.cli.common import add_run_options, read_step
from klineage.constants import (
    KERNEL_FILE,
    MAX_APPLY_STEPS,
    MAX_RETRIES,
    SKILL_FILE,
    TIMEOUT,
    RunKind,
    StepMode,
)
from klineage.errors import StructuredOutputError
from klineage.harness.artifacts import load_kernel
from klineage.kernel import Kernel
from klineage.memory import SkillCard, load_skill
from klineage.utils import optional_directory


def optimize(
    start_kernel: str | os.PathLike[str],
    memory_dir: str | os.PathLike[str] | None = None,
    workdir: str | os.PathLike[str] | None = None,
    *,
    max_apply_step: int = MAX_APPLY_STEPS,
    enable_verifier: bool = True,
    timeout: int = TIMEOUT,
    max_retries: int = MAX_RETRIES,
) -> Kernel:
    if type(max_apply_step) is not int or max_apply_step < 0:
        raise ValueError("max_apply_step must be a nonnegative integer")
    if workdir is None:
        raise ValueError("workdir is required")

    current_dir = Path(start_kernel).expanduser().resolve()
    if current_dir.is_file():
        if current_dir.name != KERNEL_FILE:
            raise ValueError(
                f"start_kernel must be a kernel directory or {KERNEL_FILE}"
            )
        current_dir = current_dir.parent
    if not (current_dir / KERNEL_FILE).is_file():
        raise FileNotFoundError(current_dir / KERNEL_FILE)

    memory = optional_directory(memory_dir)
    work = Path(workdir).expanduser().resolve()
    if memory is not None and work.is_relative_to(memory):
        raise ValueError("workdir must be outside memory_dir")

    # Freeze membership before any agent can inspect the mounted memory.
    cards = (
        tuple(load_skill(path) for path in sorted(memory.rglob(SKILL_FILE)))
        if memory is not None
        else ()
    )
    work.mkdir(parents=True, exist_ok=False)
    return apply_steps(
        current_dir,
        memory,
        cards,
        workdir=work,
        max_apply_step=max_apply_step,
        enable_verifier=enable_verifier,
        timeout=timeout,
        max_retries=max_retries,
    )


def apply_steps(
    current_dir: Path,
    memory_dir: Path | None,
    cards: Sequence[SkillCard],
    *,
    workdir: Path,
    max_apply_step: int,
    enable_verifier: bool,
    timeout: int,
    max_retries: int,
) -> Kernel:
    """Run optimization steps in an existing directory with mode-specific checks."""

    current = load_kernel(current_dir)
    mode = StepMode.BASELINE if memory_dir is None else StepMode.SKILL
    applied_skills = []
    for step in range(max_apply_step):
        apply_dir = workdir / RunKind.APPLY / str(step)
        Apply(
            current_dir,
            memory=memory_dir,
            exclude_skills=applied_skills,
            workdir=apply_dir,
            enable_verifier=enable_verifier,
            timeout=timeout,
            max_retries=max_retries,
        ).run()

        candidate, selected = read_step(current, apply_dir, RunKind.APPLY, mode)
        if selected is None and (
            mode is StepMode.SKILL or candidate.fingerprint == current.fingerprint
        ):
            return candidate
        if selected is not None:
            if selected not in cards or selected.skill_id in applied_skills:
                raise StructuredOutputError(
                    "Apply selected a modified, unknown, or excluded skill"
                )
            applied_skills.append(selected.skill_id)

        current = candidate
        current_dir = apply_dir
    return current


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start_kernel", "--start-kernel", type=Path, required=True)
    parser.add_argument(
        "--memory_dir",
        "--memory-dir",
        help="SkillCard directory; omitted or blank runs the independent baseline",
    )
    parser.add_argument(
        "--workdir", type=Path, required=True, help="Fresh run directory"
    )
    parser.add_argument("--max-apply-step", type=int, default=MAX_APPLY_STEPS)
    add_run_options(parser)
    parser.set_defaults(enable_verifier=True)
    kernel = optimize(**vars(parser.parse_args(argv)))
    print(json.dumps(kernel.to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()

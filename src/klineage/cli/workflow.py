"""Run Init, Decompose, and Apply from the command line."""

import json
import os
from collections.abc import Sequence
from pathlib import Path

from klineage.cli.common import (
    argument_parser,
    decompose_steps,
)
from klineage.cli.optimize import apply_steps
from klineage.constants import (
    MAX_APPLY_STEPS,
    MAX_DECOMPOSE_STEPS,
    MAX_RETRIES,
    MEMORY_DIR,
    SKILL_FILE,
    TIMEOUT,
    RunKind,
)
from klineage.kernel import Kernel
from klineage.memory.storage import save_skill
from klineage.utils import new_workdir


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
    workdir = Path(workdir or new_workdir(RunKind.WORKFLOW)).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=False)
    memory_dir = workdir / MEMORY_DIR
    memory_dir.mkdir()

    current_dir = workdir / RunKind.INIT
    cards = []
    for current_dir, card in decompose_steps(
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
            save_skill(card, memory_dir / current_dir.name / SKILL_FILE)
            cards.append(card)

    return apply_steps(
        current_dir,
        memory_dir,
        tuple(cards),
        workdir=workdir,
        max_apply_step=max_apply_step,
        enable_verifier=enable_verifier,
        timeout=timeout,
        max_retries=max_retries,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argument_parser(__doc__)
    parser.add_argument("--max-apply-step", type=int, default=MAX_APPLY_STEPS)
    parser.set_defaults(enable_verifier=True)
    args = parser.parse_args(argv)
    kernel = workflow(**vars(args))
    print(json.dumps(kernel.to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Extract optimization SkillCards into a memory directory."""

import json
import os
from collections.abc import Sequence
from pathlib import Path

from klineage.cli.common import (
    argument_parser,
    decompose_steps,
)
from klineage.constants import (
    MAX_DECOMPOSE_STEPS,
    MAX_RETRIES,
    SKILL_FILE,
    TIMEOUT,
    RunKind,
)
from klineage.memory.storage import save_skill
from klineage.utils import new_workdir, operation_id


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

    workdir = Path(workdir or new_workdir(RunKind.INIT_MEMORY)).expanduser().resolve()
    memory_dir = Path(memory_dir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=False)
    memory_dir.mkdir(parents=True, exist_ok=True)

    skill_paths = []
    for _, card in decompose_steps(
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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argument_parser(__doc__)
    parser.add_argument("--memory-dir", type=Path, required=True)
    parser.set_defaults(enable_verifier=False)
    args = parser.parse_args(argv)
    paths = init_memory(**vars(args))
    print(json.dumps([str(path) for path in paths], ensure_ascii=False))


if __name__ == "__main__":
    main()

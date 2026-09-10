"""Remove one mechanism and record its forward SkillCard."""

from __future__ import annotations

from pathlib import Path

from klineage.action.action import Action
from klineage.constants import MAX_RETRIES, TIMEOUT, RunKind
from klineage.kernel import Kernel
from klineage.prompts import render_prompt
from klineage.utils import new_workdir


def decompose(input_kernel: Kernel | Path):
    action = Decompose(input_kernel)
    action.run()


class Decompose(Action):
    verify_prompt = render_prompt(f"{RunKind.VERIFY}_{RunKind.DECOMPOSE}")

    def __init__(
        self,
        input_kernel: Kernel | Path,
        *,
        workdir: Path | None = None,
        enable_verifier: bool = True,
        max_retries: int = MAX_RETRIES,
        timeout: int = TIMEOUT,
    ):
        workdir = workdir or new_workdir(RunKind.DECOMPOSE)
        prompt = render_prompt(
            RunKind.DECOMPOSE,
            input_kernel=(
                input_kernel.to_dict()
                if isinstance(input_kernel, Kernel)
                else str(Path(input_kernel).expanduser().resolve())
            ),
            enable_verifier=enable_verifier,
        )
        super().__init__(
            prompt,
            workdir,
            enable_verifier,
            timeout,
            max_retries=max_retries,
        )


__all__ = ["Decompose", "decompose"]

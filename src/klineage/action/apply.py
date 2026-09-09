"""Apply one SkillCard to a kernel."""

from __future__ import annotations

import os
from pathlib import Path

from klineage._utils import new_workdir
from klineage.action.action import MAX_RETRIES, TIMEOUT, Action
from klineage.kernel import Kernel
from klineage.memory.skillcard import SkillCard
from klineage.prompts import render_prompt


def apply(
    current_kernel: Kernel | str | os.PathLike[str],
    skill: SkillCard | Path,
):
    action = Apply(current_kernel, skill)
    action.run()


class Apply(Action):
    verify_prompt = render_prompt("verify_apply")

    def __init__(
        self,
        current_kernel: Kernel | str | os.PathLike[str],
        skill: SkillCard | Path,
        *,
        workdir: Path | None = None,
        enable_verifier: bool = True,
        max_retries: int = MAX_RETRIES,
        timeout: int = TIMEOUT,
    ):
        if not isinstance(skill, (SkillCard, Path)):
            raise TypeError("skill must be a SkillCard or SKILL.md path")

        workdir = workdir or new_workdir("apply")
        prompt = render_prompt(
            "apply",
            current_kernel=(
                current_kernel.to_dict()
                if isinstance(current_kernel, Kernel)
                else str(Path(current_kernel).expanduser().resolve())
            ),
            skill=(
                skill.to_dict()
                if isinstance(skill, SkillCard)
                else str(Path(skill).expanduser().resolve())
            ),
        )
        super().__init__(
            prompt,
            workdir,
            enable_verifier,
            timeout,
            max_retries=max_retries,
        )


__all__ = ["Apply", "apply"]

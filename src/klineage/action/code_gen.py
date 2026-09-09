"""Generate a kernel from selected skills."""

from __future__ import annotations

from pathlib import Path

from klineage._utils import new_workdir
from klineage.action.action import MAX_RETRIES, TIMEOUT, Action
from klineage.backend import get_backend
from klineage.kernel import Kernel
from klineage.memory.skillcard import SkillCard
from klineage.prompts import render_prompt


def code_gen(
    current_kernel: Kernel,
    skill: SkillCard,
):
    """Materialize one SkillCard in a new runner."""

    action = CodeGen(current_kernel, skill)
    action.run()


class CodeGen(Action):
    verify_prompt = render_prompt("verify_code_gen")

    def __init__(
        self,
        current_kernel: Kernel,
        skill: SkillCard,
        workdir: Path | None = None,
        enable_verifier: bool = True,
        max_retries: int = MAX_RETRIES,
        timeout: int = TIMEOUT,
    ):
        language = current_kernel.problem.language
        if language != "python":
            get_backend(language, current_kernel.problem.platform)
        workdir = Path(workdir or new_workdir("code-gen")).expanduser().resolve()
        super().__init__(
            render_prompt(
                "code_gen",
                current_kernel=current_kernel.to_dict(),
                skill=skill.to_dict(),
            ),
            workdir,
            enable_verifier,
            timeout,
            max_retries=max_retries,
        )


__all__ = ["CodeGen", "code_gen"]

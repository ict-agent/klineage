"""Select one eligible skill using the current kernel's measured profile."""

from collections.abc import Sequence
from pathlib import Path

from klineage._utils import new_workdir
from klineage.action.action import MAX_RETRIES, TIMEOUT, Action
from klineage.kernel import Kernel
from klineage.memory.skillcard import SkillCard
from klineage.prompts import render_prompt


def retrieve(
    current_kernel: Kernel | Path,
    skills: Sequence[SkillCard | Path],
    *,
    exclude_skills: Sequence[str] = (),
):
    action = Retrieve(
        current_kernel,
        skills,
        exclude_skills=exclude_skills,
    )
    action.run()


class Retrieve(Action):
    verify_prompt = render_prompt("verify_retrieve")

    def __init__(
        self,
        current_kernel: Kernel | Path,
        skills: Sequence[SkillCard | Path],
        *,
        exclude_skills: Sequence[str] = (),
        workdir: Path | None = None,
        enable_verifier: bool = True,
        max_retries: int = MAX_RETRIES,
        timeout: int = TIMEOUT,
    ):
        workdir = workdir or new_workdir("retrieve")
        prompt = render_prompt(
            "retrieve",
            current_kernel=(
                current_kernel.to_dict()
                if isinstance(current_kernel, Kernel)
                else str(Path(current_kernel).expanduser().resolve())
            ),
            skills=[
                item.to_dict()
                if isinstance(item, SkillCard)
                else str(item.expanduser().resolve())
                for item in skills
            ],
            exclude_skills=list(exclude_skills),
        )
        super().__init__(
            prompt,
            workdir,
            enable_verifier,
            timeout,
            max_retries=max_retries,
        )


__all__ = ["Retrieve", "retrieve"]

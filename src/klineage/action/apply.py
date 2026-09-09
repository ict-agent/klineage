"""Select and apply one SkillCard to a kernel."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from klineage._utils import new_workdir, string_tuple
from klineage.action.action import MAX_RETRIES, TIMEOUT, Action
from klineage.kernel import Kernel
from klineage.memory.skillcard import SkillCard
from klineage.prompts import render_prompt


def apply(
    current_kernel: Kernel | str | os.PathLike[str],
    skill: SkillCard | Path | None = None,
    *,
    memory: Sequence[SkillCard | Path] | Path = (),
    exclude_skills: Sequence[str] = (),
):
    action = Apply(current_kernel, skill, memory=memory, exclude_skills=exclude_skills)
    action.run()


class Apply(Action):
    verify_prompt = render_prompt("verify_apply")

    def __init__(
        self,
        current_kernel: Kernel | str | os.PathLike[str],
        skill: SkillCard | Path | None = None,
        *,
        memory: Sequence[SkillCard | Path] | Path = (),
        exclude_skills: Sequence[str] = (),
        workdir: Path | None = None,
        enable_verifier: bool = True,
        max_retries: int = MAX_RETRIES,
        timeout: int = TIMEOUT,
    ):
        if isinstance(exclude_skills, (str, bytes)) or not isinstance(
            exclude_skills, Sequence
        ):
            raise TypeError("exclude_skills must be a sequence of strings")

        workdir = workdir or new_workdir("apply")
        prompt = render_prompt(
            "apply",
            current_kernel=(
                current_kernel.to_dict()
                if isinstance(current_kernel, Kernel)
                else str(Path(current_kernel).expanduser().resolve())
            ),
            skill=_skill_input(skill) if skill is not None else None,
            memory=_memory_input(memory),
            exclude_skills=list(string_tuple(exclude_skills, "exclude_skills")),
        )
        super().__init__(
            prompt,
            workdir,
            enable_verifier,
            timeout,
            max_retries=max_retries,
        )


def _skill_input(skill: SkillCard | Path) -> dict | str:
    if isinstance(skill, SkillCard):
        return skill.to_dict()
    if not isinstance(skill, Path):
        raise TypeError("skill must be a SkillCard or SKILL.md path")
    return str(skill.expanduser().resolve())


def _memory_input(memory: Sequence[SkillCard | Path] | Path) -> list | str:
    if isinstance(memory, Path):
        return str(memory.expanduser().resolve())
    if isinstance(memory, (str, bytes)) or not isinstance(memory, Sequence):
        raise TypeError("memory must be a directory or sequence of skills")
    return [_skill_input(skill) for skill in memory]


__all__ = ["Apply", "apply"]

"""Run a kernel optimization round, optionally guided by skill memory."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from klineage.action.action import Action
from klineage.artifact.kernel import Kernel
from klineage.constants import MAX_RETRIES, MEMORY_DIRECTORY, TIMEOUT, RunKind
from klineage.prompts import render_prompt
from klineage.utils import new_workdir, optional_directory, string_tuple


def apply(
    current_kernel: Kernel | str | os.PathLike[str],
    *,
    memory: str | os.PathLike[str] | None = None,
    exclude_skills: Sequence[str] = (),
):
    action = Apply(current_kernel, memory=memory, exclude_skills=exclude_skills)
    action.run()


class Apply(Action):
    def __init__(
        self,
        current_kernel: Kernel | str | os.PathLike[str],
        *,
        memory: str | os.PathLike[str] | None = None,
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

        self.memory = optional_directory(memory)
        workdir = Path(workdir or new_workdir(RunKind.APPLY)).expanduser().resolve()
        mounted_memory = str(workdir / MEMORY_DIRECTORY) if self.memory else None
        self.verify_prompt = render_prompt(
            f"{RunKind.VERIFY}_{RunKind.APPLY}", memory=mounted_memory
        )
        prompt = render_prompt(
            RunKind.APPLY,
            current_kernel=(
                current_kernel.to_dict()
                if isinstance(current_kernel, Kernel)
                else str(Path(current_kernel).expanduser().resolve())
            ),
            memory=mounted_memory,
            exclude_skills=list(string_tuple(exclude_skills, "exclude_skills")),
        )
        super().__init__(
            prompt,
            workdir,
            enable_verifier,
            timeout,
            max_retries=max_retries,
        )

    def run(self):
        self.runner.mount_memory(self.memory)
        super().run()


__all__ = ["Apply", "apply"]

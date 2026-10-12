"""Run a kernel optimization round, optionally guided by skill memory."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from klineage.action.action import Action, task_input
from klineage.artifact import Artifact, ArtifactKind
from klineage.constants import MAX_RETRIES, MEMORY_DIRECTORY, TIMEOUT, RunKind
from klineage.prompts import render_prompt
from klineage.utils import new_workdir, optional_directory, string_tuple


class Apply(Action):
    """Optimize a kernel, optionally guided by read-only skill memory.

    For an input with sources, the prompt requests the best validated improvement
    against that fixed input, or an unchanged kernel if none is found. For a
    source-free input, it requests an implementation checked against the problem
    oracle. Collection checks the latest client-saved version and its service result;
    it does not select an earlier accepted version automatically or independently
    enforce the prompt's speed requirement. The output contains no skill card.

    Args:
        current_kernel: Kernel or action artifact, serialized artifact path,
            kernel directory, or previous action directory.
        memory: Optional skill directory. Empty or omitted memory permits
            optimization from the sources alone.
        exclude_skills: Skill identifiers omitted from mounted memory.
        workdir: Output workspace; a new path is generated when omitted.
        max_retries: Additional attempts after generation or handoff failures.
        timeout: Seconds allowed for each agent attempt.
        evaluation (klineage.server.Client | None): Evaluation client; defaults
            to environment configuration.
        agent (klineage.harness.AgentRuntime | None): Optional sandbox runtime
            and mounts.
    """

    artifact_type = ArtifactKind.APPLY

    def __init__(
        self,
        current_kernel: Artifact | str | os.PathLike[str],
        *,
        memory: str | os.PathLike[str] | None = None,
        exclude_skills: Sequence[str] = (),
        workdir: Path | None = None,
        max_retries: int = MAX_RETRIES,
        timeout: int = TIMEOUT,
        evaluation=None,
        agent=None,
    ):
        if isinstance(exclude_skills, (str, bytes)) or not isinstance(
            exclude_skills, Sequence
        ):
            raise TypeError("exclude_skills must be a sequence of strings")

        self.input_kernel = current_kernel
        self.memory = optional_directory(memory)
        self.exclude_skills = string_tuple(exclude_skills, "exclude_skills")
        workdir = Path(workdir or new_workdir(RunKind.APPLY)).expanduser().resolve()
        mounted_memory = str(workdir / MEMORY_DIRECTORY) if self.memory else None
        prompt = render_prompt(
            RunKind.APPLY,
            current_kernel=task_input(current_kernel),
            memory=mounted_memory,
            exclude_skills=self.exclude_skills,
        )
        super().__init__(
            prompt,
            workdir,
            timeout,
            max_retries=max_retries,
            evaluation=evaluation,
            agent=agent,
        )

    def run(self) -> Artifact:
        """Mount selected skill memory and return an ``action.apply`` artifact."""
        self.runner.mount_memory(self.memory, exclude_skills=self.exclude_skills)
        return super().run()


__all__ = ["Apply"]

"""Remove one mechanism and record its forward SkillCard."""

from __future__ import annotations

from pathlib import Path

from klineage.action.action import Action, task_input
from klineage.artifact import Artifact, ArtifactKind
from klineage.constants import MAX_RETRIES, TIMEOUT, RunKind
from klineage.prompts import render_prompt
from klineage.utils import new_workdir


def decompose(input_kernel: Artifact | Path, *, agent=None) -> Artifact:
    """Run Decompose with default workspace, retry, and evaluation settings.

    Args:
        input_kernel: Kernel or action artifact, serialized artifact path,
            kernel directory, or previous action directory.
        agent (klineage.harness.AgentRuntime | None): Optional sandbox runtime.

    Returns:
        An ``action.decompose`` artifact with a kernel, available records, and
        a skill card when the implementation changed.
    """
    action = Decompose(input_kernel, agent=agent)
    return action.run()


class Decompose(Action):
    """Request one deoptimization and a recipe for restoring it.

    A changed kernel requires a ``skillcard`` part. An unchanged terminal result
    omits the card; evaluation still saves its complete Kernel automatically.
    Collection checks the latest saved version, preserved problem and service result.
    Atomicity and recipe quality are agent audit instructions, not structural
    guarantees. Deoptimization may make the kernel slower.

    Args:
        input_kernel: Kernel or action artifact, serialized artifact path,
            kernel directory, or previous action directory.
        workdir: Output workspace; a new path is generated when omitted.
        max_retries: Additional attempts after generation or handoff failures.
        timeout: Seconds allowed for each agent attempt.
        evaluation (klineage.server.Client | None): Evaluation client; defaults
            to environment configuration.
        agent (klineage.harness.AgentRuntime | None): Optional sandbox runtime
            and mounts.
    """

    artifact_type = ArtifactKind.DECOMPOSE

    def __init__(
        self,
        input_kernel: Artifact | Path,
        *,
        workdir: Path | None = None,
        max_retries: int = MAX_RETRIES,
        timeout: int = TIMEOUT,
        evaluation=None,
        agent=None,
    ):
        self.input_kernel = input_kernel
        workdir = workdir or new_workdir(RunKind.DECOMPOSE)
        prompt = render_prompt(
            RunKind.DECOMPOSE,
            input_kernel=task_input(input_kernel),
        )
        super().__init__(
            prompt,
            workdir,
            timeout,
            max_retries=max_retries,
            evaluation=evaluation,
            agent=agent,
        )


__all__ = ["Decompose", "decompose"]

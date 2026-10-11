"""Initialize an expert kernel from a problem and repository."""

from __future__ import annotations

import os
from pathlib import Path

from klineage.artifact import Kernel
from klineage.action.action import Action
from klineage.artifact import (
    Artifact,
    ArtifactKind,
    Problem,
    load_problem,
    stage_repository,
)
from klineage.artifact.problem import TARGET_FILE, read_problem
from klineage.constants import MAX_RETRIES, TIMEOUT, WORK_DIRECTORY, RunKind
from klineage.errors import StructuredOutputError
from klineage.prompts import render_prompt
from klineage.utils import new_workdir


def init(
    problem: Artifact | str | os.PathLike[str],
    repo: str | os.PathLike[str],
    expert_kernel: str | os.PathLike[str],
    *,
    agent=None,
) -> Artifact:
    """Run Init with default workspace, retry, and evaluation settings.

    Args:
        problem: Problem artifact, directory, or serialized artifact JSON path.
        repo: Local repository directory or Git URL.
        expert_kernel: Expert source path, relative to the staged repository
            or absolute on the controller.
        agent (klineage.harness.AgentRuntime | None): Optional sandbox runtime.

    Returns:
        An ``action.init`` artifact containing the evaluated kernel and records.
    """
    action = Init(problem, repo, expert_kernel, agent=agent)
    return action.run()


class Init(Action):
    """Adapt an expert implementation to a registered problem.

    Stage the repository as read-only agent input and request a kernel bundle
    with the problem's ABI. The returned ``action.init`` artifact contains the
    latest client-saved ``kernel`` and available service records, without a skill card.

    Args:
        problem: Problem artifact, directory, or serialized artifact JSON path. The
            evaluation service must already have the same problem registered.
        repo: Local repository directory or Git URL to stage in the workspace.
        expert_kernel: Expert source path relative to that repository, or an
            absolute controller path exposed read-only to the agent.
        workdir: Output workspace; a new path is generated when omitted.
        max_retries: Additional attempts after generation or handoff failures.
        timeout: Seconds allowed for each agent attempt.
        evaluation (klineage.server.Client | None): Evaluation client; defaults
            to environment configuration.
        agent (klineage.harness.AgentRuntime | None): Optional sandbox runtime
            and mounts.
    """

    artifact_type = ArtifactKind.INIT

    def __init__(
        self,
        problem: Artifact | str | os.PathLike[str],
        repo: str | os.PathLike[str],
        expert_kernel: str | os.PathLike[str],
        *,
        workdir: Path | None = None,
        max_retries: int = MAX_RETRIES,
        timeout: int = TIMEOUT,
        evaluation=None,
        agent=None,
    ):

        workdir = Path(workdir or new_workdir(RunKind.INIT)).expanduser().resolve()
        if isinstance(problem, Artifact):
            if problem.kind != ArtifactKind.PROBLEM:
                raise ValueError("Init requires a problem Artifact")
            problem_input = problem
        else:
            problem_input = str(Path(problem).expanduser().resolve(strict=True))
        self.repository = repo
        self.repository_path = workdir / WORK_DIRECTORY / "repository"
        self.problem_input = problem if isinstance(problem, Artifact) else Path(problem_input)
        self.expert_kernel = Path(expert_kernel)
        prompt = render_prompt(
            RunKind.INIT,
            problem=problem_input,
            repository=str(self.repository_path),
            expert_kernel=str(expert_kernel),
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
        """Stage the repository, expose inputs read-only, and run the action.

        Returns:
            An ``action.init`` artifact from the latest saved version and service result.
        """
        from .action import input_paths

        if not self.repository_path.exists():
            stage_repository(self.repository, self.repository_path)
        paths = [self.repository_path]
        if isinstance(self.problem_input, Artifact):
            paths.extend(input_paths(self.problem_input))
        else:
            paths.append(Path(self.problem_input).resolve(strict=True))
        if self.expert_kernel.is_absolute():
            paths.append(self.expert_kernel.resolve(strict=True))
        self.read_only = tuple(paths)
        result = super().run()
        actual = load_problem(result.part("kernel", Kernel))
        if (
            isinstance(self.problem_input, Path)
            and self.problem_input.is_dir()
            and not (self.problem_input / TARGET_FILE).exists()
        ):
            definition, workloads = read_problem(self.problem_input)
            expected = Problem(
                definition["name"], definition, workloads,
                language=actual.language, platform=actual.platform,
            )
        else:
            expected = load_problem(self.problem_input)
        if actual != expected:
            raise StructuredOutputError("init changed the problem")
        return result


__all__ = ["Init", "init"]

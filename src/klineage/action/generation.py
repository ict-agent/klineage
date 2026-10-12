"""Generate a measured kernel from a Problem."""

from pathlib import Path

from klineage.action.action import Action, input_paths
from klineage.artifact import Artifact, ArtifactKind, Kernel, Problem, load_problem
from klineage.constants import MAX_RETRIES, TIMEOUT, RunKind
from klineage.errors import StructuredOutputError
from klineage.prompts import render_prompt
from klineage.utils import new_workdir


class Generation(Action):
    """Create a kernel for every workload in a registered Problem.

    The agent writes candidates and evaluates them through the configured service.
    The returned ``action.generation`` Artifact contains the final measured Kernel
    and available runner records. Its Problem must match the supplied input.

    Args:
        problem: Problem artifact with at least one workload. The evaluation
            service must have the same problem registered.
        workdir: Output directory; a new path is generated when omitted.
        max_retries: Additional attempts after generation or handoff failures.
        timeout: Seconds allowed for each agent attempt.
        evaluation (klineage.server.Client | None): Evaluation client; defaults
            to environment configuration.
        agent (klineage.harness.AgentRuntime | None): Optional sandbox runtime.
        runner (klineage.harness.AgentRunner | None): Optional configured runner.
    """

    artifact_type = ArtifactKind.GENERATION

    def __init__(
        self,
        problem: Problem,
        *,
        workdir: Path | None = None,
        max_retries: int = MAX_RETRIES,
        timeout: int = TIMEOUT,
        evaluation=None,
        agent=None,
        runner=None,
    ):
        if not isinstance(problem, Problem):
            raise TypeError("Generation requires a Problem artifact")
        if not problem.workloads:
            raise ValueError("generation requires at least one workload for evaluation")
        self.problem = problem
        workdir = Path(workdir or new_workdir(RunKind.GENERATION)).expanduser().resolve()
        prompt = render_prompt(RunKind.GENERATION, problem=problem)
        super().__init__(
            prompt, workdir, timeout, max_retries=max_retries,
            evaluation=evaluation, agent=agent, runner=runner,
        )

    def run(self) -> Artifact:
        """Return the measured kernel and records, preserving the supplied Problem."""
        self.read_only = input_paths(self.problem)
        result = super().run()
        if load_problem(result.part("kernel", Kernel)) != self.problem:
            raise StructuredOutputError("generation changed the problem")
        return result


__all__ = ["Generation"]

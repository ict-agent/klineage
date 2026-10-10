"""Extract optimization recipes and return a memory Artifact."""

import argparse
from collections.abc import Iterator, Sequence
from pathlib import Path

from klineage.action import Decompose, Init
from klineage.artifact import Artifact, ArtifactKind, Node, nested
from klineage.cli.common import action_arguments, add_run_options
from klineage.constants import (
    MAX_DECOMPOSE_STEPS,
    MAX_RETRIES,
    SKILL_FILE,
    TIMEOUT,
    RunKind,
)
from klineage.harness import AgentRuntime
from klineage.memory import SkillCard, save_skill
from klineage.serialization import to_json
from klineage.server import Client
from klineage.utils import new_workdir, operation_id


def init_memory(
    problem,
    repo,
    expert_kernel,
    max_decompose_step: int = MAX_DECOMPOSE_STEPS,
    *,
    workdir: Path | None = None,
    timeout: int = TIMEOUT,
    max_retries: int = MAX_RETRIES,
    memory_dir,
    evaluation: Client | None = None,
    agent: AgentRuntime | None = None,
) -> Artifact:
    if type(max_decompose_step) is not int or max_decompose_step <= 0:
        raise ValueError("max_decompose_step must be a positive integer")
    workdir = Path(workdir or new_workdir(RunKind.INIT_MEMORY)).expanduser().resolve()
    memory_dir = Path(memory_dir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=False)
    memory_dir.mkdir(parents=True, exist_ok=True)
    children = []
    for result in decompose_steps(
        problem,
        repo,
        expert_kernel,
        max_decompose_step,
        workdir=workdir,
        timeout=timeout,
        max_retries=max_retries,
        evaluation=evaluation,
        agent=agent,
    ):
        if not any(node.name == "skillcard" for node in result.tree.children):
            continue
        card = result.part("skillcard", SkillCard)
        name = operation_id(card.skill_id)
        # Publish each successful removal before starting the next action.
        save_skill(card, memory_dir / name / SKILL_FILE)
        children.append(nested(name, card))
    tree = Node.directory(".", children, description="Reusable optimization skills.")
    return Artifact.from_tree(ArtifactKind.MEMORY, tree)


def decompose_steps(
    problem,
    repo,
    expert_kernel,
    max_decompose_step: int,
    *,
    workdir: Path,
    timeout: int,
    max_retries: int,
    evaluation: Client | None = None,
    agent: AgentRuntime | None = None,
) -> Iterator[Artifact]:
    current = Init(
        problem,
        repo,
        expert_kernel,
        workdir=workdir / RunKind.INIT,
        timeout=timeout,
        max_retries=max_retries,
        evaluation=evaluation,
        agent=agent,
    ).run()
    for step in range(max_decompose_step):
        current = Decompose(
            current,
            workdir=workdir / RunKind.DECOMPOSE / str(step),
            timeout=timeout,
            max_retries=max_retries,
            evaluation=evaluation,
            agent=agent,
        ).run()
        yield current
        if not any(node.name == "skillcard" for node in current.tree.children):
            return


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--expert-kernel", required=True)
    parser.add_argument("--max-decompose-step", type=int, default=MAX_DECOMPOSE_STEPS)
    parser.add_argument("--workdir", type=Path, help="Fresh run directory")
    add_run_options(parser)
    parser.add_argument("--memory-dir", type=Path, required=True)
    print(to_json(init_memory(**action_arguments(parser.parse_args(argv)))))


if __name__ == "__main__":
    main()

"""Optimize a kernel with optional skill memory; return an Artifact."""

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from klineage.action import Apply, same_implementation
from klineage.artifact import (
    Artifact,
    Kernel,
    load_artifact,
)
from klineage.artifact.kernel import capture_kernel, latest_kernel
from klineage.cli.common import action_arguments, add_run_options
from klineage.constants import (
    MAX_APPLY_STEPS,
    MAX_RETRIES,
    TIMEOUT,
    VERSIONS_DIRECTORY,
    RunKind,
)
from klineage.harness import AgentRuntime
from klineage.serialization import to_json
from klineage.server import Client
from klineage.server.client import SERVER_ENV
from klineage.utils import optional_directory


def optimize(
    start_kernel,
    memory_dir=None,
    workdir=None,
    *,
    max_apply_step: int = MAX_APPLY_STEPS,
    timeout: int = TIMEOUT,
    max_retries: int = MAX_RETRIES,
    evaluation: Client | None = None,
    agent: AgentRuntime | None = None,
) -> Artifact:
    if type(max_apply_step) is not int or max_apply_step < 0:
        raise ValueError("max_apply_step must be a nonnegative integer")
    if workdir is None:
        raise ValueError("workdir is required")
    current = start_kernel
    if not isinstance(current, Artifact):
        path = Path(current).expanduser()
        if path.is_dir() and (path / VERSIONS_DIRECTORY).exists():
            path = latest_kernel(path)
        if path.is_dir() and evaluation is None and os.environ.get(SERVER_ENV):
            evaluation = Client()
        if not path.is_dir():
            current = load_artifact(path)
        elif evaluation is not None:
            current = Kernel.from_tree(evaluation.info()["kernel_kind"], capture_kernel(path))
        else:
            current = load_artifact(path, Kernel)
    memory = optional_directory(memory_dir)
    work = Path(workdir).expanduser().resolve()
    if memory is not None and work.is_relative_to(memory):
        raise ValueError("workdir must be outside memory_dir")
    work.mkdir(parents=True, exist_ok=False)
    return apply_steps(
        current,
        memory,
        workdir=work,
        max_apply_step=max_apply_step,
        timeout=timeout,
        max_retries=max_retries,
        evaluation=evaluation,
        agent=agent,
    )


def apply_steps(
    current: Artifact,
    memory_dir: Path | None,
    *,
    workdir: Path,
    max_apply_step: int,
    timeout: int,
    max_retries: int,
    evaluation: Client | None = None,
    agent: AgentRuntime | None = None,
) -> Artifact:
    before = (
        current if isinstance(current, Kernel) else
        Kernel.from_tree(current.kind, current.tree) if current.kind.startswith("kernel.") else
        current.part("kernel", Kernel)
    )
    for step in range(max_apply_step):
        candidate = Apply(
            current,
            memory=memory_dir,
            workdir=workdir / RunKind.APPLY / str(step),
            timeout=timeout,
            max_retries=max_retries,
            evaluation=evaluation,
            agent=agent,
        ).run()
        after = candidate.part("kernel", Kernel)
        if same_implementation(before, after):
            return candidate
        current, before = candidate, after
    return current


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start_kernel", "--start-kernel", type=Path, required=True)
    parser.add_argument(
        "--memory_dir", "--memory-dir", help="Optional SkillCard directory"
    )
    parser.add_argument(
        "--workdir", type=Path, required=True, help="Fresh run directory"
    )
    parser.add_argument("--max-apply-step", type=int, default=MAX_APPLY_STEPS)
    add_run_options(parser)
    print(to_json(optimize(**action_arguments(parser.parse_args(argv)))))


if __name__ == "__main__":
    main()

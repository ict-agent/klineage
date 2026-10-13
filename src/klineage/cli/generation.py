"""Generate a kernel using local or separately hosted evaluation."""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from contextlib import ExitStack, redirect_stdout
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from klineage.action import Generation
from klineage.artifact import Kernel, load_problem
from klineage.artifact.problem import DEFINITION_FILE, TARGET_FILE, problem_reference
from klineage.backend import detect_backend, get_backend
from klineage.cli.common import action_arguments, add_run_options
from klineage.constants import MAX_RETRIES, TIMEOUT, RunKind
from klineage.harness import AgentRuntime
from klineage.serialization import to_json
from klineage.server import Client, Service
from klineage.server.client import SERVER_ENV, link_problem
from klineage.server.worker import WorkerRuntime
from klineage.utils import new_workdir


def generate(
    problem: str | Path,
    *,
    workdir: Path | None = None,
    timeout: int = TIMEOUT,
    max_retries: int = MAX_RETRIES,
    evaluation: Client | None = None,
    agent: AgentRuntime | None = None,
) -> Kernel:
    """Generate a measured kernel with a local or existing evaluation service.

    Args:
        problem: Directory containing definition.json and workloads.jsonl.
            All workloads are retained. Missing target metadata comes from the
            local backend or selected service.
        workdir: Fresh output directory; generated under experiment/ when omitted.
        timeout: Seconds per agent attempt and local evaluation request.
            An existing client retains its configured request timeout.
        max_retries: Additional attempts after a generation or handoff failure.
        evaluation: Existing service client; inferred from `KLINEAGE_SERVER` when
            set. If neither is supplied, start a local service. Existing services
            require locally accessible registered problem directories.
        agent: Agent runtime override. Defaults to local device access with a
            local service, or a device-free runtime with an existing service.

    Returns:
        The measured Kernel, with its sources and problem reference retained in
        workdir. Agent records and evaluation diagnostics remain there as well.

    Raises:
        RuntimeError: No supported local accelerator is available.
        ValueError: The problem has no workloads, differs from the registered
            problem, or run settings are invalid.
        FileExistsError: The output directory already exists.
        NotADirectoryError: The problem path is a file.
    """
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("timeout must be a positive integer")
    if type(max_retries) is not int or max_retries < 0:
        raise ValueError("max_retries must be a nonnegative integer")

    path = Path(problem).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise NotADirectoryError(f"{path}: expected a problem directory")
    if evaluation is None and os.environ.get(SERVER_ENV):
        evaluation = Client(timeout=timeout)
    if evaluation is None:
        backend = detect_backend()
        spec = load_problem(path, backend=backend)
    else:
        evaluation, spec, path = remote_problem(path, evaluation)
        backend = get_backend(spec.language, spec.platform)
    if not spec.workloads:
        raise ValueError("generation requires at least one workload for evaluation")
    runtime = AgentRuntime.local(backend) if evaluation is None else agent or AgentRuntime()
    work = Path(workdir or new_workdir(RunKind.GENERATION)).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=False)
    reference = problem_reference(path, spec)

    # Keep service records outside the agent's writable workspace.
    with ExitStack() as stack:
        if evaluation is None:
            root = Path(stack.enter_context(TemporaryDirectory(prefix="klineage-generation-")))
            factory = partial(
                WorkerRuntime,
                python=runtime.python,
                device_nodes=runtime.device_nodes,
                read_only=runtime.read_only,
                environment=runtime.environment,
            )
            service = Service(
                root / "service", backend, {spec.name: path},
                runtime_factory=factory, timeout=timeout,
            )
            server = stack.enter_context(service.start("unix://" + str(root / "evaluation.sock")))
            evaluation = Client(server.endpoint, spec.name, timeout=timeout)
            if not evaluation.info()["device"]["available"]:
                raise RuntimeError(f"{backend.kind} evaluation worker is unavailable")
        link_problem(work, evaluation.problem_id, reference)
        result = Generation(
            spec,
            workdir=work,
            timeout=timeout,
            max_retries=max_retries,
            evaluation=evaluation,
            agent=agent or runtime,
        ).run()
    return result.part("kernel", Kernel)


def remote_problem(path: Path, client: Client):
    """Match a local problem to a registered task without probing local devices."""
    info = client.info()
    target = path / TARGET_FILE
    name_file = target if target.exists() else path / DEFINITION_FILE
    name = json.loads(name_file.read_text())["name"]
    selected = client.problem_id or (name if name in info["problems"] else client.select_problem(info=info))
    problem = client.problem(selected)
    requested = (
        load_problem(path) if target.exists()
        else load_problem({"path": str(path), **problem.target()})
    )
    definition, registered = requested.definition, problem.definition
    pins = [
        {
            node.name: (node.reference.sha256, node.reference.size)
            for subtree in value.tree.children if subtree.name == "inputs"
            for node in subtree.children
        }
        for value in (requested, problem)
    ]
    if (
        requested.target() != problem.target()
        or definition != registered
        or any(tuple(definition[role]) != tuple(registered[role]) for role in ("inputs", "outputs"))
        or requested.workload_records() != problem.workload_records()
        or pins[0] != pins[1]
    ):
        raise ValueError("input directory differs from the registered problem")
    source = Path(info["problem_paths"][selected])
    if not source.is_dir():
        raise ValueError(f"registered problem directory must be accessible locally: {source}")
    return Client(client.endpoint, selected, timeout=client.timeout), problem, source


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "problem",
        type=Path,
        metavar="PROBLEM_DIR",
        help="Problem directory containing definition.json and workloads.jsonl",
    )
    parser.add_argument("--workdir", type=Path, help="Fresh output directory")
    add_run_options(parser)
    arguments = parser.parse_args(argv)
    with redirect_stdout(sys.stderr):
        result = generate(**action_arguments(arguments))
    print(to_json(result))


if __name__ == "__main__":
    main()

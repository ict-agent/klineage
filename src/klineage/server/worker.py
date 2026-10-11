"""Device workers isolated from service state and agent processes."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from klineage.artifact import Kernel, load_problem
from klineage.harness.eval import EvaluationRuntime, WorkerError
from klineage.harness.process import run_process
from klineage.harness.sandbox import (
    Sandbox,
    runtime_environment,
    runtime_paths,
)
from klineage.serialization import raw_text, restore, to_json

from .protocol import DEFAULT_TIMEOUT

PACKAGE_SOURCE = Path(__file__).resolve().parents[2]


def plugin_paths(names: Sequence[str]) -> tuple[Path, ...]:
    paths = []
    for name in names:
        spec = importlib.util.find_spec(name.split(".")[0])
        if spec is None:
            raise ModuleNotFoundError(name)
        if spec.submodule_search_locations:
            paths.extend(Path(path) for path in spec.submodule_search_locations)
        elif spec.origin and spec.origin not in ("built-in", "frozen"):
            paths.append(Path(spec.origin))
    return tuple(dict.fromkeys(paths))


def problem_paths(kernel: Kernel, workload_id: str | None = None) -> tuple[Path, ...]:
    metadata = json.loads(raw_text(kernel.find("metadata.json")))
    paths = [Path(metadata["problem"]["path"]).resolve(strict=True)]
    problem = load_problem(kernel)
    workloads = problem.resolve_inputs()
    if workload_id is not None and not any(case["uuid"] == workload_id for case in workloads):
        raise ValueError(f"unknown workload uuid: {workload_id}")
    paths.extend(
        Path(value["path"])
        for workload in workloads
        for value in workload.get("inputs", {}).values()
        if value.get("type") == "safetensors"
    )
    return tuple(dict.fromkeys(paths))


class WorkerRuntime(EvaluationRuntime):
    """Run evaluation and metadata requests in a device-enabled sandbox.

    Each process gets one writable work directory and declared read-only inputs.
    Configuration belongs to the service operator. Device visibility variables
    do not grant access; every device node must be listed explicitly.

    Args:
        timeout: Worker process deadline in seconds.
        python: Worker interpreter, independent of the service interpreter.
        environment: Overrides for allowlisted host runtime variables. Credential
            names and loader hooks are rejected.
        worker_plugins: Modules registering worker artifact or backend types.
        read_only: Additional dependency paths mounted read-only.
        device_nodes: Allowed device nodes; empty by default.
    """

    def __init__(
        self, *, timeout: float = DEFAULT_TIMEOUT,
        python: str | Path = sys.executable,
        environment: Mapping[str, str] | None = None,
        worker_plugins: Sequence[str] = (),
        read_only: Sequence[str | Path] = (),
        device_nodes: Sequence[str | Path] = (),
    ):
        python = Path(python).expanduser().absolute()
        super().__init__(
            timeout=timeout, python=python,
            environment=runtime_environment(environment), worker_plugins=worker_plugins,
        )
        plugins = plugin_paths(self.worker_plugins)
        self.read_only = (*runtime_paths(python), PACKAGE_SOURCE, *plugins, *read_only)
        self.device_nodes = tuple(device_nodes)
        search_paths = [str(path.parent) for path in plugins]
        if self.environment.get("PYTHONPATH"):
            search_paths.append(self.environment["PYTHONPATH"])
        self.environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(search_paths))

    def launch(self, command, request, timeout, environment):
        started = time.monotonic()
        if request.get("operation") not in {"evaluate", "info"}:
            raise WorkerError("server workers accept evaluation and info requests only")
        config = request["config"]
        workspace = Path(config["build_root"]).resolve(strict=True).parent
        paths = [*self.read_only, *config["include_paths"]]
        for field in ("kernel", "reference"):
            value = request.get(field)
            if value is not None:
                kernel = value if isinstance(value, Kernel) else restore(value, Kernel)
                paths.extend(problem_paths(kernel, request.get("workload_id")))
        sandbox = Sandbox(
            workspace, read_only=paths, devices=self.device_nodes,
            environment=environment,
        )
        payload = to_json(request, indent=None)
        command = sandbox.command(command)
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise WorkerError("worker deadline expired during sandbox preparation")
        return run_process(
            command, payload=payload, timeout=remaining,
            environment=sandbox.environment,
        )

"""Evaluate kernel artifacts locally or through a configured service."""

from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from klineage.artifact import Kernel, Node, Problem, load_artifact, load_problem
from klineage.artifact.kernel import capture_kernel
from klineage.artifact.problem import DEFINITION_FILE
from klineage.backend import detect_backend, get_backend
from klineage.constants import CURRENT_KERNEL_DIRECTORY, KERNEL_METADATA
from klineage.harness.logging import kernel_fields, log_dir, measurement_fields, now, record
from klineage.harness.sandbox import AgentRuntime
from klineage.serialization import raw_text
from klineage.server import Client, Evaluation, Service
from klineage.server.client import PROBLEMS_DIRECTORY, SERVER_ENV, save_result
from klineage.server.protocol import DEFAULT_TIMEOUT, deadline_seconds, submission
from klineage.server.worker import WorkerRuntime


def evaluate(
    kernel: Kernel | Path,
    *,
    reference: Kernel | Path | Evaluation | str | None = None,
    work: Path | None = None,
    timeout: float | None = None,
) -> Evaluation:
    """Measure every workload and save one complete Kernel under versions/<N>.

    ``KLINEAGE_SERVER`` selects remote evaluation; otherwise the local backend
    runs isolated workers without an HTTP listener. Remote failures never trigger
    local execution. Directory inputs are read without compiling on the caller.

    Args:
        kernel: Kernel artifact or its source directory.
        reference: Optional timing baseline: a Kernel, directory or Evaluation.
            A job ID is accepted only with a configured service.
        work: Output directory; defaults to the current directory.
        timeout: Optional evaluation budget in seconds.

    Returns:
        Evaluation containing the measured Kernel, per-workload validation and
        diagnostics. Failed measurements are returned and saved; invalid input
        and transport failures raise.
    """
    work = Path(work or Path.cwd()).expanduser().absolute()
    budget = DEFAULT_TIMEOUT if timeout is None else deadline_seconds(timeout)
    if os.environ.get(SERVER_ENV):
        client = Client(timeout=budget)
        settings = client.info()
        candidate = read_kernel(kernel, settings["kernel_kind"])
        selected = select_problem(client, settings, kernel_reference(candidate))
        baseline = reference.kernel if isinstance(reference, Evaluation) else reference
        return client.evaluate(candidate, work, reference=baseline, problem_id=selected)

    candidate = read_kernel(kernel)
    registered = kernel_reference(candidate)
    spec = load_problem(candidate)
    backend = get_backend(spec.language, spec.platform)
    runtime = AgentRuntime.local(backend)
    if isinstance(reference, str):
        raise ValueError("local reference requires a Kernel, Path or Evaluation, not a job ID")
    baseline = reference.kernel if isinstance(reference, Evaluation) else reference
    if baseline is not None:
        baseline = read_kernel(baseline)
        if load_problem(baseline) != spec:
            raise ValueError("candidate and baseline must reference the same problem")
    factory = partial(
        WorkerRuntime, python=runtime.python, device_nodes=runtime.device_nodes,
        read_only=runtime.read_only, environment=runtime.environment,
    )
    started = now()
    directory = log_dir()
    fields = {"problem_id": spec.name, "accepted": False, **kernel_fields(candidate)}
    try:
        with TemporaryDirectory(prefix="klineage-evaluation-") as temporary:
            service = Service(
                Path(temporary), backend, {spec.name: Path(registered["path"])},
                runtime_factory=factory, timeout=budget,
            )
            reply = service.evaluate({
                "problem_id": spec.name, "submission": submission(candidate),
                "reference": submission(baseline) if baseline is not None else None,
                "timeout": budget,
            })
            fields.update(job_id=reply.job_id, accepted=reply.accepted, error=reply.error,
                          **measurement_fields(reply.validation))
            return save_result(reply, work)
    except Exception as error:
        fields["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if directory is not None:
            record("evaluate", directory=directory, at=started, finished_at=now(), **fields)


def info() -> dict:
    """Return backend, device and available problem directories.

    With ``KLINEAGE_SERVER``, read the service without probing local hardware.
    Otherwise detect the local backend and inspect this workspace's problem links
    or current_kernel metadata.
    """
    if os.environ.get(SERVER_ENV):
        return Client().info()
    backend = detect_backend()
    paths = local_problems()
    return {
        "backend": str(backend.kind), "kernel_kind": backend.kernel_type.artifact_kind,
        "device": backend.info(), "problems": list(paths),
        "problem_paths": {name: str(path) for name, path in paths.items()},
        "timeout": DEFAULT_TIMEOUT,
    }


def problem(problem_id: str | Path | None = None) -> Problem:
    """Read a problem without executing its reference implementation.

    A configured service accepts its registered ID. Locally, provide a problem
    directory or the name of a workspace problems/ link. Omit the ID only when
    one problem is available; current_kernel's reference is used if no links exist.
    """
    if os.environ.get(SERVER_ENV):
        return Client().problem(problem_id)
    paths = local_problems()
    if problem_id is None:
        if len(paths) != 1:
            raise ValueError("select one problem directory or workspace problem name")
        path = next(iter(paths.values()))
    else:
        path = paths.get(str(problem_id), Path(problem_id).expanduser())
    return load_problem(path)


def result(best: Evaluation | str, *, work: Path | None = None) -> Evaluation:
    """Append an earlier measured result as the latest version without rerunning it.

    A configured service retrieves the authoritative record by ID. Without a
    service, pass the existing Evaluation object; local job IDs are not a store.
    """
    work = Path(work or Path.cwd()).expanduser().absolute()
    if os.environ.get(SERVER_ENV):
        return Client().result(best.job_id if isinstance(best, Evaluation) else best, work=work)
    if not isinstance(best, Evaluation):
        raise ValueError("local result selection requires the existing Evaluation object")
    return save_result(best, work)


def read_kernel(value: Kernel | Path, kind: str | None = None) -> Kernel:
    if isinstance(value, Kernel):
        return value
    path = Path(value).expanduser().absolute()
    if kind is not None and path.is_dir():
        return Kernel.from_tree(kind, capture_kernel(path))
    return load_artifact(path, Kernel)


def kernel_reference(kernel: Kernel) -> dict:
    node = kernel.find(KERNEL_METADATA)
    if not isinstance(node, Node):
        raise ValueError("kernel requires exactly one metadata.json")
    reference = json.loads(raw_text(node))["problem"]
    if not isinstance(reference, dict) or set(reference) != {"path", "name", "language", "platform"}:
        raise ValueError("kernel problem requires path, name, language and platform")
    if any(not isinstance(value, str) or not value.strip() for value in reference.values()):
        raise ValueError("kernel problem fields must be nonempty text")
    if not Path(reference["path"]).is_absolute():
        raise ValueError("kernel problem path must be absolute")
    return reference


def select_problem(client: Client, settings: dict, reference: dict) -> str:
    path = Path(reference["path"]).resolve()
    matches = [name for name, value in settings.get("problem_paths", {}).items()
               if Path(value).resolve() == path]
    if not matches:
        matches = [name for name in settings["problems"]
                   if name == reference["name"] or client.problem(name).name == reference["name"]]
    if len(matches) != 1:
        raise ValueError("kernel problem is unregistered or ambiguous in the evaluation service")
    return matches[0]


def local_problems() -> dict[str, Path]:
    root = Path.cwd() / PROBLEMS_DIRECTORY
    paths = {path.name: path.resolve() for path in sorted(root.iterdir())
             if path.is_dir() and (path / DEFINITION_FILE).is_file()} if root.is_dir() else {}
    if paths:
        return paths
    metadata = Path.cwd() / CURRENT_KERNEL_DIRECTORY / KERNEL_METADATA
    if metadata.is_file():
        reference = json.loads(metadata.read_text())["problem"]
        return {reference["name"]: Path(reference["path"]).resolve()}
    return {}


__all__ = ["evaluate", "info", "problem", "result"]

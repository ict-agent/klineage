"""Device-free HTTP and Unix-socket evaluation client."""

from __future__ import annotations

import http.client
import json
import os
import socket
from dataclasses import replace
from http import HTTPStatus
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from klineage.artifact import (
    Artifact,
    Kernel,
    Problem,
    json_node,
    load_problem,
)
from klineage.artifact.kernel import snapshot_kernel
from klineage.serialization import raw_text, restore, to_json
from klineage.tools import agent_function
from klineage.utils import DirectoryMode, safe_directory

from .protocol import (
    DEFAULT_TIMEOUT,
    REQUEST_LIMIT,
    RESPONSE_LIMIT,
    Evaluation,
    ServerError,
    deadline_seconds,
    identifier,
    source_digest,
    strict_json,
    submission,
    transport_artifact,
)

SERVER_ENV = "KLINEAGE_SERVER"
PROBLEMS_DIRECTORY = Path("problems")
RESPONSE_GRACE_SECONDS = 10


class UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self.socket_path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_path)


class Client:
    """Submit source artifacts for server-side build, correctness and timing.

    This client needs no local accelerator or compiler. Returned kernels contain
    measured sources and validation; their callables are not built locally.

    Args:
        endpoint (str | None): HTTP(S) URL or `unix:///absolute/socket/path`; defaults to
            `KLINEAGE_SERVER`.
        problem_id (str | None): Default registered task. An unbound client selects
            the sole registered problem automatically; multiple tasks require an ID.
        timeout (float): Default request budget in seconds. Network reads allow an
            additional response grace period.
    """

    def __init__(
        self, endpoint=None, problem_id=None, timeout=DEFAULT_TIMEOUT
    ):
        self.endpoint = endpoint or os.environ.get(SERVER_ENV)
        self.problem_id = identifier(problem_id) if problem_id is not None else None
        self.timeout = deadline_seconds(timeout)
        if not self.endpoint:
            raise ValueError("evaluation server endpoint is required")
        if not isinstance(self.endpoint, str) or any(char in self.endpoint for char in "\0\r\n\t"):
            raise ValueError("server endpoint must be a URL without control characters")
        target = urlsplit(self.endpoint)
        if target.scheme not in ("http", "https", "unix"):
            raise ValueError("server endpoint must use http, https or unix")
        if target.username or target.password or target.query or target.fragment:
            raise ValueError(
                "server endpoint cannot contain credentials, query or fragment"
            )
        if target.scheme == "unix" and (
            target.netloc or not target.path.startswith("/")
            or unquote(target.path).rstrip("/") == "" or "\0" in unquote(target.path)
        ):
            raise ValueError("Unix endpoint needs an absolute socket path")
        if target.scheme in ("http", "https"):
            if not target.hostname or any(char.isspace() for char in target.hostname):
                raise ValueError("HTTP endpoint needs a hostname")
            if target.port is not None and not 1 <= target.port <= 65535:
                raise ValueError("HTTP endpoint needs a valid port")

    def request(self, method: str, path: str, payload=None, *, timeout=None):
        budget = self.timeout if timeout is None else deadline_seconds(timeout)
        target = urlsplit(self.endpoint)
        if target.scheme == "unix":
            connection = UnixConnection(
                unquote(target.path), budget + RESPONSE_GRACE_SECONDS
            )
            prefix = ""
        else:
            connection_type = (
                http.client.HTTPSConnection
                if target.scheme == "https"
                else http.client.HTTPConnection
            )
            connection = connection_type(
                target.hostname, target.port, timeout=budget + RESPONSE_GRACE_SECONDS
            )
            prefix = target.path.rstrip("/")
        body = (
            None
            if payload is None
            else to_json(payload, indent=None).encode()
        )
        if body is not None and len(body) > REQUEST_LIMIT:
            raise ValueError("evaluation request exceeds its byte limit")
        try:
            connection.request(
                method,
                prefix + path,
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            data = response.read(RESPONSE_LIMIT + 1)
            if len(data) > RESPONSE_LIMIT:
                raise ServerError("evaluation response exceeds its byte limit")
            value = strict_json(data)
            if not isinstance(value, dict):
                raise ServerError("evaluation response must be an object")
            if response.status != HTTPStatus.OK:
                raise ServerError(
                    f"HTTP {response.status}: {value.get('error', value)}"
                )
            return value
        except (OSError, http.client.HTTPException, ValueError) as error:
            raise ServerError(f"evaluation service request failed: {error}") from error
        finally:
            connection.close()

    @agent_function
    def info(self):
        """Read the backend, cached worker device metadata and registered tasks."""
        return self.request("GET", "/info")

    def health(self):
        return self.request("GET", "/health")

    def select_problem(self, problem_id=None, info=None) -> str:
        selected = problem_id if problem_id is not None else self.problem_id
        if selected is not None:
            return identifier(selected)
        problems = (info if info is not None else self.info())["problems"]
        if len(problems) != 1:
            raise ValueError("problem_id is required when the service does not expose exactly one problem")
        return identifier(problems[0])

    @agent_function
    def problem(self, problem_id=None) -> Problem:
        """Read a registered problem without downloading fixed input binaries.


        Args:
            problem_id (str | None): Task identifier, defaulting to this client's
                default task or the sole registered problem.

        Returns:
            Oracle and workload description; fixed input FileRefs remain
            server-owned and need not resolve on the client.
            ``problem.definition`` is a dict and ``problem.workloads`` is a tuple
            of dicts. Read each case with ``case["uuid"]``, ``case["axes"]`` and
            ``case["inputs"]``.
        """
        selected = self.select_problem(problem_id)
        payload = self.request("GET", "/problems/" + quote(selected, safe=""))
        data = transport_artifact(payload["problem"])
        return restore(data, Problem)

    def local_result(self, payload: dict, work: Path, *, problem_id=None) -> Evaluation:
        result = restore(payload, Evaluation)
        expected = self.problem_id if problem_id is None else problem_id
        if expected is not None and result.problem_id != expected:
            raise ValueError("result belongs to another problem")
        return save_result(result, work)

    @agent_function
    def evaluate(
        self, kernel: Artifact | Path, work: Path, *, reference=None, timeout=None,
        problem_id: str | None = None,
    ) -> Evaluation:
        """Submit text sources for remote build, correctness and timing.

        No local evaluation fallback is attempted. The returned Kernel is
        automatically saved under `work/versions/<N>/`; the input is not updated.

        Args:
            kernel: Kernel artifact or source directory, read without compiling.
            work: Destination for snapshots with measurements and diagnostics. A
                `problems/<problem_id>` symlink points to the original problem.
            reference (Kernel | Path | str | Evaluation | None): Timing baseline:
                a Kernel, directory, accepted job ID or prior `Evaluation` for
                the same problem.
            timeout (float | None): Request budget in seconds; defaults to the
                client's timeout.
            problem_id: Registered problem to evaluate, defaulting to the bound
                task or the sole registered problem.

        Returns:
            Measurement result, including failed gates and worker diagnostics.

        Raises:
            ServerError: The request fails before a valid result is available.
            ValueError: No fixed problem is selected or submission data is invalid.
        """
        budget = self.timeout if timeout is None else deadline_seconds(timeout)
        info = self.info()
        selected = self.select_problem(problem_id, info)
        kind = info["kernel_kind"]
        candidate = submission(kernel, kind)
        if isinstance(reference, Evaluation):
            reference = reference.job_id
        elif reference is not None and not isinstance(reference, str):
            reference = submission(reference, kind)
        payload = self.request(
            "POST",
            "/evaluate",
            {
                "problem_id": selected,
                "submission": candidate,
                "reference": reference,
                "timeout": budget,
            },
            timeout=budget,
        )
        return self.local_result(payload, work, problem_id=selected)

    @agent_function
    def result(self, job_id: str, *, work: Path | None = None) -> Evaluation:
        """Fetch a recorded result and restore its local handoff files.

        Args:
            job_id: Server-issued measurement identifier.
            work: Destination for the problem link and measured Kernel snapshots;
                defaults to the current directory.

        Returns:
            Recorded evaluation with its kernel linked to the local problem copy.
            A new `versions/<N>/` snapshot makes this the latest local result.
        """
        payload = self.request("GET", "/jobs/" + quote(job_id, safe=""))
        work = work or Path.cwd()
        return self.local_result(payload, work)

    def verify(self, kernel: Kernel, job_id: str | None = None) -> Evaluation:
        """Match a handed-off kernel to the server's recorded measurement.

        Check the problem directory reference before opening it, then compare
        the problem, submitted sources, compile flags and validation. This checks
        record consistency; inspect `accepted` on the result for gate acceptance.

        Args:
            kernel: Handed-off artifact referencing its original problem directory.
            job_id: Measurement record to compare against; defaults to the job ID
                in the Kernel's metadata.

        Returns:
            Server evaluation record without rewriting local files.

        Raises:
            ValueError: The handoff differs from the registered problem or record.
            ServerError: The server request fails.
        """
        metadata = json.loads(raw_text(kernel.find("metadata.json")))
        measurement = metadata.get("evaluation", {})
        if job_id is None:
            job_id = measurement.get("job_id")
        if not job_id:
            raise ValueError("kernel metadata has no evaluation job ID")
        if measurement and measurement.get("job_id") != job_id:
            raise ValueError("kernel evaluation job ID differs from the selected job")

        result = restore(self.request("GET", "/jobs/" + quote(job_id, safe="")), Evaluation)
        if self.problem_id and self.problem_id != result.problem_id:
            raise ValueError("measurement belongs to another problem")
        problem = self.problem(result.problem_id)
        reference = json.loads(result.kernel.find("metadata.json").content)["problem"]
        if metadata["problem"] != reference:
            raise ValueError("handoff problem reference differs from the registered directory")
        if Path(reference["path"]).exists() and load_problem(kernel) != problem:
            raise ValueError("handoff changed the registered problem")
        if source_digest(result.problem_id, submission(kernel)) != result.digest:
            raise ValueError("handoff sources differ from measured sources")
        if kernel.validation != result.validation:
            raise ValueError("handoff validation differs from measured result")
        if measurement and measurement != {
            "job_id": result.job_id, "error": result.error, "diagnostics": result.diagnostics,
        }:
            raise ValueError("kernel evaluation metadata differs from the server result")
        current = type(result.kernel)(
            result.kernel.tree.with_child(json_node("metadata.json", metadata))
        )
        return replace(result, kernel=current)


def save_result(result: Evaluation, work: Path) -> Evaluation:
    """Attach diagnostics and persist one measured Kernel with its problem link."""
    metadata = json.loads(raw_text(result.kernel.find("metadata.json")))
    link_problem(work, result.problem_id, metadata["problem"])
    metadata["evaluation"] = {
        "job_id": result.job_id, "error": result.error, "diagnostics": result.diagnostics,
    }
    kernel = type(result.kernel)(
        result.kernel.tree.with_child(json_node("metadata.json", metadata))
    )
    saved = replace(result, kernel=kernel)
    snapshot_kernel(saved.kernel, work)
    return saved


def link_problem(work: Path, problem_id: str, reference: dict) -> Path:
    """Expose the original problem directory without copying its files."""
    target = Path(reference["path"])
    if not target.is_absolute():
        raise ValueError("problem directory reference must be absolute")
    root = Path(work).expanduser().absolute() / PROBLEMS_DIRECTORY
    name = identifier(problem_id)
    with safe_directory(root, DirectoryMode.CREATE) as directory:
        try:
            os.symlink(str(target), name, dir_fd=directory)
        except FileExistsError:
            if os.readlink(name, dir_fd=directory) != str(target):
                raise ValueError("problem link differs from the registered directory") from None
    return root / name

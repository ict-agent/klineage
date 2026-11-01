"""One device's evaluation service."""

from __future__ import annotations

import hashlib
import json
import socketserver
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from klineage.artifact import (
    Kernel,
    Node,
    Problem,
    ValidationResult,
    json_node,
    load_problem,
    source_tree,
)
from klineage.artifact.problem import (
    DEFINITION_FILE,
    TARGET_FILE,
    WORKLOADS_FILE,
    problem_reference,
)
from klineage.backend import Backend, get_backend
from klineage.serialization import restore, to_json
from klineage.utils import safe_file, write_json

from .protocol import (
    DEFAULT_TIMEOUT,
    DIAGNOSTIC_LIMIT,
    REQUEST_LIMIT,
    Evaluation,
    check_submission,
    deadline_seconds,
    identifier,
    source_digest,
    strict_json,
    submission,
)

HTTP_READ_TIMEOUT = 30
PROBLEM_ENV = "KLINEAGE_PROBLEM"


def load_problems(
    paths: str | Path | Sequence[str | Path], *, backend: Backend | None = None,
) -> dict[str, Path]:
    """Load one problem directory or collections of immediate problem directories.

    Args:
        paths: Directory path or paths. Each problem contains `definition.json`
            and `workloads.jsonl`; collection files such as README are ignored.
        backend: Target for problems without stored target metadata.

    Returns:
        Original directories keyed by declared names. Empty workload files are allowed.

    Raises:
        ValueError: Names repeat or no problem directories are found.
        OSError: A directory or either required problem file is missing.
    """
    paths = (paths,) if isinstance(paths, (str, Path)) else paths
    problems = {}
    for value in paths:
        root = Path(value).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(f"expected a problem directory or collection: {root}")
        candidates = (root,) if any((root / name).exists() for name in (DEFINITION_FILE, WORKLOADS_FILE)) else sorted(
            path for path in root.iterdir() if path.is_dir()
            and any((path / name).exists() for name in (DEFINITION_FILE, WORKLOADS_FILE))
        )
        if not candidates:
            raise ValueError(f"no problem directories found: {root}")
        for path in candidates:
            problem = load_problem(path, backend=backend)
            name = identifier(problem.name)
            if name in problems:
                raise ValueError(f"duplicate problem name: {name}")
            problems[name] = path
    if not problems:
        raise ValueError("provide at least one problem directory")
    return problems


def problem_digest(problem: Problem) -> str:
    """Hash problem data and input pins, preserving the ordered tensor ABI."""
    definition = problem.definition
    for role in ("inputs", "outputs"):
        definition[role] = list(definition[role].items())
    inputs = {
        node.name: (node.reference.path, node.reference.sha256, node.reference.size)
        for subtree in problem.tree.children if subtree.name == "inputs"
        for node in subtree.children
    }
    payload = {
        "target": problem.target(), "definition": definition,
        "workloads": problem.workload_records(), "inputs": inputs,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class Service:
    """Own registered problems, serialized device access and measured records.

    Requests supply source text; the service supplies problem references, include
    paths and evaluation policy. The default runtime isolates worker filesystem
    access. Candidate and oracle execute in the same worker, so isolation is not
    proof against a candidate modifying the in-process checker.

    Args:
        root: Service-owned directory for in-memory problems, jobs and diagnostics.
        backend: Backend shared by all registered problems.
        problems: Nonempty mapping from task identifiers to trusted problem
            directories or in-memory specs.
        runtime_factory: Callable accepting `timeout=seconds` and returning a
            runtime with `info()` and `evaluate()`; defaults to `WorkerRuntime`.
        include_paths: Operator-selected compiler include directories.
        timeout: Maximum evaluation budget, including time waiting for the device.
    """

    def __init__(
        self,
        root: Path,
        backend: Backend,
        problems: Mapping[str, Problem | Path],
        *,
        runtime_factory: Callable | None = None,
        include_paths: Sequence[Path] = (),
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.backend = backend
        if runtime_factory is None:
            from .worker import WorkerRuntime

            runtime_factory = WorkerRuntime
        self.runtime_factory = runtime_factory
        self.include_paths = tuple(
            Path(path).resolve(strict=True) for path in include_paths
        )
        self.timeout = deadline_seconds(timeout)
        self.lock = threading.Lock()
        self.device_info = None
        self.problems = {}
        self.problem_paths = {}
        for problem_id, value in problems.items():
            identifier(problem_id)
            if isinstance(value, Problem):
                problem = value
                path = self.root / "probs" / problem_id
                if path.exists() and load_problem(path) != problem:
                    raise ValueError(f"{problem_id}: registered problem cannot change")
                records = "".join(
                    to_json({"definition": problem.definition["name"], "workload": case}, indent=None) + "\n"
                    for case in problem.resolve_inputs()
                )
                for name, text in (
                    (DEFINITION_FILE, to_json(problem.definition) + "\n"),
                    (WORKLOADS_FILE, records),
                    (TARGET_FILE, to_json(problem.target()) + "\n"),
                ):
                    with safe_file(path / name, "w") as stream:
                        stream.write(text)
            else:
                path = Path(value).expanduser().resolve(strict=True)
                problem = load_problem(path, backend=backend)
            if get_backend(problem.language, problem.platform).kind != backend.kind:
                raise ValueError(
                    f"{problem_id}: problem target differs from service backend"
                )
            # Keep the registered contract for detecting later directory changes.
            problem.resolve_inputs()
            self.problems[problem_id] = problem
            self.problem_paths[problem_id] = path
        if not self.problems:
            raise ValueError("register at least one problem")

    def problem(self, problem_id: str) -> Problem:
        """Return a registered problem, raising `KeyError` for an unknown task."""
        try:
            return self.problems[identifier(problem_id)]
        except KeyError:
            raise KeyError(f"unknown problem: {problem_id}") from None

    def info(self) -> dict:
        """Return task metadata and cached information from the worker runtime.

        The first call queries the worker under the device lock. Later calls reuse
        that result and return a copy of its device metadata.
        """
        if self.device_info is None:
            with self.lock:
                if self.device_info is None:
                    runtime = self.runtime_factory(timeout=self.timeout)
                    self.device_info = runtime.info(
                        self.backend, build_root=self.root / "info" / "work" / "build",
                        log_dir=self.root / "info" / "evaluations",
                    )
        return {
            "backend": str(self.backend.kind),
            "kernel_kind": self.backend.kernel_type.artifact_kind,
            "device": deepcopy(self.device_info),
            "problems": list(self.problems),
            "problem_paths": {name: str(path) for name, path in self.problem_paths.items()},
            "timeout": self.timeout,
        }

    def check_problem(self, problem_id: str) -> dict:
        """Reject changes to the registered directory and return its reference."""
        problem = self.problem(problem_id)
        reference = problem_reference(self.problem_paths[problem_id], problem)
        if load_problem(reference) != problem:
            raise ValueError(f"{problem_id}: registered problem changed")
        return reference

    def kernel(self, problem_id: str, value: dict) -> Kernel:
        value = check_submission(value)
        if value["kind"] != self.backend.kernel_type.artifact_kind:
            raise ValueError("kernel kind differs from service backend")
        reference = self.check_problem(problem_id)
        metadata = {
            "name": value["name"],
            "compile_flags": value["compile_flags"],
            "validation": None,
            "problem": reference,
        }
        return self.backend.kernel_type(
            Node.directory(
                ".",
                (
                    json_node("metadata.json", metadata),
                    Node.directory(
                        "submission", source_tree(value["sources"]).children
                    ),
                ),
            )
        )

    def result(self, job_id: str) -> Evaluation:
        """Read a completed job only while its registered problem is unchanged."""
        path = self.root / "jobs" / identifier(job_id) / "result.json"
        if not path.is_file():
            raise KeyError(f"unknown or unfinished job: {job_id}")
        result = restore(strict_json(path.read_text()), Evaluation)
        self.check_problem(result.problem_id)
        request = strict_json((path.parent / "request.json").read_text())
        if request.get("problem_digest") != problem_digest(self.problem(result.problem_id)):
            raise ValueError("job belongs to a different registered problem")
        return result

    def evaluate(self, request: dict) -> Evaluation:
        """Evaluate submitted sources against a registered problem.

        Args:
            request: Mapping with `problem_id`, `submission`, optional `reference`
                and `timeout`. A reference is a source submission or an accepted
                job ID for the same problem; it is only a performance baseline.

        Returns:
            Persisted evaluation record. Worker failures and expired device-queue
            deadlines produce failed validation and diagnostics.

        Raises:
            ValueError: Request fields, sources, target or baseline are invalid.
            KeyError: A required field, problem or baseline record is missing.
        """
        started = time.monotonic()
        allowed = {"problem_id", "submission", "reference", "timeout"}
        if not isinstance(request, dict) or set(request) - allowed:
            raise ValueError("unsupported evaluation fields")
        problem_id = request["problem_id"]
        kernel = self.kernel(problem_id, request["submission"])
        workload_ids = tuple(value["uuid"] for value in self.problem(problem_id).workloads)
        budget = min(
            self.timeout, deadline_seconds(request.get("timeout", self.timeout))
        )
        reference = request.get("reference")
        if isinstance(reference, str):
            measured = self.result(reference)
            if measured.problem_id != problem_id or not measured.accepted:
                raise ValueError("baseline must be an accepted result for this problem")
            reference = self.kernel(problem_id, submission(measured.kernel))
        elif reference is not None:
            reference = self.kernel(problem_id, reference)

        job_id = uuid.uuid4().hex
        directory = self.root / "jobs" / job_id
        directory.mkdir(parents=True, mode=0o700)
        write_json(directory / "request.json", request | {
            "problem_digest": problem_digest(self.problem(problem_id)),
        })
        failed = ValidationResult(False, False, False)
        validation = (
            ValidationResult.combine(dict.fromkeys(workload_ids, failed))
            if workload_ids else failed
        )
        error = None
        locked = False
        try:
            if not workload_ids:
                raise ValueError("problem has no workloads to evaluate")
            locked = self.lock.acquire(
                timeout=max(0, budget - (time.monotonic() - started))
            )
            remaining = budget - (time.monotonic() - started)
            if not locked or remaining <= 0:
                raise TimeoutError(
                    "evaluation deadline expired while waiting for device"
                )
            runtime = self.runtime_factory(timeout=remaining)
            remaining = budget - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("evaluation deadline expired during runtime preparation")
            measured = runtime.evaluate(
                kernel,
                reference=reference,
                include_paths=self.include_paths,
                build_root=directory / "work" / "build",
                log_dir=directory / "evaluations",
                timeout=remaining,
            )
            if not isinstance(measured, ValidationResult):
                raise TypeError("runtime must return ValidationResult")
            if set(measured.workloads) != set(workload_ids):
                raise ValueError("runtime results must cover exactly the registered workloads")
            self.check_problem(problem_id)
            validation = measured
            if time.monotonic() - started >= budget:
                raise TimeoutError("evaluation deadline expired before receiving results")
        except Exception as failure:
            error = f"{type(failure).__name__}: {failure}"
            (directory / "error.log").write_text(error + "\n")
        finally:
            if locked:
                self.lock.release()

        metadata = json.loads(kernel.find("metadata.json").content)
        metadata["validation"] = validation
        kernel = self.backend.kernel_type(
            kernel.tree.with_child(json_node("metadata.json", metadata))
        )
        logs = [directory / "error.log", *(directory / "evaluations").rglob("*")]
        diagnostics = {
            str(path.relative_to(directory)): path.read_text(errors="replace")[
                -DIAGNOSTIC_LIMIT:
            ]
            for path in sorted(logs)
            if not path.is_symlink()
            and path.is_file()
            and (path.suffix == ".log" or path.name.endswith("-process.json"))
        }
        result = Evaluation(
            job_id,
            problem_id,
            source_digest(problem_id, request["submission"]),
            kernel,
            validation,
            diagnostics,
            error,
        )
        write_json(directory / "result.json", result)
        return result

    def start(self, endpoint: str = "http://127.0.0.1:0") -> RunningServer:
        """Query worker metadata and start a background listener.

        Args:
            endpoint: `http://host:port` or `unix:///absolute/socket/path`.
                TCP port zero selects an available port.

        Returns:
            Started server handle; use it as a context manager or call `close()`.
        """
        self.info()
        return RunningServer(self, endpoint).start()


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "KLineage"

    def setup(self):
        super().setup()
        self.connection.settimeout(HTTP_READ_TIMEOUT)

    def log_message(self, format, *args):
        pass

    def send_json(self, status: HTTPStatus, payload):
        data = to_json(payload, indent=None).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.dispatch()

    def do_POST(self):
        self.dispatch()

    def dispatch(self):
        service = self.server.service
        try:
            path = urlsplit(self.path).path
            if self.command == "GET":
                payload = self.get_resource(service, path)
            elif path == "/evaluate":
                if self.headers.get_all("Transfer-Encoding"):
                    raise ValueError("chunked requests are unsupported")
                lengths = self.headers.get_all("Content-Length", [])
                if (
                    len(lengths) != 1
                    or not lengths[0].isascii()
                    or not lengths[0].isdecimal()
                ):
                    raise ValueError("one decimal Content-Length is required")
                length = int(lengths[0])
                if not 0 < length <= REQUEST_LIMIT:
                    self.send_json(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        {"error": "invalid request size"},
                    )
                    return
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("incomplete request body")
                payload = service.evaluate(strict_json(body))
            else:
                raise KeyError("unknown endpoint")
            self.send_json(HTTPStatus.OK, payload)
        except KeyError as error:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": str(error)})
        except (ValueError, TypeError, OSError, RecursionError) as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception as error:
            self.send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"{type(error).__name__}: {error}"},
            )

    def get_resource(self, service, path):
        if path == "/health":
            return {"status": "ok"}
        if path == "/info":
            return service.info()
        if path.startswith("/problems/"):
            problem_id = unquote(path.removeprefix("/problems/"))
            problem = service.problem(problem_id)
            return {
                "problem_id": problem_id,
                "problem": problem,
            }
        if path.startswith("/jobs/"):
            return service.result(path.removeprefix("/jobs/"))
        raise KeyError("unknown endpoint")


class UnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    pass


class RunningServer:
    """Own a bound HTTP or Unix listener and its serving thread.

    Construction binds the endpoint; `start()` begins serving. `Service.start()`
    performs both steps. The context manager closes the listener on exit but
    does not start an unstarted handle. TLS termination is external.

    Args:
        service: Service implementation handling evaluation requests.
        endpoint: `http://host:port` or `unix:///absolute/socket/path`.

    Attributes:
        endpoint: Bound address, including the allocated port when zero was given.
    """

    def __init__(self, service: Service, endpoint: str):
        if not isinstance(endpoint, str) or any(char in endpoint for char in "\0\r\n\t"):
            raise ValueError("listen endpoint must be a URL without control characters")
        target = urlsplit(endpoint)
        if target.username or target.password or target.query or target.fragment:
            raise ValueError("listen endpoint cannot contain credentials, query or fragment")
        self.socket_path = None
        self.socket_identity = None
        if target.scheme == "unix":
            path = unquote(target.path)
            if target.netloc or not path.startswith("/") or not path.rstrip("/") or "\0" in path:
                raise ValueError("Unix listen endpoint needs an absolute socket path")
            self.socket_path = Path(path)
            self.socket_path.parent.mkdir(parents=True, exist_ok=True)
            self.server = UnixHTTPServer(str(self.socket_path), RequestHandler)
            self.socket_path.chmod(0o600)
            info = self.socket_path.lstat()
            self.socket_identity = (info.st_dev, info.st_ino)
            self.endpoint = "unix://" + str(self.socket_path)
        elif target.scheme == "http" and target.hostname:
            if target.path not in ("", "/") or any(char.isspace() for char in target.hostname):
                raise ValueError("HTTP listen endpoint must use only a hostname and port")
            self.server = ThreadingHTTPServer(
                (target.hostname, target.port or 0), RequestHandler
            )
            host, port = self.server.server_address
            self.endpoint = f"http://{host}:{port}"
        else:
            raise ValueError("listen endpoint must be http://host:port or unix:///path")
        # Drain request handlers before their service state can be removed.
        self.server.daemon_threads = False
        self.server.service = service
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        """Start the serving thread once and return this handle."""
        if self.server is None:
            raise RuntimeError("server is closed")
        self.thread.start()
        return self

    def close(self):
        """Stop accepting requests and drain handlers; repeated calls are harmless."""
        server, self.server = self.server, None
        if server is None:
            return
        if self.thread.ident is not None:
            server.shutdown()
            self.thread.join()
        server.server_close()
        if self.socket_path is not None:
            try:
                info = self.socket_path.lstat()
            except FileNotFoundError:
                return
            if (info.st_dev, info.st_ino) == self.socket_identity:
                self.socket_path.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

"""Expose registered problems through a workspace Unix socket."""

from pathlib import Path
from urllib.parse import quote

from klineage.serialization import restore

from .client import Client
from .protocol import Evaluation, deadline_seconds
from .service import RunningServer, identifier


class ClientGateway:
    """Expose a fixed set of remote problems through a local Unix socket.

    The upstream client's network access remains on the host. Other problems
    are rejected.
    Use as a context manager to start and close the listener.

    Args:
        client: Upstream client. A bound client exposes one problem; an unbound
            client exposes the registered set discovered at gateway construction.
        socket_path: Unix socket path to expose inside the agent sandbox.

    Attributes:
        endpoint: `unix://` URL of the local gateway.
    """

    def __init__(self, client: Client, socket_path: Path):
        problems = (client.problem_id,) if client.problem_id is not None else client.info()["problems"]
        self.problem_ids = tuple(identifier(value) for value in problems)
        if not self.problem_ids:
            raise ValueError("gateway requires at least one registered problem")
        self.client = client
        self.socket_path = Path(socket_path).absolute()
        self.endpoint = "unix://" + str(self.socket_path)
        self.running = None

    def info(self):
        info = self.client.info()
        return info | {
            "problems": list(self.problem_ids),
            "problem_paths": {
                name: path for name, path in info.get("problem_paths", {}).items()
                if name in self.problem_ids
            },
        }

    def problem(self, problem_id):
        if problem_id not in self.problem_ids:
            raise ValueError("gateway only exposes its registered problems")
        return self.client.problem(problem_id)

    def evaluate(self, request):
        if (
            not isinstance(request, dict)
            or request.get("problem_id") not in self.problem_ids
        ):
            raise ValueError("gateway only exposes its registered problems")
        budget = min(
            self.client.timeout,
            deadline_seconds(request.get("timeout", self.client.timeout)),
        )
        payload = self.client.request(
            "POST", "/evaluate", request | {"timeout": budget}, timeout=budget,
        )
        return self.checked_result(payload)

    def result(self, job_id):
        payload = self.client.request(
            "GET", "/jobs/" + quote(identifier(job_id), safe="")
        )
        return self.checked_result(payload)

    def checked_result(self, payload):
        if payload.get("problem_id") not in self.problem_ids:
            raise ValueError("gateway result belongs to another problem")
        return restore(payload, Evaluation)

    def start(self):
        """Start the socket listener and return this gateway."""
        self.running = RunningServer(self, self.endpoint).start()
        return self

    def close(self):
        """Close the listener, if started."""
        if self.running is not None:
            self.running.close()
            self.running = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.close()

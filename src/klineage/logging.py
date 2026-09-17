"""One append-only timeline for kernel measurements and model spend.

`evaluate` and `profile` append a record when they finish; the API proxy appends
one per model round trip. Both carry an ISO timestamp, so a kernel's latency and
the tokens spent producing it line up on the same axis.

Collection is off unless KLINEAGE_LOG_DIR names a directory. Nothing here raises:
observability must never fail the work it observes.
"""

from __future__ import annotations

import functools
import json
import os
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

#: Set to a directory to collect records; unset disables collection entirely.
LOG_ENV = "KLINEAGE_LOG_DIR"
#: Records land here under the log directory.
LOG_FILE = "events.jsonl"
#: Responses-API event carrying one round trip's usage.
COMPLETED = "response.completed"


def now() -> str:
    return datetime.now(UTC).isoformat()


def log_dir() -> Path | None:
    """The configured record directory, or None when collection is off."""

    value = os.environ.get(LOG_ENV, "").strip()
    return Path(value).expanduser() if value else None


def record(event: str, /, *, directory: Path | None = None, **fields: Any) -> None:
    """Append one event. Never raises: collection is best effort.

    `directory` names the target explicitly, for a writer that serves a workspace
    other than this process's own; None falls back to the configured directory.
    """

    target = log_dir() if directory is None else directory
    if target is None:
        return
    entry = {"event": event, "at": now(), **fields}
    try:
        target.mkdir(parents=True, exist_ok=True)
        with (target / LOG_FILE).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001 - collection must not fail the caller
        pass


def kernel_fields(kernel: Any) -> dict[str, Any]:
    """Identify a kernel without embedding its sources."""

    try:
        return {
            "kernel_name": getattr(kernel, "name", None),
            "fingerprint": kernel.fingerprint,
            "platform": kernel.problem.platform,
        }
    except Exception:  # noqa: BLE001
        return {}


def measurement_fields(result: Any) -> dict[str, Any]:
    """Flatten a ValidationResult or profile result into record fields.

    Only the fields the timeline has always carried are emitted, so a record's
    shape matches the ones already collected. A paired reference latency and its
    ratio stay out; the comparison reports them separately.
    """

    fields: dict[str, Any] = {}
    for name in ("compile_passed", "correctness_passed", "profile_passed"):
        value = getattr(result, name, None)
        if value is not None:
            fields[name] = value
    latency = getattr(result, "latency_ms", None)
    if latency is not None:
        fields["latency_ms"] = latency
    if isinstance(result, dict):
        if isinstance(result.get("metrics"), list):
            fields["metrics"] = len(result["metrics"])
        for name in ("tool", "tool_version"):
            if name in result:
                fields[name] = result[name]
    return fields


def observed(event: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a measurement so every call appends one event with its outcome.

    The kernel is the first positional argument, matching `evaluate` and `profile`.
    Failures are recorded and re-raised unchanged, so observing a call never
    changes what the caller sees.
    """

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(function)
        def wrapper(kernel, *args, **kwargs):
            started = now()
            fields = kernel_fields(kernel)
            try:
                result = function(kernel, *args, **kwargs)
            except Exception as error:
                record(event, at=started, finished_at=now(), **fields,
                       error=f"{type(error).__name__}: {error}")
                raise
            record(event, at=started, finished_at=now(), **fields,
                   **measurement_fields(result))
            return result

        return wrapper

    return decorate


def usage_from_event(payload: bytes) -> dict[str, Any] | None:
    """Return the usage of a completed response event; None for any other frame."""

    try:
        event = json.loads(payload)
    except (ValueError, TypeError):
        return None
    if not isinstance(event, dict) or event.get("type") != COMPLETED:
        return None
    response = event.get("response")
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    return {
        "response_id": response.get("id"),
        "model": response.get("model"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens"),
        "reasoning_tokens": (usage.get("output_tokens_details") or {}).get(
            "reasoning_tokens"
        ),
    }


#: Headers a forwarded request must not inherit from the client.
HOP_BY_HOP = frozenset(
    {
        "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
        "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
    }
)


def data_lines(chunk: bytes) -> Iterator[bytes]:
    """Yield each SSE payload, ignoring every other line."""

    for line in chunk.split(b"\n"):
        line = line.strip()
        if line.startswith(b"data:"):
            yield line[len(b"data:") :].strip()


def chunk(payload: bytes) -> bytes:
    """Frame one HTTP/1.1 chunked-transfer segment."""

    return b"%x\r\n%s\r\n" % (len(payload), payload)


#: Terminator for a chunked HTTP/1.1 body.
LAST_CHUNK = b"0\r\n\r\n"


def complete_events(buffer: bytes) -> tuple[list[bytes], bytes]:
    """Split a stream buffer into complete SSE frames, returning the remainder.

    A frame ends at a blank line. One split across reads must not be parsed
    early, so the trailing partial frame carries into the next call.
    """

    frames = []
    while b"\n\n" in buffer:
        frame, buffer = buffer.split(b"\n\n", 1)
        frames.append(frame)
    return frames, buffer


def record_usage(frames: Iterator[bytes], directory: Path | None = None) -> None:
    """Record one api_response event per completed response frame."""

    for frame in frames:
        for payload in data_lines(frame):
            if payload and payload != b"[DONE]":
                usage = usage_from_event(payload)
                if usage is not None:
                    record("api_response", directory=directory, **usage)


def relay(
    source,
    sink,
    *,
    chunked: bool = True,
    directory: Path | None = None,
) -> None:
    """Copy an upstream body to the client, recording usage as frames arrive.

    The body is re-framed when it is chunked, because only whole chunks may be
    written; a declared but absent framing would corrupt the client's parse.
    """

    pending = b""
    for data in iter(lambda: source.read(8192), b""):
        sink.write(chunk(data) if chunked else data)
        sink.flush()
        frames, pending = complete_events(pending + data)
        record_usage(iter(frames), directory)
    if pending:
        record_usage(iter([pending]), directory)
    if chunked:
        sink.write(LAST_CHUNK)
        sink.flush()


def handler_for(upstream: str, directory: Path | None = None):
    """Build a request handler forwarding to `upstream`, e.g. https://host/api/v1."""

    class Proxy(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep the default access log quiet
            pass

        def forward(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else None
            headers = {
                name: value
                for name, value in self.headers.items()
                if name.lower() not in HOP_BY_HOP
            }
            path = self.path if self.path.startswith("/") else "/" + self.path
            request = Request(
                upstream.rstrip("/") + path,
                data=body,
                headers=headers,
                method=self.command,
            )
            try:
                with urlopen(request, timeout=None) as response:
                    # Upstream framing headers are dropped above, so the body is
                    # re-framed here; every response to the client is chunked.
                    self.send_response(response.status)
                    for name, value in response.headers.items():
                        if name.lower() not in HOP_BY_HOP:
                            self.send_header(name, value)
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    relay(response, self.wfile, chunked=True, directory=directory)
            except HTTPError as error:
                payload = error.read()
                self.send_response(error.code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception as error:  # noqa: BLE001 - report upstream failures
                record("api_proxy_error", directory=directory, error=repr(error))
                payload = json.dumps({"error": {"message": str(error)}}).encode()
                try:
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except Exception:  # noqa: BLE001
                    pass

        do_GET = forward
        do_POST = forward
        do_DELETE = forward

    return Proxy


class UsageProxy:
    """Serve `upstream` on a local port, recording each round trip's usage.

    `directory` pins the target explicitly. Without it the proxy would fall back
    to the configured directory, which only works when the caller and the process
    serving requests share one environment; a proxy started to serve another
    process's runs should name its directory instead.
    """

    def __init__(
        self,
        upstream: str,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        directory: Path | None = None,
    ):
        self.upstream = upstream
        self.directory = directory
        handler = handler_for(upstream, directory)
        self.server = ThreadingHTTPServer((host, port), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> UsageProxy:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


__all__ = [
    "LOG_ENV",
    "LOG_FILE",
    "UsageProxy",
    "handler_for",
    "kernel_fields",
    "log_dir",
    "measurement_fields",
    "now",
    "observed",
    "record",
    "relay",
    "usage_from_event",
]

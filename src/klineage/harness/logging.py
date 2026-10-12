"""One append-only timeline for kernel measurements and model spend.

`evaluate` appends a record when it finishes; the API proxy appends
one per model round trip. Both carry an ISO timestamp, so a kernel's latency and
the tokens spent producing it line up on the same axis.

An explicit directory or KLINEAGE_LOG_DIR enables collection. Event writes are
best effort.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from klineage.serialization import raw_text, to_json
from klineage.utils import safe_file

#: Set to a directory to collect records; unset disables collection entirely.
LOG_ENV = "KLINEAGE_LOG_DIR"
#: Records land here under the log directory.
LOG_FILE = "stats.jsonl"
#: Responses-API event carrying one round trip's usage.
COMPLETED = "response.completed"
USAGE_BUFFER_BYTES = 1024 * 1024
SSE_BOUNDARY = re.compile(br"\r?\n\r?\n")


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
        with safe_file(target / LOG_FILE, "a") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.write(to_json(entry, indent=None) + "\n")
            stream.flush()
    except Exception:  # noqa: BLE001 - collection must not fail the caller
        pass


def kernel_fields(kernel: Any) -> dict[str, Any]:
    """Identify a kernel without embedding its sources."""

    try:
        metadata = json.loads(raw_text(kernel.find("metadata.json")))
        return {
            "kernel_name": metadata["name"],
            "platform": metadata["problem"]["platform"],
        }
    except Exception:  # noqa: BLE001
        return {}


def measurement_fields(result: Any) -> dict[str, Any]:
    """Record summary gates and preserve each workload's measurements."""

    fields: dict[str, Any] = {}
    for name in ("compile_passed", "correctness_passed", "profile_passed"):
        value = getattr(result, name, None)
        if value is not None:
            fields[name] = value
    latency = getattr(result, "latency_ms", None)
    if latency is not None:
        fields["latency_ms"] = latency
    workloads = getattr(result, "workloads", None)
    if workloads:
        fields["workloads"] = dict(workloads)
    return fields


def usage_from_event(payload: bytes) -> dict[str, Any] | None:
    """Read usage from a completed SSE event or a non-streamed response."""

    try:
        event = json.loads(payload)
    except (ValueError, TypeError, RecursionError):
        return None
    if not isinstance(event, dict) or event.get("type") not in (None, COMPLETED):
        return None
    response = event.get("response") if event.get("type") == COMPLETED else event
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    input_details = usage.get("input_tokens_details")
    output_details = usage.get("output_tokens_details")
    return {
        "response_id": response.get("id"),
        "model": response.get("model"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": input_details.get("cached_tokens")
        if isinstance(input_details, dict) else None,
        "reasoning_tokens": output_details.get("reasoning_tokens")
        if isinstance(output_details, dict) else None,
    }


def data_lines(chunk: bytes) -> Iterator[bytes]:
    """Yield each SSE payload, ignoring every other line."""

    for line in chunk.split(b"\n"):
        line = line.strip()
        if line.startswith(b"data:"):
            yield line[len(b"data:") :].strip()


def complete_events(buffer: bytes) -> tuple[list[bytes], bytes]:
    """Split a stream buffer into complete SSE frames, returning the remainder.

    A frame ends at a blank line. One split across reads must not be parsed
    early, so the trailing partial frame carries into the next call.
    """

    frames = SSE_BOUNDARY.split(buffer)
    return frames[:-1], frames[-1]


class ResponseUsage:
    """Inspect streamed frames or JSON with bounded buffering, retaining usage only."""

    def __init__(self, content_type: str):
        media_type = content_type.partition(";")[0].strip().lower()
        self.streaming = media_type == "text/event-stream" if media_type else None
        self.pending = b""
        self.discarding = False
        self.usage = {}
        self.message = {}

    def feed(self, data: bytes) -> None:
        # Some Responses providers stream SSE without a Content-Type header.
        if self.streaming is None:
            data = data.lstrip()
            if not data:
                return
            self.streaming = not data.startswith((b"{", b"["))

        if not self.streaming:
            if self.discarding:
                return
            self.pending += data
            if len(self.pending) > USAGE_BUFFER_BYTES:
                self.pending = b""
                self.discarding = True
            return

        frames, pending = complete_events(self.pending + data)
        if self.discarding and frames:
            frames = frames[1:]
            self.discarding = False
        for frame in frames:
            if len(frame) <= USAGE_BUFFER_BYTES:
                self.capture(b"\n".join(data_lines(frame)))
        self.discarding = self.discarding or len(pending) > USAGE_BUFFER_BYTES
        # Keep enough bytes to recognize a CRLF separator split across reads.
        self.pending = pending[-3:] if self.discarding else pending

    def capture(self, payload: bytes):
        try:
            event = json.loads(payload)
        except (ValueError, TypeError, RecursionError):
            return
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        if kind not in ("message", "message_start", "message_delta"):
            self.usage = usage_from_event(payload) or self.usage
            return
        message = event.get("message", {}) if kind == "message_start" else event
        if not isinstance(message, dict) or not isinstance(message.get("usage"), dict):
            return
        # Message deltas carry cumulative output usage, not a new request.
        self.message.update(message["usage"])
        for key in ("id", "model"):
            if key in message:
                self.message[key] = message[key]
        usage = self.message
        fields = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
        if any(type(usage.get(key, 0)) is not int or usage.get(key, 0) < 0 for key in fields):
            return
        cached = usage.get("cache_read_input_tokens", 0)
        created = usage.get("cache_creation_input_tokens", 0)
        inputs = usage.get("input_tokens", 0) + cached + created
        outputs = usage.get("output_tokens", 0)
        self.usage = {
            "response_id": usage.get("id"), "model": usage.get("model"),
            "input_tokens": inputs, "output_tokens": outputs,
            "total_tokens": inputs + outputs, "cached_tokens": cached,
            "cache_creation_tokens": created,
        }

    def finish(self) -> dict[str, Any]:
        if self.pending and not self.discarding:
            payload = b"\n".join(data_lines(self.pending)) if self.streaming else self.pending
            self.capture(payload)
        return self.usage


__all__ = [
    "LOG_ENV",
    "LOG_FILE",
    "kernel_fields",
    "log_dir",
    "measurement_fields",
    "now",
    "record",
    "usage_from_event",
]

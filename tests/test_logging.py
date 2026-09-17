"""Tests for the measurement timeline and its usage-recording proxy."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen

from klineage.logging import (
    LOG_ENV,
    UsageProxy,
    complete_events,
    observed,
    record,
    usage_from_event,
)


def completed_frame(identifier: str = "r1") -> bytes:
    """One SSE frame carrying a Responses-API usage report."""

    payload = {
        "type": "response.completed",
        "response": {
            "id": identifier,
            "model": "deepseek-flash",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
                "input_tokens_details": {"cached_tokens": 4},
                "output_tokens_details": {"reasoning_tokens": 1},
            },
        },
    }
    return f"event: response.completed\ndata: {json.dumps(payload)}\n\n".encode()


class FakeUpstream(BaseHTTPRequestHandler):
    """Answer every request with one completed response frame."""

    protocol_version = "HTTP/1.1"
    body = b""

    def log_message(self, *args):
        pass

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)


def events(directory: Path) -> list[dict]:
    path = directory / "events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class FakeKernel:
    """The shape `kernel_fields` reads: name, fingerprint, problem.platform."""

    name = "k"
    fingerprint = "fp"

    class problem:
        platform = "p"


class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.target = self.root / "stats"

    def tearDown(self):
        self.temp.cleanup()

    def collecting(self, target: Path | None = None):
        """Collect into a directory for the duration of a `with` block."""

        return patch.dict(os.environ, {LOG_ENV: str(target or self.target)})

    def test_collection_is_off_without_configuration(self):
        environment = {k: v for k, v in os.environ.items() if k != LOG_ENV}
        with patch.dict(os.environ, environment, clear=True):
            record("ignored")
        self.assertFalse((self.target / "events.jsonl").exists())

    def test_record_targets_an_explicit_directory(self):
        with self.collecting():
            record("probe", value=1)
        [entry] = events(self.target)
        self.assertEqual(entry["event"], "probe")
        self.assertEqual(entry["value"], 1)
        self.assertIn("at", entry)

    def test_observed_records_outcome_and_reraises(self):
        class Measurement:
            """The ValidationResult shape `measurement_fields` flattens."""

            compile_passed = True
            correctness_passed = True
            profile_passed = True
            latency_ms = 2.0
            reference_latency_ms = 4.0

        @observed("evaluate")
        def succeeds(kernel):
            return Measurement()

        @observed("evaluate")
        def fails(kernel):
            raise RuntimeError("boom")

        with self.collecting():
            with self.assertRaises(RuntimeError):
                fails(FakeKernel())
            self.assertEqual(succeeds(FakeKernel()).latency_ms, 2.0)

        failure, success = events(self.target)
        self.assertEqual(failure["error"], "RuntimeError: boom")
        self.assertEqual(failure["kernel_name"], "k")
        self.assertTrue(success["profile_passed"])
        self.assertIn("finished_at", success)
        # A record's shape stays fixed: no reference latency or derived ratio.
        self.assertEqual(
            list(success),
            ["event", "at", "finished_at", "kernel_name", "fingerprint",
             "platform", "compile_passed", "correctness_passed",
             "profile_passed", "latency_ms"],
        )

    def test_usage_parsing_ignores_other_frames(self):
        self.assertIsNone(usage_from_event(b'{"type":"response.in_progress"}'))
        self.assertIsNone(usage_from_event(b"not json"))
        usage = usage_from_event(completed_frame().split(b"data: ")[1].strip())
        self.assertEqual(usage["response_id"], "r1")
        self.assertEqual(usage["cached_tokens"], 4)
        self.assertEqual(usage["reasoning_tokens"], 1)

    def test_complete_events_holds_a_partial_frame(self):
        frames, pending = complete_events(b"data: a\n\ndata: par")
        self.assertEqual(frames, [b"data: a"])
        self.assertEqual(pending, b"data: par")


class UsageProxyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        FakeUpstream.body = completed_frame()
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstream)
        threading.Thread(target=self.upstream.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.upstream.server_address[1]}"

    def tearDown(self):
        self.upstream.shutdown()
        self.upstream.server_close()
        self.temp.cleanup()

    def test_proxy_forwards_and_records_usage(self):
        target = self.root / "stats"
        with UsageProxy(self.url, directory=target) as proxy:
            body = urlopen(
                Request(proxy.url + "/v1/responses", data=b"{}"), timeout=10
            ).read()
        # The client sees the upstream frame, re-framed as chunked.
        self.assertIn(b"response.completed", body)
        [entry] = events(target)
        self.assertEqual(entry["event"], "api_response")
        self.assertEqual(entry["total_tokens"], 12)

    def test_proxy_records_into_the_named_directory_only(self):
        named, untouched = self.root / "named", self.root / "untouched"
        with UsageProxy(self.url, directory=named) as proxy:
            urlopen(Request(proxy.url + "/v1/responses", data=b"{}"), timeout=10).read()
        self.assertTrue((named / "events.jsonl").is_file())
        self.assertFalse(untouched.exists())


if __name__ == "__main__":
    unittest.main()

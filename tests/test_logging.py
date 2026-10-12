from __future__ import annotations

import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from klineage.harness.logging import LOG_ENV, USAGE_BUFFER_BYTES, ResponseUsage, record


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


def events(directory: Path) -> list[dict]:
    path = directory / "stats.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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

    def test_anthropic_stream_usage(self):
        frames = [
            {"type": "message_start", "message": {
                "id": "msg-1", "model": "claude-test", "usage": {
                    "input_tokens": 10, "output_tokens": 1,
                    "cache_creation_input_tokens": 4, "cache_read_input_tokens": 6,
                },
            }},
            {"type": "content_block_delta", "delta": {"text": "private"}},
            {"type": "message_delta", "usage": {"output_tokens": 7}},
            {"type": "message_stop"},
        ]
        stream = b"".join(b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames)
        usage = ResponseUsage("text/event-stream")
        for offset in range(0, len(stream), 7):
            usage.feed(stream[offset:offset + 7])
        self.assertEqual(usage.finish(), {
            "response_id": "msg-1", "model": "claude-test", "input_tokens": 20,
            "output_tokens": 7, "total_tokens": 27, "cached_tokens": 6,
            "cache_creation_tokens": 4,
        })

    def test_collection_is_off_without_configuration(self):
        environment = {k: v for k, v in os.environ.items() if k != LOG_ENV}
        with patch.dict(os.environ, environment, clear=True):
            record("ignored")
        self.assertFalse((self.target / "stats.jsonl").exists())

    def test_record_targets_an_explicit_directory(self):
        with self.collecting():
            record("probe", value=1)
        [entry] = events(self.target)
        self.assertEqual(entry["event"], "probe")
        self.assertEqual(entry["value"], 1)
        self.assertIn("at", entry)

    def test_concurrent_records(self):
        payload = "content" * 10000
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(record, "concurrent", directory=self.target,
                                index=index, payload=payload)
                for index in range(20)
            ]
            for future in futures:
                future.result()
        entries = events(self.target)
        self.assertEqual({entry["index"] for entry in entries}, set(range(20)))
        self.assertTrue(all(entry["payload"] == payload for entry in entries))

    def test_split_usage_lines(self):
        for ending in (b"\n", b"\r\n"):
            with self.subTest(ending=ending):
                collector = ResponseUsage("text/event-stream; charset=utf-8")
                for byte in completed_frame().replace(b"\n", ending):
                    collector.feed(bytes([byte]))
                self.assertEqual(collector.finish()["total_tokens"], 12)

    def test_nonstreaming_usage(self):
        response = json.loads(completed_frame().split(b"data: ")[1].strip())["response"]
        collector = ResponseUsage("application/json")
        payload = json.dumps(response).encode()
        for offset in range(0, len(payload), 13):
            collector.feed(payload[offset:offset + 13])
        self.assertEqual(collector.finish()["total_tokens"], 12)

    def test_oversized_frames(self):
        collector = ResponseUsage("text/event-stream")
        for _ in range(3):
            collector.feed(b"x" * USAGE_BUFFER_BYTES)
            self.assertLessEqual(len(collector.pending), USAGE_BUFFER_BYTES)
        collector.feed(b"\r\n\r\n" + completed_frame())
        self.assertEqual(collector.finish()["total_tokens"], 12)

        collector = ResponseUsage("application/json")
        collector.feed(b"x" * (USAGE_BUFFER_BYTES + 1))
        self.assertEqual(collector.finish(), {})
        self.assertLessEqual(len(collector.pending), USAGE_BUFFER_BYTES)


if __name__ == "__main__":
    unittest.main()

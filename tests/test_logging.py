"""Tests for the measurement timeline and streamed model usage."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from klineage.serialization import raw_text
from klineage.harness.logging import (
    LOG_ENV,
    USAGE_BUFFER_BYTES,
    ResponseUsage,
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


def events(directory: Path) -> list[dict]:
    path = directory / "stats.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fake_kernel():
    from kernel_fixtures import kernel

    return kernel("k", platform="sm120")


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
        self.assertFalse((self.target / "stats.jsonl").exists())

    def test_disabled_skips_metadata(self):
        @observed("evaluate")
        def evaluate(value):
            return value

        with (
            patch.dict(os.environ, {LOG_ENV: ""}),
            patch("klineage.harness.logging.kernel_fields", side_effect=AssertionError("unused metadata")),
        ):
            self.assertEqual(evaluate("result"), "result")

    def test_metadata_without_problem(self):
        from klineage.artifact import json_node
        from klineage.harness.logging import kernel_fields

        current = fake_kernel()
        metadata = json.loads(raw_text(current.find("metadata.json")))
        metadata["problem"]["path"] = str(self.root / "missing-problem")
        current = type(current)(current.tree.with_child(json_node("metadata.json", metadata)))
        self.assertEqual(kernel_fields(current), {"kernel_name": "k", "platform": "sm120"})

    def test_no_symlink_log_file(self):
        outside = self.root / "host-file"
        outside.write_text("private")
        self.target.mkdir()
        (self.target / "stats.jsonl").symlink_to(outside)
        record("attack", directory=self.target)
        self.assertEqual(outside.read_text(), "private")

    def test_no_symlink_log_parent(self):
        outside = self.root / "host-directory"
        outside.mkdir()
        self.target.symlink_to(outside, target_is_directory=True)
        record("attack", directory=self.target / "nested")
        self.assertEqual(list(outside.iterdir()), [])

    def test_no_hardlink_log_file(self):
        outside = self.root / "host-file"
        outside.write_text("private")
        self.target.mkdir()
        (self.target / "stats.jsonl").hardlink_to(outside)
        record("attack", directory=self.target)
        self.assertEqual(outside.read_text(), "private")

    def test_log_directory_race(self):
        outside = self.root / "host-directory"
        outside.mkdir()
        self.target.mkdir()
        moved = self.root / "original-stats"
        original_open = os.open

        def replace_parent(path, flags, mode=0o777, *, dir_fd=None):
            if path == "stats.jsonl" and not moved.exists():
                self.target.rename(moved)
                self.target.symlink_to(outside, target_is_directory=True)
            return original_open(path, flags, mode, dir_fd=dir_fd)

        with patch("klineage.utils.os.open", side_effect=replace_parent):
            record("race", directory=self.target)
        self.assertEqual(list(outside.iterdir()), [])
        self.assertEqual(events(moved)[0]["event"], "race")

    def test_referenced_metadata(self):
        from klineage.artifact import Node, file_ref
        from klineage.harness.logging import kernel_fields

        current = fake_kernel()
        path = self.root / "metadata.json"
        path.write_text(raw_text(current.find("metadata.json")))
        referenced = type(current)(
            current.tree.with_child(
                Node.file("metadata.json", reference=file_ref(path))
            )
        )
        self.assertEqual(kernel_fields(referenced), {"kernel_name": "k", "platform": "sm120"})

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
                fails(fake_kernel())
            self.assertEqual(succeeds(fake_kernel()).latency_ms, 2.0)

        failure, success = events(self.target)
        self.assertEqual(failure["error"], "RuntimeError: boom")
        self.assertEqual(failure["kernel_name"], "k")
        self.assertTrue(success["profile_passed"])
        self.assertIn("finished_at", success)
        # A record's shape stays fixed: no reference latency or derived ratio.
        self.assertEqual(
            list(success),
            ["event", "at", "finished_at", "kernel_name",
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

    def test_malformed_usage_details(self):
        payload = json.loads(completed_frame().split(b"data: ")[1].strip())
        payload["response"]["usage"]["input_tokens_details"] = 7
        payload["response"]["usage"]["output_tokens_details"] = ["unexpected"]
        usage = usage_from_event(json.dumps(payload).encode())
        self.assertEqual(usage["total_tokens"], 12)
        self.assertIsNone(usage["cached_tokens"])
        self.assertIsNone(usage["reasoning_tokens"])

    def test_complete_events_holds_a_partial_frame(self):
        frames, pending = complete_events(b"data: a\n\ndata: par")
        self.assertEqual(frames, [b"data: a"])
        self.assertEqual(pending, b"data: par")

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

    def test_usage_without_header(self):
        response = json.loads(completed_frame().split(b"data: ")[1].strip())["response"]
        for payload in (completed_frame(), json.dumps(response).encode()):
            with self.subTest(streaming=payload.startswith(b"event:")):
                collector = ResponseUsage("")
                collector.feed(b" \r\n")
                for byte in payload:
                    collector.feed(bytes([byte]))
                self.assertEqual(collector.finish()["total_tokens"], 12)
                self.assertEqual(collector.finish()["response_id"], "r1")

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

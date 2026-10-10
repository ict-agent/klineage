import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from kernel_fixtures import accepted, edit_kernel, kernel

from klineage.artifact import load_artifact, save_artifact
from klineage.harness.eval import ValidationResult
from klineage.serialization import from_json, restore, to_json


class KernelSchemaTests(unittest.TestCase):
    def test_invalid_latency(self):
        for latency in (float("nan"), float("inf"), True, "1.0"):
            with (
                self.subTest(latency=latency),
                self.assertRaises((TypeError, ValueError)),
            ):
                restore({
                        "compile_passed": True,
                        "correctness_passed": True,
                        "profile_passed": True,
                        "latency_ms": latency,
                    }, ValidationResult)

    def test_timing_requires_latency(self):
        with self.assertRaises(ValueError):
            ValidationResult(True, True, True)

    def test_invalid_reference_latency(self):
        for latency in (0, -1, float("nan"), float("inf"), True, "1.0"):
            for candidate in (None, 1.0):
                with (
                    self.subTest(reference=latency, candidate=candidate),
                    self.assertRaises((TypeError, ValueError)),
                ):
                    restore({
                            "compile_passed": True,
                            "correctness_passed": True,
                            "profile_passed": False,
                            "latency_ms": candidate,
                            "reference_latency_ms": latency,
                        }, ValidationResult)

    def test_reference_latency_optional(self):
        restored = restore({
                "compile_passed": True,
                "correctness_passed": True,
                "profile_passed": True,
                "latency_ms": 1.0,
            }, ValidationResult)
        self.assertIsNone(restored.reference_latency_ms)
        self.assertIsNone(json.loads(to_json(restored))["reference_latency_ms"])

    def test_roundtrip_schema(self):
        current = edit_kernel(kernel(), validation=replace(accepted(), reference_latency_ms=2.5))
        expected = {"artifact_type", "tree"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_artifact(current, root / "kernel.json")
            artifact = from_json((root / "kernel.json").read_text(), type(current))
            raw = json.loads(to_json(artifact))
            self.assertEqual(set(raw), expected)
            self.assertEqual(
                set(json.loads(to_json(artifact.validation))),
                {
                    "compile_passed",
                    "correctness_passed",
                    "profile_passed",
                    "latency_ms",
                    "reference_latency_ms",
                },
            )
            self.assertEqual(artifact.validation.reference_latency_ms, 2.5)
            self.assertEqual(load_artifact(root / "kernel.json", type(current)), current)

    def test_rejects_invalid_sources(self):
        for sources in (
            {"../escape.cu": "code"},
            {"kernel.cu": 1},
        ):
            with (
                self.subTest(sources=sources),
                self.assertRaises((TypeError, ValueError)),
            ):
                edit_kernel(kernel(), source_files=sources)

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from kernel_fixtures import accepted, kernel

from klineage.artifact.kernel import Kernel
from klineage.harness.artifacts import load_kernel, save_kernel
from klineage.harness.eval import ValidationResult


class KernelSchemaTests(unittest.TestCase):
    def test_invalid_latency(self):
        for latency in (float("nan"), float("inf"), True, "1.0"):
            with (
                self.subTest(latency=latency),
                self.assertRaises((TypeError, ValueError)),
            ):
                ValidationResult.from_dict(
                    {
                        "compile_passed": True,
                        "correctness_passed": True,
                        "profile_passed": True,
                        "latency_ms": latency,
                    }
                )

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
                    ValidationResult.from_dict(
                        {
                            "compile_passed": True,
                            "correctness_passed": True,
                            "profile_passed": False,
                            "latency_ms": candidate,
                            "reference_latency_ms": latency,
                        }
                    )

    def test_reference_latency_optional(self):
        restored = ValidationResult.from_dict(
            {
                "compile_passed": True,
                "correctness_passed": True,
                "profile_passed": True,
                "latency_ms": 1.0,
            }
        )
        self.assertIsNone(restored.reference_latency_ms)
        self.assertIsNone(restored.to_dict()["reference_latency_ms"])

    def test_roundtrip_schema(self):
        current = replace(
            kernel(), validation=replace(accepted(), reference_latency_ms=2.5)
        )
        expected = {"name", "problem", "source_files", "validation"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_kernel(current, root)
            raw = json.loads((root / "kernel.json").read_text())
            self.assertEqual(set(raw), expected)
            self.assertEqual(
                set(raw["validation"]),
                {
                    "compile_passed",
                    "correctness_passed",
                    "profile_passed",
                    "latency_ms",
                    "reference_latency_ms",
                },
            )
            self.assertEqual(raw["validation"]["reference_latency_ms"], 2.5)
            self.assertEqual(load_kernel(root), current)

    def test_sources_bind_fingerprint(self):
        current = kernel()
        helper = replace(
            current, source_files={**current.source_files, "helper.cuh": "// helper"}
        )
        self.assertNotEqual(helper.fingerprint, current.fingerprint)
        self.assertEqual(
            replace(current, validation=accepted()).fingerprint, current.fingerprint
        )
        restored = Kernel.from_dict(json.loads(json.dumps(helper.to_dict())))
        self.assertEqual(restored.fingerprint, helper.fingerprint)

    def test_rejects_invalid_sources(self):
        for sources in (
            {},
            {"../escape.cu": "code"},
            {"kernel.cu": "\0"},
            {"kernel.cu": 1},
        ):
            with (
                self.subTest(sources=sources),
                self.assertRaises((TypeError, ValueError)),
            ):
                replace(kernel(), source_files=sources)

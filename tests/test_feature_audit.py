import hashlib
import tempfile
import unittest
from pathlib import Path

from test_action import CUDA_SOURCE, FakeRunner, FakeSandbox, accepted, kernel

from klineage.action._state import _observe
from klineage.errors import StructuredOutputError
from klineage.kernel import Feature


class FeatureAuditTests(unittest.TestCase):
    def test_unknown_is_not_absent(self):
        feature = Feature("main", "mma")
        report = {"checks": [{
            "feature": feature.to_dict(), "status": "unknown", "evidence": [],
        }]}
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = FakeSandbox(FakeRunner(Path(temporary), [report]))
            with self.assertRaises(StructuredOutputError):
                _observe(sandbox, kernel(validation=accepted()), (feature,))

    def test_requires_source_evidence(self):
        feature = Feature("gemm.main", "mma")
        report = {"checks": [{
            "feature": feature.to_dict(), "status": "present",
            "reason": "MMA executes here.", "evidence": [{
                "path": "kernel.cu", "start": 1, "end": 1, "quote": "invented",
            }],
        }]}
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = FakeSandbox(FakeRunner(Path(temporary), [report]))
            with self.assertRaises(StructuredOutputError):
                _observe(sandbox, kernel(validation=accepted()), (feature,))

    def test_only_records_observed(self):
        present = Feature("gemm.main", "mma")
        absent = Feature("gemm.other", "mma")
        report = {"checks": [
            {"feature": present.to_dict(), "status": "present",
             "reason": "MMA executes at this locus.", "evidence": [{
                "path": "kernel.cu", "start": 1, "end": 1,
            }]},
            {"feature": absent.to_dict(), "status": "absent",
             "reason": "This locus uses scalar arithmetic.", "evidence": [{
                "path": "kernel.cu", "start": 1, "end": 1,
            }]},
        ]}
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = FakeSandbox(FakeRunner(Path(temporary), [report]))
            result = _observe(sandbox, kernel(validation=accepted()), (present, absent))

        self.assertEqual(result.features, (present,))
        checks = result.validation.details["feature_verifier"]["checks"]
        for check in checks:
            self.assertEqual(check["evidence"][0]["quote"], CUDA_SOURCE)
            self.assertEqual(
                check["evidence"][0]["sha256"], hashlib.sha256(CUDA_SOURCE.encode()).hexdigest(),
            )

    def test_absence_requires_evidence(self):
        feature = Feature("gemm.main", "mma")
        report = {"checks": [{
            "feature": feature.to_dict(), "status": "absent",
            "reason": "MMA is absent.", "evidence": [],
        }]}
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = FakeSandbox(FakeRunner(Path(temporary), [report]))
            with self.assertRaises(StructuredOutputError):
                _observe(sandbox, kernel(), (feature,))

    def test_requires_feature_reason(self):
        feature = Feature("gemm.main", "mma")
        for status in ("present", "absent"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
                report = {"checks": [{
                    "feature": feature.to_dict(), "status": status, "evidence": [{
                        "path": "kernel.cu", "start": 1, "end": 1, "quote": CUDA_SOURCE,
                    }],
                }]}
                sandbox = FakeSandbox(FakeRunner(Path(temporary), [report]))
                with self.assertRaises(StructuredOutputError):
                    _observe(sandbox, kernel(), (feature,))

    def test_feature_citation_bounds(self):
        feature = Feature("gemm.main", "mma")
        source = "\n".join(f"compute_{line}();" for line in range(13))
        for status in ("present", "absent"):
            for count, change in (
                (5, {}), (1, {"end": 13}), (1, {"start": 0}),
                (1, {"path": "outside.cu"}), (1, {"quote": "invented"}),
            ):
                with (
                    self.subTest(status=status, count=count, change=change),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    report = {"checks": [{
                        "feature": feature.to_dict(), "status": status,
                        "reason": "Inspect the computation at this locus.",
                        "evidence": [{"path": "kernel.cu", "start": 1, "end": 1, **change}] * count,
                    }]}
                    sandbox = FakeSandbox(FakeRunner(Path(temporary), [report]))
                    with self.assertRaises(StructuredOutputError):
                        _observe(sandbox, kernel(source=source), (feature,))

    def test_missing_check_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = FakeSandbox(FakeRunner(Path(temporary), [{"checks": []}]))
            with self.assertRaises(StructuredOutputError):
                _observe(sandbox, kernel(), (Feature("main", "mma"),))

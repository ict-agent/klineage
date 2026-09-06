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
            "feature": feature.to_dict(), "status": "present", "evidence": [{
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
            {"feature": present.to_dict(), "status": "present", "evidence": [{
                "path": "kernel.cu", "start": 1, "end": 1, "quote": CUDA_SOURCE,
            }]},
            {"feature": absent.to_dict(), "status": "absent", "evidence": []},
        ]}
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = FakeSandbox(FakeRunner(Path(temporary), [report]))
            result = _observe(sandbox, kernel(validation=accepted()), (present, absent))

        self.assertEqual(result.features, (present,))
        self.assertIn("feature_verifier", result.validation.details)

    def test_missing_check_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = FakeSandbox(FakeRunner(Path(temporary), [{"checks": []}]))
            with self.assertRaises(StructuredOutputError):
                _observe(sandbox, kernel(), (Feature("main", "mma"),))

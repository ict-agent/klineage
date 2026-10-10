"""Per-workload evidence is preserved without averaging different shapes."""

from klineage.serialization import from_json
import json
import unittest

from klineage.artifact import ValidationResult
from klineage.harness.logging import measurement_fields
from klineage.serialization import to_json


class WorkloadValidationTests(unittest.TestCase):
    def test_report_preserves_cases(self):
        results = {
            "small": ValidationResult(True, True, True, 1.0, 2.0),
            "large": ValidationResult(True, True, True, 100.0, 150.0),
        }
        report = ValidationResult.combine(results)
        self.assertTrue(report.accepted)
        self.assertIsNone(report.latency_ms)
        self.assertIsNone(report.reference_latency_ms)
        self.assertEqual(report.workloads, results)
        self.assertEqual(from_json(to_json(report), ValidationResult), report)
        results.clear()
        self.assertEqual(tuple(report.workloads), ("small", "large"))
        with self.assertRaises(TypeError):
            report.workloads["new"] = ValidationResult(False, False, False)

    def test_single_case_keeps_latency(self):
        measured = ValidationResult(True, True, True, 1.0, 2.0)
        report = ValidationResult.combine({"only": measured})
        self.assertEqual(report.latency_ms, measured.latency_ms)
        self.assertEqual(report.reference_latency_ms, measured.reference_latency_ms)
        self.assertEqual(report.workloads["only"], measured)

    def test_failed_case_keeps_others(self):
        report = ValidationResult.combine({
            "first": ValidationResult(True, True, True, 1.0),
            "failed": ValidationResult(True, False, False),
            "last": ValidationResult(True, True, True, 3.0),
        })
        self.assertTrue(report.compile_passed)
        self.assertFalse(report.correctness_passed)
        self.assertFalse(report.accepted)
        self.assertEqual(report.workloads["last"].latency_ms, 3.0)

    def test_empty_report_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "workload"):
            ValidationResult.combine({})

    def test_inconsistent_summary(self):
        with self.assertRaisesRegex(ValueError, "summary"):
            ValidationResult(
                True, True, True,
                workloads={"failed": ValidationResult(False, False, False)},
            )

    def test_no_mixed_shape_latency(self):
        with self.assertRaisesRegex(ValueError, "latency"):
            ValidationResult(
                True, True, True, 50.0,
                workloads={
                    "small": ValidationResult(True, True, True, 1.0),
                    "large": ValidationResult(True, True, True, 100.0),
                },
            )

    def test_nested_report_rejected(self):
        child = ValidationResult.combine({
            "one": ValidationResult(True, True, True, 1.0),
        })
        with self.assertRaisesRegex(ValueError, "nested"):
            ValidationResult.combine({"nested": child})

    def test_timeline_keeps_cases(self):
        cases = {
            "small": ValidationResult(True, True, True, 1.0),
            "large": ValidationResult(True, True, True, 100.0),
        }
        fields = measurement_fields(ValidationResult.combine(cases))
        self.assertEqual(fields.get("workloads"), cases)
        self.assertEqual(json.loads(to_json(fields))["workloads"]["small"]["latency_ms"], 1.0)
        self.assertNotIn("latency_ms", fields)


if __name__ == "__main__":
    unittest.main()

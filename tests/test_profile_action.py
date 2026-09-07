import unittest
from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import Mock, patch

from test_action import accepted, kernel

from klineage.action.profile import _profile, profile
from klineage.kernel import Kernel
from klineage.profiling import KernelProfile, ProfileOptions


def measured(current):
    target = current.context.to_dict()
    target.pop("prior_actions")
    return KernelProfile(
        kernel_fingerprint=current.fingerprint, target=target,
        options=ProfileOptions(), tool="ncu", tool_version="2025.3.1",
        device={"uuid": "GPU-test", "name": "test", "capability": "sm120"},
        metrics=({"kernel": "gemm", "launch_id": "0", "section": "SpeedOfLight",
                  "metric": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                  "unit": "%", "value": "45.0"},),
        report_path="/reports/target.ncu-rep", raw_path="/reports/target.csv",
        collected_at="2026-09-07T00:00:00+00:00",
    )


class ProfileActionTests(unittest.TestCase):
    def test_profile_roundtrip(self):
        current = kernel(validation=accepted())
        result = replace(current, profile=measured(current))
        self.assertEqual(Kernel.from_dict(result.to_dict()), result)
        self.assertEqual(result.fingerprint, current.fingerprint)
        self.assertEqual(result.validation, current.validation)
        self.assertEqual(result._prompt_input()["profile"], result.profile.to_dict())

    def test_changed_state_drops_profile(self):
        current = kernel()
        current = replace(current, profile=measured(current))
        self.assertIsNone(replace(current, source=current.source + " // edit").profile)
        self.assertIsNone(replace(current, context=replace(
            current.context, platform="sm90",
        )).profile)
        self.assertIsNotNone(replace(current, context=current.context.with_actions(
            ("mma",),
        )).profile)

    def test_refresh_and_reuse(self):
        current = kernel(validation=accepted())
        evidence = measured(current)
        data = evidence.to_dict()
        for key in ("kernel_fingerprint", "target", "options"):
            data.pop(key)
        sandbox = Mock()
        sandbox._profile.return_value = data
        current = replace(current, profile=evidence)
        self.assertIs(_profile(sandbox, current), current)
        sandbox._profile.assert_not_called()
        with patch("klineage.action.profile._next", return_value=nullcontext(sandbox)):
            result = profile(current)
        sandbox._profile.assert_called_once_with(current, ProfileOptions())
        self.assertEqual(result.profile, evidence)
        self.assertEqual(result.validation, current.validation)

    def test_empty_profile_fails(self):
        with self.assertRaisesRegex(ValueError, "metrics"):
            replace(measured(kernel()), metrics=())

    def test_invalid_options_fail(self):
        for options in ({"set": "unknown"}, {"timeout_seconds": 0},
                        {"sections": ("--export",)}, {"kernel_filter": ""}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ProfileOptions(**options)

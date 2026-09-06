import importlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from test_action import CUDA_SOURCE, FakeRunner, FakeSandbox, kernel

from klineage.contract import KernelABI, ProblemSpec
from klineage.errors import ValidationGateError
from klineage.harness.codex_runner import CodexRunnerError
from klineage.kernel import Kernel


class InitRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.runner = FakeRunner(Path(temporary.name), [
            {"_write_source": CUDA_SOURCE, "done": True},
        ])
        self.sandbox = FakeSandbox(self.runner)
        self.expert = replace(kernel(), problem=ProblemSpec("gemm", "GEMM"), abi=KernelABI())
        self.module = importlib.import_module("klineage.action.init")
        self.trace = self.runner.state_dir / "timeout.jsonl"

    def checkpoint(self):
        return json.loads((self.sandbox._logs / "checkpoint-01.json").read_text())

    def test_retry_keeps_candidate(self):
        failure = CodexRunnerError("timed out", trace_path=self.trace)
        with patch.object(self.module, "verify_mechanism", side_effect=[
            failure, {"passed": True},
        ]) as audit:
            result = self.module._generate(self.sandbox, self.expert)

        self.assertTrue(result.validation.accepted)
        self.assertEqual(len(self.runner.prompts), 1)
        self.assertEqual(len(self.sandbox._evaluator.calls), 3)
        self.assertEqual(audit.call_count, 2)
        self.assertEqual(audit.call_args_list[0].args[2].fingerprint,
                         audit.call_args_list[1].args[2].fingerprint)
        errors = result.validation.details["mechanism_verifier"]["attempts"]
        self.assertEqual(errors[0]["trace_path"], str(self.trace))
        self.assertTrue(self.checkpoint()["candidate"]["validation"]["profile_passed"])

    def test_audit_has_checkpoint(self):
        def audit(_ask, expert, candidate, _repo):
            checkpoint = self.checkpoint()
            restored = Kernel.from_dict(checkpoint["candidate"])
            self.assertFalse(restored.validation.accepted)
            self.assertEqual(restored.fingerprint, candidate.fingerprint)
            self.assertEqual(Kernel.from_dict(checkpoint["expert"]).fingerprint,
                             expert.fingerprint)
            self.assertTrue(checkpoint["selection"]["profile_passed"])
            self.assertEqual(restored.artifact_path.read_text(), candidate.source)
            return {"passed": True}

        with patch.object(self.module, "verify_mechanism", side_effect=audit):
            self.assertTrue(self.module._generate(self.sandbox, self.expert).validation.accepted)

    def test_retry_limit_fails_closed(self):
        with (
            patch.object(self.module, "verify_mechanism", side_effect=CodexRunnerError(
                "timed out", trace_path=self.trace,
            )) as audit,
            self.assertRaises(ValidationGateError) as raised,
        ):
            self.module._generate(self.sandbox, self.expert)

        self.assertEqual(audit.call_count, self.module._AUDIT_ATTEMPTS)
        self.assertEqual(len(self.runner.prompts), 1)
        self.assertEqual(len(self.sandbox._evaluator.calls), 2)
        candidate = raised.exception.kernel
        self.assertFalse(candidate.validation.accepted)
        self.assertEqual(candidate.source, CUDA_SOURCE)
        checkpoint = self.checkpoint()
        self.assertEqual(checkpoint["candidate"], candidate.to_dict())
        self.assertTrue(checkpoint["selection"]["profile_passed"])
        attempts = candidate.validation.details["mechanism_verifier"]["attempts"]
        self.assertEqual(len(attempts), self.module._AUDIT_ATTEMPTS)
        self.assertTrue(all(item["error_type"] == "CodexRunnerError" for item in attempts))
        self.assertTrue((self.sandbox._logs / "fidelity-01.json").is_file())

    def test_confirmation_is_pending(self):
        evaluate = self.sandbox._evaluate

        def check(candidate, *, reference=None):
            if len(self.sandbox._evaluator.calls) == 2:
                checkpoint = self.checkpoint()
                self.assertFalse(checkpoint["candidate"]["validation"]["profile_passed"])
                self.assertTrue(checkpoint["candidate"]["validation"]["details"]
                                ["mechanism_verifier"]["passed"])
            return evaluate(candidate, reference=reference)

        with (
            patch.object(self.module, "verify_mechanism", return_value={"passed": True}),
            patch.object(self.sandbox, "_evaluate", side_effect=check),
        ):
            self.module._generate(self.sandbox, self.expert)

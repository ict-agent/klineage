import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from klineage.action import (
    Action,
    Apply,
    Decompose,
    Init,
    Verify,
    verify,
)
from klineage.errors import StructuredOutputError, ValidationGateError


class ActionBaseTests(unittest.TestCase):
    def setUp(self):
        self.workdir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.runner = Mock(return_value=SimpleNamespace(final_message="done"))
        self.factory = self.enterContext(
            patch(
                "klineage.action.action.CodexRunner",
                return_value=self.runner,
            )
        )

    def test_run_retains_response(self):
        action = Action("generate files", self.workdir)
        with patch.object(action, "verify", return_value=True) as check:
            self.assertIsNone(action.run())
        self.runner.assert_called_once()
        self.assertEqual(self.runner.call_args.args, ("generate files",))
        self.assertEqual(action.response, "done")
        check.assert_called_once_with(action.verify_prompt)

    def test_timeout_reaches_runner(self):
        action = Action("generate", self.workdir, timeout=42)
        self.assertEqual(self.factory.call_args.kwargs["timeout"], 42)
        self.runner.return_value.final_message = "true"
        self.assertTrue(action.verify("Inspect files"))
        self.assertEqual(self.factory.call_args.kwargs["timeout"], 42)

    def test_verification_is_boolean(self):
        for raw, expected in (("true", True), ("false", False)):
            self.runner.return_value.final_message = raw
            self.assertIs(verify("inspect files", self.workdir), expected)
        for raw in ('{"passed": true}', "1", "True", "```true```", "unknown", None):
            self.runner.return_value.final_message = raw
            with self.subTest(raw=raw), self.assertRaises(StructuredOutputError):
                verify("inspect files", self.workdir)

    def test_verify_does_not_recurse(self):
        self.runner.return_value.final_message = "false"
        action = Verify("inspect files", self.workdir)
        with patch.object(action, "verify") as check:
            action.run()
        self.assertFalse(action.passed)
        self.assertFalse(action.enable_verifier)
        check.assert_not_called()
        self.runner.assert_called_once()

    def test_action_specific_criteria(self):
        from kernel_fixtures import kernel, skill

        current, cards = kernel(), (skill("tile"),)
        initial = Init.__new__(Init)
        Action.__init__(initial, "init", self.workdir)
        actions = (
            initial,
            Decompose(current, workdir=self.workdir),
            Apply(current, cards[0], workdir=self.workdir),
        )
        self.assertEqual(
            len({action.verify_prompt for action in actions}), len(actions)
        )
        with patch("klineage.action.verify.verify", return_value=True) as check:
            for action in actions:
                with self.subTest(action=type(action).__name__):
                    self.assertTrue(action.verify(action.verify_prompt))
                    prompt, directory = check.call_args.args
                    self.assertTrue(prompt.startswith(action.verify_prompt))
                    self.assertIn("prompt.txt", prompt)
                    self.assertEqual(directory, self.workdir)

                    for args, kwargs in (
                        (("Custom verification",), {}),
                        ((), {"prompt": "Custom verification"}),
                    ):
                        self.assertTrue(action.verify(*args, **kwargs))
                        prompt, directory = check.call_args.args
                        self.assertTrue(prompt.startswith("Custom verification\n"))

                    with self.assertRaises(TypeError):
                        action.verify(None)


class ActionRunTests(unittest.TestCase):
    def setUp(self):
        self.work = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.runner = Mock(return_value=SimpleNamespace(final_message="generated"))
        self.enterContext(
            patch("klineage.action.action.CodexRunner", return_value=self.runner)
        )

    def test_retries_failed_verification(self):
        action = Action("generate", self.work, max_retries=1)
        with patch.object(action, "verify", side_effect=(False, True)):
            action.run()
        self.assertEqual(self.runner.call_count, 2)
        self.assertEqual(action.attempt, 2)
        self.assertIn("ValidationGateError", self.runner.call_args.args[0])

    def test_retries_execution_failure(self):
        self.runner.side_effect = (
            RuntimeError("temporary failure"),
            SimpleNamespace(final_message="fixed"),
        )
        action = Action("generate", self.work, max_retries=1)
        with patch.object(action, "verify", return_value=True) as check:
            action.run()
        self.assertEqual(action.response, "fixed")
        self.assertIn("temporary failure", self.runner.call_args.args[0])
        check.assert_called_once()

    def test_retry_verifies_its_record(self):
        self.runner.side_effect = [
            SimpleNamespace(final_message=text)
            for text in ("first", "false", "second", "true")
        ]
        action = Action("generate", self.work, max_retries=1)
        action.run()
        first, check_first, second, check_second = self.runner.call_args_list
        for generation, check in ((first, check_first), (second, check_second)):
            record = f".klineage/codex-runs/{generation.kwargs['run_id']}"
            self.assertIn(f'Generation record: "{record}"', check.args[0])
        self.assertNotIn(first.kwargs["run_id"], check_second.args[0])

    def test_retry_budget_is_bounded(self):
        action = Action("generate", self.work, max_retries=2)
        with (
            patch.object(action, "verify", return_value=False),
            self.assertRaises(ValidationGateError),
        ):
            action.run()
        self.assertEqual(self.runner.call_count, 3)
        identifiers = [call.kwargs["run_id"] for call in self.runner.call_args_list]
        self.assertEqual(len(set(identifiers)), 3)

    def test_repeated_verify_keeps_logs(self):
        from klineage.action import verify

        def record(prompt, *, run_id):
            directory = self.work / run_id
            directory.mkdir()
            (directory / "prompt.txt").write_text(prompt)
            return SimpleNamespace(final_message="true")

        self.runner.side_effect = record
        self.assertTrue(verify("first check", self.work))
        self.assertTrue(verify("second check", self.work))
        self.assertEqual(len(list(self.work.glob("*/prompt.txt"))), 2)

    def test_zero_means_one_attempt(self):
        action = Action("generate", self.work, max_retries=0)
        with (
            patch.object(action, "verify", return_value=False),
            self.assertRaises(ValidationGateError),
        ):
            action.run()
        self.runner.assert_called_once()

    def test_nonboolean_is_rejected(self):
        for value in ("true", 1, None):
            action = Action("generate", self.work, max_retries=0)
            with (
                self.subTest(value=value),
                patch.object(action, "verify", return_value=value),
                self.assertRaises(StructuredOutputError),
            ):
                action.run()

    def test_interrupt_is_not_retried(self):
        for error in (KeyboardInterrupt(), SystemExit()):
            action = Action("generate", self.work, max_retries=3)
            self.runner.reset_mock()
            self.runner.side_effect = error
            with self.subTest(error=error), self.assertRaises(type(error)):
                action.run()
            self.runner.assert_called_once()
            self.assertEqual(self.runner.call_args.args, (action.prompt,))
            self.assertTrue(
                self.runner.call_args.kwargs["run_id"].startswith("Action-")
            )

    def test_disabled_runs_once(self):
        action = Action("generate", self.work, enable_verifier=False)
        with patch.object(action, "verify") as check:
            action.run()
        self.runner.assert_called_once()
        check.assert_not_called()

    def test_disabled_error_propagates(self):
        action = Action("generate", self.work, enable_verifier=False)
        self.runner.side_effect = RuntimeError("failed")
        with patch.object(action, "verify") as check, self.assertRaises(RuntimeError):
            action.run()
        self.runner.assert_called_once()
        check.assert_not_called()
        self.assertIsNone(action.response)

    def test_invalid_budget_rejected(self):
        for value in (-1, True, 1.5, "2"):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                Action("generate", self.work, max_retries=value)


class ImportTests(unittest.TestCase):
    def test_import_without_gpu_libs(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
class NoGpuImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_npu', 'flashinfer', 'cupti'}:
            raise AssertionError('unexpected GPU dependency: ' + fullname)
sys.meta_path.insert(0, NoGpuImports())
from klineage.action import Action, Verify, init_memory, workflow
from klineage.agent_api import function_docs
from klineage.agent_tools import profile, retrieve
from klineage.harness import evaluate
from klineage.harness.artifacts import BundleLoader
from klineage.kernel import Kernel
assert 'klineage.agent_tools.profile' in function_docs()
""",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )

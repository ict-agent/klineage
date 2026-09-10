"""Evaluation worker transport and failure evidence."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from problem_fixtures import problem_spec

from klineage.artifact.kernel import Kernel
from klineage.harness.eval import EvaluationRuntime, ValidationResult, WorkerError
from klineage.harness.timing import TimingPolicy


class EvalTransportTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.problem = self.root / "reference.py"
        self.problem.touch()
        self.logs = self.root / "logs"
        self.logs.mkdir()
        self.runtime = EvaluationRuntime()

    def test_interrupt_stops_worker(self):
        from klineage.harness.process import run_process

        for error in (KeyboardInterrupt(), SystemExit()):
            process = Mock()
            process.communicate.side_effect = (error, ("", ""))
            with (
                self.subTest(error=type(error).__name__),
                patch(
                    "klineage.harness.process.subprocess.Popen", return_value=process
                ),
                patch("klineage.harness.process.stop_process") as stop,
                self.assertRaises(type(error)),
            ):
                run_process(("worker",), payload="", timeout=1, environment={})
            stop.assert_called_once_with(process)

    def test_transport_keeps_paths(self):
        artifact = self.root / "kernel.cu"
        artifact.write_text("__global__ void kernel() {}")
        validation = ValidationResult(True, True, True, 1.0, reference_latency_ms=2.0)
        kernel = Kernel(
            "kernel",
            problem_spec("gemm", "cuda", "sm90"),
            source_files={"kernel.cu": artifact.read_text()},
            validation=validation,
        )
        with patch.object(
            self.runtime, "invoke", return_value=validation.to_dict()
        ) as invoke:
            self.assertEqual(
                self.runtime.evaluate(
                    kernel,
                    reference=kernel,
                    include_paths=(),
                    build_root=self.root / "build",
                    log_dir=self.logs,
                ),
                validation,
            )
        request = invoke.call_args.args[0]
        for key in ("kernel", "reference"):
            self.assertNotIn("validation", request[key])
            self.assertEqual(
                Kernel.from_dict(request[key]).source_files,
                {"kernel.cu": artifact.read_text()},
            )
        self.assertNotIn("problem_path", request["config"])
        self.assertEqual(request["config"]["timing"], TimingPolicy().to_dict())

    def test_launches_local_python(self):
        result = subprocess.CompletedProcess((), 0, '{"done": true}', "")
        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}),
            patch("klineage.harness.eval.run_process", return_value=result) as run,
        ):
            payload = self.runtime.invoke(
                {"operation": "inspect"}, self.logs, "inspect"
            )
        command = run.call_args.args[0]
        options = run.call_args.kwargs
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1], "-c")
        self.assertIn("from klineage.harness.eval import main", command[2])
        self.assertEqual(options["environment"]["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(json.loads(options["payload"]), {"operation": "inspect"})
        self.assertEqual(payload, {"done": True})
        self.assertTrue((self.logs / "inspect-process.json").is_file())
        record = json.loads((self.logs / "inspect-process.json").read_text())
        self.assertEqual(record["command"], list(command))

    def test_timeout_retains_logs(self):
        error = subprocess.TimeoutExpired(
            "worker", 1, output="partial", stderr="waiting"
        )
        with (
            patch("klineage.harness.eval.run_process", side_effect=error),
            self.assertRaisesRegex(WorkerError, "timed out"),
        ):
            self.runtime.invoke({}, self.logs, "timeout", timeout=1)
        self.assertEqual((self.logs / "timeout-stdout.log").read_text(), "partial")
        self.assertEqual((self.logs / "timeout-stderr.log").read_text(), "waiting")
        self.assertTrue(
            json.loads((self.logs / "timeout-process.json").read_text())["timed_out"]
        )

    def test_bad_worker_output_fails(self):
        for result in (
            subprocess.CompletedProcess((), 1, "", "failure"),
            subprocess.CompletedProcess((), 0, "not JSON", ""),
        ):
            with (
                self.subTest(result=result),
                patch("klineage.harness.eval.run_process", return_value=result),
                self.assertRaises(WorkerError),
            ):
                self.runtime.invoke({}, self.logs, "invalid")

    def test_inspection_reports_error(self):
        with (
            patch.object(self.runtime, "invoke", return_value={"error": "bad problem"}),
            self.assertRaisesRegex(WorkerError, "bad problem"),
        ):
            self.runtime.inspect(self.problem, log_dir=self.logs)


if __name__ == "__main__":
    unittest.main()

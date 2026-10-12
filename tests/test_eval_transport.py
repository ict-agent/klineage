"""Evaluation worker transport and failure evidence."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from kernel_fixtures import edit_kernel, make_kernel
from problem_fixtures import problem_spec

from klineage.artifact import json_node, load_problem, source_files
from klineage.artifact.kernel import next_version
from klineage.harness.eval import (
    EvaluationRuntime,
    ValidationResult,
    WorkerError,
    evaluate,
    inspect_problem,
)
from klineage.harness.timing import TimingPolicy
from klineage.serialization import raw_text, to_json


class EvalTransportTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.problem = self.root / "reference.py"
        self.problem.touch()
        self.logs = self.root / "logs"
        self.logs.mkdir()
        self.runtime = EvaluationRuntime()

    def test_buffered_native_output(self):
        # C stdio must flush before stdout returns to the JSON transport.
        source = '''
import ctypes, sys
from klineage.harness.eval import _descriptor_stdout, _emit
with _descriptor_stdout(sys.stderr):
    ctypes.CDLL(None).printf(b"native diagnostic\\n")
_emit({"done": True})
'''
        reply = subprocess.run(
            [sys.executable, "-c", source], capture_output=True, text=True,
            check=True, timeout=5,
        )
        self.assertEqual(json.loads(reply.stdout), {"done": True})
        self.assertIn("native diagnostic", reply.stderr)

    def test_shutdown_native_output(self):
        # CANN may write diagnostics after the worker emits its reply.
        source = '''
import atexit, ctypes
from klineage.harness.eval import _emit
libc = ctypes.CDLL(None)
atexit.register(lambda: libc.printf(b"runtime shutdown\\n"))
_emit({"done": True})
'''
        reply = subprocess.run(
            [sys.executable, "-c", source], capture_output=True, text=True,
            check=True, timeout=5,
        )
        self.assertEqual(json.loads(reply.stdout), {"done": True})
        self.assertIn("runtime shutdown", reply.stderr)

    def test_problem_loaded_once(self):
        problem = problem_spec()
        kernel = make_kernel("kernel", problem, {"kernel.cu": "// source"})
        validation = ValidationResult(True, True, True, 1.0)
        reply = subprocess.CompletedProcess((), 0, to_json(validation, indent=None), "")
        with (
            patch("klineage.harness.eval.load_problem", side_effect=(problem, AssertionError("reloaded problem"))),
            patch("klineage.harness.eval.run_process", return_value=reply),
        ):
            result = self.runtime.evaluate(
                kernel, include_paths=(), build_root=self.root / "build", log_dir=self.logs,
            )
        self.assertTrue(result.accepted)

    def test_success_stops_detached_child(self):
        from klineage.harness.process import run_process

        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(lambda: unrelated.poll() is None and unrelated.kill())
        pidfile = self.root / "child.pid"
        command = [sys.executable, "-c", """
import subprocess,sys,pathlib
child = subprocess.Popen(
    [sys.executable, '-c', 'import time; time.sleep(30)'],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
)
pathlib.Path(sys.argv[1]).write_text(str(child.pid))
print('result')
print('diagnostic', file=sys.stderr)
""", str(pidfile)]
        environment = {"PATH": os.environ["PATH"]}
        result = run_process(command, payload="", timeout=2, environment=environment)
        pid = int(pidfile.read_text())
        try:
            try:
                state = Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0]
            except (FileNotFoundError, ProcessLookupError):
                state = "exited"
            self.assertIn(state, ("Z", "exited"))
            self.assertIsNone(unrelated.poll())
            self.assertEqual(result.stdout, "result\n")
            self.assertEqual(result.stderr, "diagnostic\n")
            self.assertEqual(set(environment), {"PATH"})
        finally:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            unrelated.kill()
            unrelated.wait()

    def test_interrupt_stops_worker(self):
        from klineage.harness.process import run_process

        for error in (KeyboardInterrupt(), SystemExit()):
            process = Mock()
            process.communicate.side_effect = (error, ("", ""))
            with (
                self.subTest(error=type(error).__name__),
                patch(
                    "klineage.harness.process.subprocess.Popen", return_value=process
                ) as launch,
                patch("klineage.harness.process.stop_process") as stop,
                self.assertRaises(type(error)),
            ):
                run_process(("worker",), payload="", timeout=1, environment={})
            scope = launch.call_args.kwargs["env"]["KLINEAGE_PROCESS_SCOPE"]
            stop.assert_called_once_with(process, scope=scope)

    def test_transport_keeps_paths(self):
        artifact = self.root / "kernel.cu"
        artifact.write_text("__global__ void kernel() {}")
        validation = ValidationResult(True, True, True, 1.0, reference_latency_ms=2.0)
        kernel = make_kernel(
            "kernel",
            problem_spec("gemm", "cuda", "sm90"),
            source_files={"kernel.cu": artifact.read_text()},
            validation=validation,
        )
        kernel = edit_kernel(kernel, compile_flags=("-DNDEBUG",))
        with patch.object(
            self.runtime, "invoke", return_value=json.loads(to_json(validation))
        ) as invoke:
            self.assertEqual(
                self.runtime.evaluate(
                    kernel,
                    reference=kernel,
                    include_paths=(),
                    build_root=self.root / "build",
                    log_dir=self.logs,
                ),
                ValidationResult.combine({load_problem(kernel).workloads[0]["uuid"]: validation}),
            )
        request = invoke.call_args.args[0]
        for key in ("kernel", "reference"):
            self.assertEqual(
                tuple(
                    json.loads(
                        raw_text(request[key].find("metadata.json"))
                    ).get("compile_flags", ())
                ),
                ("-DNDEBUG",),
            )
            self.assertEqual(request[key].kind, "kernel.cuda")
            self.assertEqual(
                source_files(request[key].find("submission")),
                source_files(kernel.find("submission")),
            )
        self.assertNotIn("problem_path", request["config"])
        self.assertEqual(request["workload_id"], load_problem(kernel).workloads[0]["uuid"])
        self.assertEqual(request["config"]["timing"], json.loads(to_json(TimingPolicy())))
        self.assertEqual(request["config"]["correctness"], {"rtol": 1e-2, "atol": 1e-2})

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

    def test_worker_budget_override(self):
        kernel = make_kernel(
            "kernel",
            problem_spec("gemm", "cuda", "sm90"),
            source_files={"kernel.cu": "__global__ void kernel() {}"},
        )
        validation = ValidationResult(True, True, True, 1.0)
        extended = self.runtime.timeout * 2
        requests = []

        # Simulate a measurement longer than the default worker budget.
        def complete(command, **options):
            requests.append(json.loads(options["payload"]))
            if options["timeout"] <= self.runtime.timeout:
                raise subprocess.TimeoutExpired(command, options["timeout"])
            return subprocess.CompletedProcess(
                command, 0, to_json(validation, indent=None), ""
            )

        with patch("klineage.harness.eval.run_process", side_effect=complete) as run:
            self.assertFalse(evaluate(kernel, self.root).accepted)
            self.assertEqual(
                evaluate(kernel, self.root, timeout=extended),
                ValidationResult.combine({load_problem(kernel).workloads[0]["uuid"]: validation}),
            )

        self.assertLessEqual(run.call_args.kwargs["timeout"], extended)
        self.assertGreater(run.call_args.kwargs["timeout"], extended - 1)
        self.assertEqual(requests[0], requests[1])
        records = [
            json.loads(text)
            for path in (self.root / "versions").glob("*/metadata.json")
            for name, text in json.loads(path.read_text())["evaluation"]["diagnostics"].items()
            if name.endswith("eval-0001-process.json")
        ]
        completed = [record for record in records if not record["timed_out"]]
        self.assertEqual(len(completed), 1)
        self.assertGreaterEqual(completed[0]["timeout_seconds"], run.call_args.kwargs["timeout"])
        self.assertLess(completed[0]["timeout_seconds"] - run.call_args.kwargs["timeout"], 1)

    def test_evaluate_numbers_one_version_per_call(self):
        kernel = make_kernel(
            "kernel",
            problem_spec("gemm", "cuda", "sm90"),
            source_files={"kernel.cu": "__global__ void kernel() {}"},
        )
        validation = ValidationResult(True, True, True, 1.0)

        def complete(command, **options):
            return subprocess.CompletedProcess(
                command, 0, to_json(validation, indent=None), ""
            )

        with patch("klineage.harness.eval.run_process", side_effect=complete):
            for _ in range(3):
                evaluate(kernel, self.root)

        versions = self.root / "versions"
        self.assertEqual(sorted(p.name for p in versions.iterdir()), ["0", "1", "2"])
        for number in ("0", "1", "2"):
            from klineage.artifact import load_artifact

            saved = load_artifact(versions / number, artifact_type=type(kernel))
            self.assertEqual(
                json.loads(raw_text(saved.find("metadata.json")))["name"], "kernel"
            )
            self.assertEqual(source_files(saved.find("submission")), source_files(kernel.find("submission")))

    def test_timings_saved_in_metadata(self):
        kernel = make_kernel("test", problem_spec(), {"kernel.cu": "// source"})
        metadata = json.loads(raw_text(kernel.find("metadata.json")))
        metadata["evaluation"] = {"job_id": "previous", "error": None, "diagnostics": {}}
        kernel = type(kernel)(kernel.tree.with_child(json_node("metadata.json", metadata)))
        validation = ValidationResult.combine({
            load_problem(kernel).workloads[0]["uuid"]: ValidationResult(True, True, True, 2.5),
        })
        logs = []

        def measure(candidate, **options):
            initial = json.loads((self.root / "versions/0/metadata.json").read_text())
            self.assertIsNone(initial["validation"])
            self.assertNotIn("evaluation", initial)
            logs.append(options["log_dir"])
            (options["log_dir"] / "stderr.log").write_text("compiler note")
            return validation

        with patch.object(EvaluationRuntime, "evaluate", side_effect=measure):
            self.assertEqual(evaluate(kernel, self.root), validation)
        saved = json.loads((self.root / "versions/0/metadata.json").read_text())
        self.assertEqual(saved["validation"], json.loads(to_json(validation)))
        self.assertEqual(saved["evaluation"], {"error": None, "diagnostics": {"stderr.log": "compiler note"}})
        self.assertFalse(logs[0].exists())
        self.assertFalse((self.root / "evaluations").exists())

    def test_exception_keeps_diagnostics(self):
        kernel = make_kernel("test", problem_spec(), {"kernel.cu": "// source"})

        def fail(candidate, **options):
            (options["log_dir"] / "stderr.log").write_text("before failure")
            raise ValueError("runtime failed")

        with (
            patch.object(EvaluationRuntime, "evaluate", side_effect=fail),
            self.assertRaisesRegex(ValueError, "runtime failed"),
        ):
            evaluate(kernel, self.root)
        metadata = json.loads((self.root / "versions/0/metadata.json").read_text())
        self.assertIn("runtime failed", metadata["evaluation"]["error"])
        self.assertEqual(metadata["evaluation"]["diagnostics"], {"stderr.log": "before failure"})
        self.assertFalse(metadata["validation"]["compile_passed"])
        self.assertFalse((self.root / "evaluations").exists())

    def test_version_snapshot_survives_a_failed_evaluation(self):
        # The snapshot names what was measured, so it is taken even when the
        # measurement then fails; a missing snapshot is evidence lost.
        kernel = make_kernel(
            "kernel",
            problem_spec("gemm", "cuda", "sm90"),
            source_files={"kernel.cu": "__global__ void kernel() {}"},
        )
        with patch("klineage.harness.eval.run_process", side_effect=OSError("no worker")):
            self.assertFalse(evaluate(kernel, self.root).accepted)

        saved = self.root / "versions" / "0" / "metadata.json"
        self.assertTrue(saved.is_file())

    def test_next_version_ignores_gaps_and_foreign_directories(self):
        versions = self.root / "versions"
        (versions / "1").mkdir(parents=True)
        (versions / "3").mkdir()
        (versions / "notes").mkdir()
        (versions / "7.txt").write_text("not a version")

        # Highest wins, so a gap is never reused and stray entries do not number.
        self.assertEqual(next_version(self.root), 4)

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

    def test_inspect_keeps_fixed_inputs(self):
        logs = []
        problem = problem_spec()

        def inspect(path, *, log_dir):
            logs.append(log_dir)
            captured = log_dir / "inputs.safetensors"
            captured.write_bytes(b"fixed input")
            workloads = problem.workloads
            workloads[0]["inputs"]["a"] = {
                "type": "safetensors", "path": str(captured), "tensor_key": "a",
            }
            return {
                "problem_name": problem.name, "definition": problem.definition,
                "workloads": workloads, "language": problem.language, "platform": problem.platform,
            }

        with patch.object(EvaluationRuntime, "inspect", side_effect=inspect):
            result = inspect_problem(self.problem, self.root)
        fixed = Path(result.resolve_inputs()[0]["inputs"]["a"]["path"])
        self.assertEqual(fixed.read_bytes(), b"fixed input")
        self.assertEqual(fixed.parent, self.root / "inputs")
        self.assertFalse(logs[0].exists())
        self.assertFalse((self.root / "evaluations").exists())


if __name__ == "__main__":
    unittest.main()

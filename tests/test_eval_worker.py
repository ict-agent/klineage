from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kernel_fixtures import edit_kernel, make_kernel
from problem_fixtures import io_problem, problem_spec

from klineage.artifact import Problem, load_problem, source_files
from klineage.harness import eval as worker
from klineage.harness.eval import EvaluationRuntime, ValidationResult, WorkerError
from klineage.harness.timing import TimingPolicy
from klineage.serialization import raw_text, to_json

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from klineage.backend import runtime as tensor
    from klineage.backend.runtime import OutputStyle
    from klineage.harness import eval as worker


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable")
class EvalWorkerTests(unittest.TestCase):
    def setUp(self):
        # Exercise tensor contracts on CPU; backend tests cover device selection.
        runtime = SimpleNamespace(device=lambda device: nullcontext())
        self.enterContext(
            patch("klineage.backend.runtime.device_runtime", return_value=runtime)
        )
        self.enterContext(patch("klineage.backend.runtime.current_device", return_value=torch.device("cpu")))

    def test_numeric_error_details(self):
        expected = torch.tensor([1.0, 2.0, 3.0])
        for value in (2.1, float("nan")):
            with self.subTest(value=value):
                actual = torch.tensor([1.0, value, 3.0])
                with self.assertRaises(AssertionError) as caught:
                    worker.check_close(actual, expected)
                message = str(caught.exception)
                self.assertIn("Mismatched elements:", message)
                self.assertIn("Greatest absolute difference:", message)
                self.assertIn("index (1,)", message)

    def test_bundle_destination_outputs(self):
        problem = io_problem(
            inputs={"x": {"dtype": "float32", "shape": [2]}},
            outputs={
                "sum": {"dtype": "float32", "shape": [2]},
                "product": {"dtype": "float32", "shape": [2]},
            },
        )
        calls = []

        def run(x, summed, product):
            calls.append((x, summed, product))
            summed.copy_(x + 1)
            product.copy_(x * 2)

        with (
            patch.object(
                tensor, "require_tensor", side_effect=lambda value, *args: value
            ),
            patch.object(torch.cuda, "device", return_value=nullcontext()),
        ):
            fn = tensor.BundleCallable(problem, run, OutputStyle.DESTINATION)
            first = fn(torch.tensor([1.0, 2.0]))
            second = fn(torch.tensor([3.0, 4.0]))

        self.assertEqual(len(calls), 2)
        torch.testing.assert_close(first[0], torch.tensor([2.0, 3.0]))
        torch.testing.assert_close(second[1], torch.tensor([6.0, 8.0]))

    def test_bundle_return_contract(self):
        problem = io_problem(
            inputs={"x": {"dtype": "float32", "shape": [2]}},
            outputs={"output": {"dtype": "float32", "shape": [2]}},
        )
        with (
            patch.object(
                tensor, "require_tensor", side_effect=lambda value, *args: value
            ),
            patch.object(torch.cuda, "device", return_value=nullcontext()),
        ):
            fn = tensor.BundleCallable(problem, lambda x: x + 1, OutputStyle.RETURN)
            torch.testing.assert_close(
                fn(torch.tensor([2.0, 3.0])), torch.tensor([3.0, 4.0])
            )
            with self.assertRaisesRegex(ValueError, "input count"):
                fn()
            with self.assertRaisesRegex(ValueError, "dtype"):
                fn(torch.tensor([2, 3]))
            bad = tensor.BundleCallable(problem, lambda x: x[:1], OutputStyle.RETURN)
            with self.assertRaisesRegex(ValueError, "shape"):
                bad(torch.tensor([2.0, 3.0]))

    def test_bundle_tuple_outputs(self):
        value = {"dtype": "float32", "shape": [2]}
        problem = io_problem(
            inputs={"x": value}, outputs={"x": value, "other": value}
        )
        with (
            patch.object(
                tensor, "require_tensor", side_effect=lambda value, *args: value
            ),
            patch.object(torch.cuda, "device", return_value=nullcontext()),
        ):
            for container in (tuple, list):
                with self.subTest(container=container):
                    fn = tensor.BundleCallable(
                        problem,
                        lambda x, container=container: container((x + 1, x + 2)),
                        OutputStyle.RETURN,
                    )
                    outputs = fn(torch.tensor([2.0, 3.0]))
                    self.assertIsInstance(outputs, tuple)
                    torch.testing.assert_close(outputs[0], torch.tensor([3.0, 4.0]))
                    torch.testing.assert_close(outputs[1], torch.tensor([4.0, 5.0]))

    def test_topk_allows_output_order(self):
        values = torch.tensor([[5.0, 1.0, 4.0, 3.0]])
        module = SimpleNamespace(
            OPERATOR="topk", run=lambda x: torch.topk(x, 2, sorted=False)
        )
        runtime = worker.problem_runtime(module, 0)
        with patch.object(worker, "load_inputs", return_value={"values": values}):
            inputs = runtime.make_inputs()
        expected = runtime.reference(*inputs.args)
        runtime.check_outputs(
            (expected.values.flip(-1), expected.indices.flip(-1)), expected
        )

    def test_topk_checks_indices(self):
        values = torch.tensor([[5.0, 5.0, 5.0, 0.0]])
        module = SimpleNamespace(
            OPERATOR="topk", run=lambda x: torch.topk(x, 2, sorted=False)
        )
        runtime = worker.problem_runtime(module, 0)
        with patch.object(worker, "load_inputs", return_value={"values": values}):
            inputs = runtime.make_inputs()
        expected = runtime.reference(*inputs.args)
        runtime.check_outputs(
            (torch.tensor([[5.0, 5.0]]), torch.tensor([[0, 1]])), expected
        )
        for indices in (
            torch.tensor([[0, 0]]),
            torch.tensor([[0, 3]]),
            torch.tensor([[-1, 0]]),
            torch.tensor([[0, 4]]),
            torch.tensor([[0, 1]], dtype=torch.int32),
        ):
            with (
                self.subTest(indices=indices),
                self.assertRaises((ValueError, AssertionError)),
            ):
                runtime.check_outputs((torch.tensor([[5.0, 5.0]]), indices), expected)


class Clock:
    def __init__(self):
        self.value = 100.0

    def monotonic(self):
        return self.value


class WorkloadEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.runtime = worker.EvaluationRuntime(timeout=10)
        self.workloads = (
            {"uuid": "small", "axes": {"M": 2}, "inputs": {}},
            {"uuid": "large", "axes": {"M": 8}, "inputs": {}},
        )
        self.problem = SimpleNamespace(workloads=self.workloads, language="cuda", platform="sm120")
        self.kernel = {"artifact_type": "test"}
        backend = SimpleNamespace(timing=SimpleNamespace(DEFAULT_POLICY=TimingPolicy()))
        self.enterContext(
            patch.object(worker, "load_problem", return_value=self.problem)
        )
        self.enterContext(patch.object(worker, "get_backend", return_value=backend))

    def evaluate(self, **options):
        return self.runtime.evaluate(
            self.kernel,
            include_paths=(),
            build_root=self.root / "build",
            log_dir=self.root / "logs",
            **options,
        )

    def complete(self, command, **options):
        return subprocess.CompletedProcess(
            command,
            0,
            to_json(worker.ValidationResult(True, True, True, 1.0), indent=None),
            "",
        )

    def test_runs_each_workload(self):
        with patch.object(worker, "run_process", side_effect=self.complete) as run:
            result = self.evaluate()

        self.assertEqual(run.call_count, 2)
        requests = [json.loads(call.kwargs["payload"]) for call in run.call_args_list]
        self.assertEqual(
            [request["workload_id"] for request in requests], ["small", "large"]
        )
        self.assertNotEqual(
            requests[0]["config"]["build_root"], requests[1]["config"]["build_root"]
        )
        self.assertEqual(tuple(result.workloads), ("small", "large"))
        self.assertTrue(result.accepted)
        self.assertIsNone(result.latency_ms)
        self.assertEqual(
            len(list((self.root / "logs").rglob("eval-0001-result.json"))), 2
        )

    def test_crash_keeps_results(self):
        replies = (
            subprocess.CompletedProcess((), 1, "", "worker crashed"),
            self.complete(()),
        )
        with patch.object(worker, "run_process", side_effect=replies):
            result = self.evaluate()

        self.assertFalse(result.accepted)
        self.assertFalse(result.workloads["small"].accepted)
        self.assertTrue(result.workloads["large"].accepted)
        self.assertTrue(
            any(
                "worker crashed" in path.read_text()
                for path in (self.root / "logs").rglob("*.log")
            )
        )

    def test_workloads_share_deadline(self):
        self.problem.workloads = (
            *self.workloads,
            {"uuid": "last", "axes": {}, "inputs": {}},
        )
        clock = Clock()
        backend = worker.get_backend(self.problem.language, self.problem.platform)

        def prepare(language, platform):
            clock.value += 1
            return backend

        def complete(command, **options):
            clock.value += 5
            return self.complete(command, **options)

        with (
            patch.object(worker, "time", clock),
            patch.object(worker, "get_backend", side_effect=prepare),
            patch.object(worker, "run_process", side_effect=complete) as run,
        ):
            result = self.evaluate()

        self.assertEqual(
            [call.kwargs["timeout"] for call in run.call_args_list], [9.0, 4.0]
        )
        self.assertTrue(result.workloads["small"].accepted)
        self.assertFalse(result.workloads["large"].accepted)
        self.assertFalse(result.workloads["last"].accepted)
        self.assertTrue(
            any(
                "deadline" in path.read_text()
                for path in (self.root / "logs").rglob("*.log")
            )
        )

    def test_no_workloads_cannot_pass(self):
        self.problem.workloads = ()
        with patch.object(worker, "run_process", side_effect=self.complete) as run:
            result = self.evaluate()

        run.assert_not_called()
        self.assertFalse(result.accepted)
        self.assertEqual(dict(result.workloads), {})
        self.assertTrue(
            any(
                "no workloads" in path.read_text()
                for path in (self.root / "logs").rglob("*.log")
            )
        )


CPU_WORKER = """
from contextlib import nullcontext
from dataclasses import asdict
from types import SimpleNamespace
import torch
from klineage.backend.cuda import CudaBackend
from klineage.backend import runtime
from klineage.backend.timing import TimingResult
from klineage.harness import eval as worker
CudaBackend.device_type = 'cpu'
runtime.device_runtime = worker.device_runtime = lambda backend: SimpleNamespace(device=lambda device: nullcontext())
runtime.current_device = worker.current_device = lambda backend: torch.device('cpu')
def bench(fn, *, args, policy):
    fn(*args)
    samples = tuple((1.0,) * policy.repeat for _ in range(policy.trials))
    return TimingResult(samples, 1.0, 'cupti', {'policy': asdict(policy), 'cupti_version': '13.0'})
worker.bench = bench
raise SystemExit(worker.main())
"""


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
class WorkloadProcessTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.problem = Problem(
            "add",
            {
                "name": "add",
                "op_type": "add",
                "axes": {"N": {"type": "var"}},
                "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
                "outputs": {"output": {"shape": ["N"], "dtype": "float32"}},
                "reference": "def run(x): return x + 1\n",
            },
            workloads=tuple(
                {"uuid": name, "axes": {"N": size}, "inputs": {"x": {"type": "random"}}}
                for name, size in (("small", 2), ("large", 7))
            ),
            language="python",
            platform="sm120",
        )

    def evaluate(self, source):
        kernel = make_kernel(
            "add",
            self.problem,
            {
                "config.toml": '[solution]\nname="add"\ndefinition="add"\nauthor="test"\n'
                '[build]\nlanguage="python"\nentry_point="kernel.py::run"\n',
                "solution/kernel.py": source,
            },
        )
        runtime = worker.EvaluationRuntime(timeout=30)

        def launch(command, request, timeout, environment):
            return worker.run_process(
                (runtime.python, "-c", CPU_WORKER),
                payload=to_json(request, indent=None),
                timeout=timeout,
                environment=environment,
            )

        with patch.object(runtime, "launch", side_effect=launch):
            return runtime.evaluate(
                kernel,
                include_paths=(),
                build_root=self.root / "build",
                log_dir=self.root / "logs",
            )

    def test_workers_bind_shapes(self):
        result = self.evaluate("def run(x): return x + 1\n")
        self.assertTrue(result.accepted)
        self.assertEqual(tuple(result.workloads), ("small", "large"))
        self.assertTrue(all(value.accepted for value in result.workloads.values()))
        self.assertTrue(all(value.reference_latency_ms == 1.0 for value in result.workloads.values()))
        self.assertIsNone(result.reference_latency_ms)


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

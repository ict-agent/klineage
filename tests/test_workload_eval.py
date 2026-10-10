"""Every registered workload receives its own evaluation and evidence."""

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kernel_fixtures import make_kernel

from klineage.artifact import Problem
from klineage.backend import get_backend
from klineage.harness import eval as worker
from klineage.harness.timing import TimingPolicy
from klineage.serialization import to_json


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

    def test_evaluation_budget_cap(self):
        clock = Clock()
        for timeout in (3.0, 20.0):
            with (
                self.subTest(timeout=timeout),
                patch.object(worker, "time", clock),
                patch.object(worker, "run_process", side_effect=self.complete) as run,
            ):
                result = self.evaluate(timeout=timeout)
                self.assertTrue(result.accepted)
                self.assertEqual(
                    [call.kwargs["timeout"] for call in run.call_args_list],
                    [min(timeout, 10.0)] * 2,
                )
                self.assertEqual(self.runtime.timeout, 10)

    def test_invoke_counts_setup(self):
        clock = Clock()
        logs = self.root / "logs"
        logs.mkdir()
        write_json = worker.write_json

        def write_request(path, value):
            write_json(path, value)
            if path.name.endswith("-request.json"):
                clock.value += 3

        with (
            patch.object(worker, "time", clock),
            patch.object(worker, "write_json", side_effect=write_request),
            patch.object(worker, "run_process", side_effect=self.complete) as run,
        ):
            self.runtime.invoke({}, logs, "probe", timeout=10)
        self.assertEqual(run.call_args.kwargs["timeout"], 7)

    def test_setup_timeout_skips_run(self):
        clock = Clock()
        logs = self.root / "logs"
        logs.mkdir()
        write_json = worker.write_json

        def write_request(path, value):
            write_json(path, value)
            if path.name.endswith("-request.json"):
                clock.value += 11

        with (
            patch.object(worker, "time", clock),
            patch.object(worker, "write_json", side_effect=write_request),
            patch.object(worker, "run_process", side_effect=self.complete) as run,
        ):
            with self.assertRaisesRegex(worker.WorkerError, "timed out"):
                self.runtime.invoke({}, logs, "probe", timeout=10)
        run.assert_not_called()
        self.assertTrue(json.loads((logs / "probe-process.json").read_text())["timed_out"])

    def test_launch_counts_encoding(self):
        clock = Clock()

        def encode(request, **options):
            clock.value += 3
            return "{}"

        with (
            patch.object(worker, "time", clock),
            patch.object(worker, "to_json", side_effect=encode),
            patch.object(worker, "run_process") as run,
        ):
            self.runtime.launch(("worker",), {}, 10, {})
        self.assertEqual(run.call_args.kwargs["timeout"], 7)

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

    def test_worker_exit_continues(self):
        result = self.evaluate(
            "import os\ndef run(x):\n"
            "    if x.numel() == 2: os._exit(7)\n"
            "    return x + 1\n"
        )
        self.assertFalse(result.accepted)
        self.assertFalse(result.workloads["small"].accepted)
        self.assertTrue(result.workloads["large"].accepted)
        records = [
            json.loads(path.read_text())
            for path in (self.root / "logs").rglob("*-process.json")
        ]
        self.assertEqual(sorted(record["returncode"] for record in records), [0, 7])

    def test_inspection_keeps_cases(self):
        backend = get_backend("cuda")
        for workloads in ((), self.problem.workloads):
            module = worker.trace_module(self.problem.definition, workloads)
            with (
                self.subTest(count=len(workloads)),
                patch.object(
                    worker,
                    "load_problem_module",
                    return_value=(module, self.root / "add.json"),
                ),
                patch.object(worker, "current_device", return_value="cpu"),
                patch.object(
                    worker, "require_tensor", side_effect=lambda value, *args: value
                ),
                patch.object(type(backend), "info", return_value={"platform": "sm120"}),
            ):
                result = worker.inspect_reference(
                    self.root / "add.json",
                    workload_path=self.root / "captured.safetensors",
                    backend=backend,
                )
            self.assertEqual(result["workloads"], workloads)
            self.assertFalse((self.root / "captured.safetensors").exists())

    def test_inputs_require_case(self):
        module = worker.trace_module(self.problem.definition, self.problem.workloads)
        with self.assertRaisesRegex(ValueError, "selected workload"):
            worker.load_inputs(module, 0, get_backend("cuda"))


if __name__ == "__main__":
    unittest.main()

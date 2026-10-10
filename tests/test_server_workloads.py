"""The service accepts only reports covering its registered workloads."""

from klineage.serialization import from_json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kernel_fixtures import make_kernel
from problem_fixtures import problem_spec

from klineage.artifact import ValidationResult
from klineage.backend import get_backend
from klineage.serialization import to_json
from klineage.server import Service
from klineage.server.protocol import Evaluation, submission
from klineage.server.worker import WorkerRuntime, problem_paths


class ServerWorkloadTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        base = problem_spec()
        first = base.workloads[0] | {"uuid": "small"}
        second = first | {"uuid": "large", "axes": {"M": 256, "N": 256, "K": 256}}
        self.problem = base.replace(workloads=(first, second))

    def service(self, problem, result):
        return Service(
            self.root / "server", get_backend("cuda"), {"task": problem},
            runtime_factory=lambda **kwargs: SimpleNamespace(evaluate=lambda *args, **kwargs: result),
        )

    def evaluate(self, service, problem):
        kernel = make_kernel("test", problem, {"kernel.cu": "test"})
        return service.evaluate({"problem_id": "task", "submission": submission(kernel)})

    def test_missing_case_cannot_pass(self):
        result = ValidationResult.combine({"small": ValidationResult(True, True, True, 1.0)})
        reply = self.evaluate(self.service(self.problem, result), self.problem)
        self.assertFalse(reply.accepted)
        self.assertIn("workload", reply.error)
        self.assertEqual(set(reply.validation.workloads), {"small", "large"})

    def test_extra_case_cannot_pass(self):
        result = ValidationResult.combine({
            name: ValidationResult(True, True, True, 1.0)
            for name in ("small", "large", "unregistered")
        })
        reply = self.evaluate(self.service(self.problem, result), self.problem)
        self.assertFalse(reply.accepted)
        self.assertIn("workload", reply.error)

    def test_complete_report_roundtrip(self):
        result = ValidationResult.combine({
            "small": ValidationResult(True, True, True, 1.0),
            "large": ValidationResult(True, True, True, 9.0),
        })
        reply = self.evaluate(self.service(self.problem, result), self.problem)
        self.assertTrue(reply.accepted)
        self.assertEqual(from_json(to_json(reply), Evaluation).validation, result)
        self.assertIsNone(reply.validation.latency_ms)

    def test_definition_only_cannot_pass(self):
        problem = self.problem.replace(workloads=())
        optimistic = ValidationResult(True, True, True, 1.0)
        reply = self.evaluate(self.service(problem, optimistic), problem)
        self.assertFalse(reply.accepted)
        self.assertIn("workload", reply.error)

    def test_mount_all_directory_inputs(self):
        workloads = []
        paths = []
        for workload in self.problem.workloads:
            path = self.root / f'{workload["uuid"]}.safetensors'
            path.write_bytes(workload["uuid"].encode())
            paths.append(path)
            workload["inputs"]["a"] = {
                "type": "safetensors", "path": str(path), "tensor_key": "a",
            }
            workloads.append(workload)
        problem = self.problem.replace(workloads=workloads)
        kernel = make_kernel("test", problem)
        mounted = problem_paths(kernel, workload_id="small")
        self.assertIn(paths[0], mounted)
        self.assertIn(paths[1], mounted)
        with self.assertRaisesRegex(ValueError, "unknown workload"):
            problem_paths(kernel, workload_id="missing")

    def test_factory_uses_budget(self):
        clock = [0.0]
        report = ValidationResult.combine({
            case["uuid"]: ValidationResult(True, True, True, 1.0)
            for case in self.problem.workloads
        })
        evaluate = Mock(return_value=report)

        def factory(**kwargs):
            clock[0] += 3.0
            return SimpleNamespace(evaluate=evaluate)

        service = Service(
            self.root / "server", get_backend("cuda"), {"task": self.problem},
            timeout=5, runtime_factory=factory,
        )
        with patch("time.monotonic", side_effect=lambda: clock[0]):
            result = self.evaluate(service, self.problem)
        self.assertTrue(result.accepted)
        self.assertEqual(evaluate.call_args.kwargs.get("timeout"), 2.0)

    def test_late_report_not_accepted(self):
        clock = [0.0]
        report = ValidationResult.combine({
            case["uuid"]: ValidationResult(True, True, True, 1.0)
            for case in self.problem.workloads
        })

        def evaluate(*args, **kwargs):
            clock[0] = 6.0
            return report

        service = Service(
            self.root / "server", get_backend("cuda"), {"task": self.problem},
            timeout=5,
            runtime_factory=lambda **kwargs: SimpleNamespace(evaluate=evaluate),
        )
        with patch("time.monotonic", side_effect=lambda: clock[0]):
            result = self.evaluate(service, self.problem)
        self.assertFalse(result.accepted)
        self.assertIn("deadline", result.error)
        self.assertEqual(result.validation.workloads, report.workloads)

    def test_mounting_uses_budget(self):
        clock = [0.0]
        build = self.root / "build"
        build.mkdir()
        runtime = WorkerRuntime(python=sys.executable)

        def sandbox(*args, **kwargs):
            clock[0] += 3.0
            return SimpleNamespace(command=list, environment={})

        with (
            patch("time.monotonic", side_effect=lambda: clock[0]),
            patch("klineage.server.worker.Sandbox", side_effect=sandbox),
            patch("klineage.server.worker.run_process") as run,
        ):
            runtime.launch(
                [sys.executable],
                {"operation": "info", "config": {"build_root": str(build), "include_paths": []}},
                5.0, {},
            )
        self.assertEqual(run.call_args.kwargs["timeout"], 2.0)


if __name__ == "__main__":
    unittest.main()

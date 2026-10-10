"""Input descriptions remain readable without the evaluator's filesystem."""

from klineage.serialization import from_json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kernel_fixtures import make_kernel
from problem_fixtures import problem_spec

from klineage.artifact import Problem
from klineage.backend import get_backend
from klineage.serialization import to_json
from klineage.server import Service
from klineage.server.protocol import submission
from klineage.server.worker import problem_paths


class ProblemInputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "input.safetensors"
        self.path.write_bytes(b"fixed input")
        problem = problem_spec()
        workload = problem.workloads[0]
        workload["inputs"]["a"] = {
            "type": "safetensors", "path": str(self.path), "tensor_key": "a",
        }
        self.problem = problem.replace(workloads=(workload,))

    def test_contract_without_inputs(self):
        restored = from_json(to_json(self.problem), Problem)
        self.path.unlink()
        self.assertEqual(restored.workloads[0]["inputs"]["a"]["path"], str(self.path))
        self.assertEqual(restored.workloads[0]["axes"]["M"], 128)
        self.assertEqual(restored.values("inputs")["a"]["shape"], [128, 128])
        self.assertEqual(restored.values("outputs")["output"]["dtype"], "float16")

    def test_runtime_checks_input_digest(self):
        self.assertEqual(self.problem.resolve_inputs(), self.problem.workloads)
        self.path.write_bytes(b"changed input")
        self.assertEqual(self.problem.workloads[0]["axes"]["M"], 128)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.problem.resolve_inputs()

    def test_replace_keeps_input_capture(self):
        updated = self.problem.replace(platform="sm90")
        self.assertEqual(updated.resolve_inputs(), self.problem.resolve_inputs())
        self.path.unlink()
        with self.assertRaises(ValueError):
            self.problem.replace(platform="sm90")

    def test_replace_rejects_changed_input(self):
        self.path.write_bytes(b"changed input")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.problem.replace(platform="sm90")

    def test_replace_without_old_data(self):
        self.path.unlink()
        workloads = problem_spec().workloads
        updated = self.problem.replace(workloads=workloads)
        self.assertEqual(updated.resolve_inputs(), workloads)
        self.assertNotIn("inputs", [node.name for node in updated.tree.children])

    def test_registration_and_mount_check(self):
        kernel = make_kernel("test", self.problem)
        self.assertIn(self.path, problem_paths(kernel))
        self.path.write_bytes(b"changed input")
        with self.assertRaisesRegex(ValueError, "changed"):
            Service(
                self.root / "server", get_backend("cuda"), {"test": self.problem},
                runtime_factory=lambda **kwargs: None,
            )

    def test_eval_checks_before_oracle(self):
        kernel = make_kernel("test", self.problem, {"kernel.cu": "// source"})
        factory = self.enterContext(patch("klineage.server.worker.WorkerRuntime"))
        service = Service(
            self.root / "server", get_backend("cuda"), {"test": self.problem},
            runtime_factory=factory,
        )
        self.path.write_bytes(b"changed input")
        with patch(
            "klineage.harness.eval.trace_module", side_effect=AssertionError("oracle ran")
        ):
            with self.assertRaisesRegex(ValueError, "changed"):
                service.evaluate({"problem_id": "test", "submission": submission(kernel)})
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()

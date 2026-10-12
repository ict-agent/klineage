from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kernel_fixtures import make_kernel, measured
from problem_fixtures import problem_spec
from runtime_fixtures import FakeRuntime

from klineage.artifact import load_problem
from klineage.backend import get_backend
from klineage.server import Client, Service
from klineage.server import service as service_module
from klineage.server.protocol import submission


class ProblemCollectionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.backend = get_backend("cuda")
        self.enterContext(patch.object(type(self.backend), "info", return_value={"available": True, "platform": "sm120"}))

    def write(self, directory, name):
        directory.mkdir(parents=True)
        problem = problem_spec(name)
        (directory / "definition.json").write_text(json.dumps(problem.definition))
        (directory / "workloads.jsonl").write_text(json.dumps({
            "definition": name, "workload": problem.workloads[0],
        }) + "\n")
        return directory

    def test_single_and_collection(self):
        first = self.write(self.root / "group" / "one", "alpha")
        self.write(self.root / "group" / "two", "beta")
        (self.root / "group" / "README.md").write_text("notes")
        self.assertEqual(list(service_module.load_problems(first, backend=self.backend)), ["alpha"])
        self.assertEqual(list(service_module.load_problems(first.parent, backend=self.backend)), ["alpha", "beta"])

    def test_registration_keeps_directory(self):
        directory = self.write(self.root / "original", "alpha")
        problems = service_module.load_problems(directory, backend=self.backend)
        self.assertEqual(problems, {"alpha": directory})
        service = Service(self.root / "server", self.backend, problems, runtime_factory=FakeRuntime)
        self.assertEqual(service.problem_paths, {"alpha": directory})
        self.assertEqual(service.info()["problem_paths"], {"alpha": str(directory)})
        self.assertFalse(list(service.root.rglob("problem.json")))
        candidate = make_kernel("alpha", service.problem("alpha"), {"kernel.cu": "// source"})
        result = service.evaluate({"problem_id": "alpha", "submission": submission(candidate)})
        self.assertTrue(result.accepted)
        self.assertFalse((service.root / "jobs" / result.job_id / "candidate.json").exists())
        self.assertEqual(json.loads(result.kernel.find("metadata.json").content)["problem"], {
            "path": str(directory), **service.problem("alpha").target(),
        })

    def test_changes_during_eval_rejected(self):
        directory = self.write(self.root / "original", "alpha")
        service = Service(self.root / "server", self.backend, {"alpha": directory}, runtime_factory=FakeRuntime)
        candidate = make_kernel("alpha", service.problem("alpha"), {"kernel.cu": "// source"})

        def evaluate(kernel, **kwargs):
            validation = measured(kernel)
            definition = service.problem("alpha").definition
            definition["description"] = "changed during evaluation"
            (directory / "definition.json").write_text(json.dumps(definition))
            return validation

        with patch.object(FakeRuntime, "evaluate", side_effect=evaluate):
            result = service.evaluate({"problem_id": "alpha", "submission": submission(candidate)})
        self.assertFalse(result.accepted)
        self.assertIn("changed", result.error)

    def test_restart_checks_input_pins(self):
        directory = self.write(self.root / "original", "alpha")
        fixed = directory / "input.safetensors"
        fixed.write_bytes(b"before")
        case = problem_spec("alpha").workloads[0]
        case["inputs"]["a"] = {"type": "safetensors", "path": str(fixed), "tensor_key": "a"}
        (directory / "workloads.jsonl").write_text(json.dumps({"definition": "alpha", "workload": case}) + "\n")
        service = Service(self.root / "server", self.backend, {"alpha": directory}, runtime_factory=FakeRuntime)
        candidate = make_kernel("alpha", service.problem("alpha"), {"kernel.cu": "// source"})
        old = service.evaluate({"problem_id": "alpha", "submission": submission(candidate)})
        fixed.write_bytes(b"after ")
        restarted = Service(service.root, self.backend, {"alpha": directory}, runtime_factory=FakeRuntime)
        with self.assertRaisesRegex(ValueError, "registered problem"):
            restarted.result(old.job_id)

    def test_duplicate_names_rejected(self):
        self.write(self.root / "one", "same")
        self.write(self.root / "two", "same")
        with self.assertRaisesRegex(ValueError, "duplicate problem"):
            service_module.load_problems(self.root, backend=self.backend)

    def test_client_collection(self):
        problems = {name: problem_spec(name) for name in ("alpha", "beta")}
        service = Service(self.root / "server", self.backend, problems, runtime_factory=FakeRuntime)
        with service.start() as running, patch.dict(os.environ, {"KLINEAGE_PROBLEM": str(self.root / "mounted")}):
            client = Client(running.endpoint)
            self.assertIsNone(client.problem_id)
            with self.assertRaisesRegex(ValueError, "problem_id"):
                client.problem()
            self.assertEqual(client.problem("beta"), problems["beta"])
            work = self.root / "work"
            results = []
            for name, problem in problems.items():
                candidate = make_kernel(name, problem, {"kernel.cu": "// source"})
                results.append(client.evaluate(candidate, work, problem_id=name))
            for result in results:
                self.assertEqual(load_problem(result.kernel), problems[result.problem_id])
                self.assertTrue(client.verify(result.kernel, result.job_id).accepted)
            restored = client.result(results[0].job_id, work=work)
            self.assertEqual(load_problem(restored.kernel), problems["alpha"])


if __name__ == "__main__":
    unittest.main()

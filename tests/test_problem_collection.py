"""Problem directories and task selection keep independent source references."""

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from kernel_fixtures import make_kernel, measured
from problem_fixtures import problem_spec
from test_server import FakeRuntime

from klineage.artifact import Problem, load_problem
from klineage.backend import get_backend
from klineage.serialization import to_json
from klineage.server import Client, ClientGateway, ServerError, Service
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
        self.assertEqual(json.loads(result.kernel.node("metadata.json").content)["problem"], {
            "path": str(directory), **service.problem("alpha").target(),
        })

    def test_memory_problem_uses_tree(self):
        problem = problem_spec("alpha")
        service = Service(self.root / "server", self.backend, {"alpha": problem}, runtime_factory=FakeRuntime)
        directory = service.problem_paths["alpha"]
        self.assertTrue(directory.is_dir())
        self.assertEqual({path.name for path in directory.iterdir()}, {
            "definition.json", "workloads.jsonl", "target.json",
        })
        self.assertEqual(load_problem(directory), problem)

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

    def test_restart_rejects_changed_job(self):
        directory = self.write(self.root / "original", "alpha")
        service = Service(self.root / "server", self.backend, {"alpha": directory}, runtime_factory=FakeRuntime)
        candidate = make_kernel("alpha", service.problem("alpha"), {"kernel.cu": "// source"})
        request = {"problem_id": "alpha", "submission": submission(candidate)}
        old = service.evaluate(request)
        changed = problem_spec("alpha", size=64)
        (directory / "workloads.jsonl").write_text(changed.node("workloads.jsonl").content)

        restarted = Service(service.root, self.backend, {"alpha": directory}, runtime_factory=FakeRuntime)
        with self.assertRaisesRegex(ValueError, "registered problem"):
            restarted.result(old.job_id)
        with self.assertRaisesRegex(ValueError, "registered problem"):
            restarted.evaluate(request | {"reference": old.job_id})
        with restarted.start() as running:
            with self.assertRaisesRegex(ServerError, "registered problem"):
                Client(running.endpoint, "alpha").verify(old.kernel, old.job_id)

    def test_live_changes_reject_old_job(self):
        directory = self.write(self.root / "original", "alpha")
        service = Service(self.root / "server", self.backend, {"alpha": directory}, runtime_factory=FakeRuntime)
        candidate = make_kernel("alpha", service.problem("alpha"), {"kernel.cu": "// source"})
        old = service.evaluate({"problem_id": "alpha", "submission": submission(candidate)})
        definition = service.problem("alpha").definition
        definition["reference"] = "def run(a, b): return a + b\n"
        (directory / "definition.json").write_text(json.dumps(definition))
        with self.assertRaisesRegex(ValueError, "registered problem"):
            service.result(old.job_id)

    def test_remote_verify_rejects_change(self):
        directory = self.write(self.root / "original", "alpha")
        service = Service(self.root / "server", self.backend, {"alpha": directory}, runtime_factory=FakeRuntime)
        candidate = make_kernel("alpha", service.problem("alpha"), {"kernel.cu": "// source"})
        old = service.evaluate({"problem_id": "alpha", "submission": submission(candidate)})
        with service.start() as running, patch("klineage.server.client.Path") as local_path:
            local_path.return_value.exists.return_value = False
            client = Client(running.endpoint, "alpha")
            self.assertTrue(client.verify(old.kernel, old.job_id).accepted)
            with patch("klineage.server.client.load_problem", side_effect=AssertionError("client has no problem directory")):
                changed = problem_spec("alpha", size=64)
                (directory / "workloads.jsonl").write_text(changed.node("workloads.jsonl").content)
                with self.assertRaisesRegex(ServerError, "registered problem"):
                    client.verify(old.kernel, old.job_id)

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

    def test_digest_ignores_annotations(self):
        problem = problem_spec("alpha")
        annotated = Problem.from_tree("problem", replace(problem.tree, description="updated tree documentation"))
        digest = service_module.problem_digest(problem)
        self.assertEqual(service_module.problem_digest(annotated), digest)
        definition = problem.definition
        definition["inputs"] = dict(reversed(tuple(definition["inputs"].items())))
        self.assertNotEqual(service_module.problem_digest(problem.replace(definition=definition)), digest)

    def test_gateway_hides_other_paths(self):
        problems = {name: problem_spec(name) for name in ("alpha", "beta")}
        service = Service(self.root / "server", self.backend, problems, runtime_factory=FakeRuntime)
        with service.start() as running:
            gateway = ClientGateway(Client(running.endpoint, "alpha"), self.root / "gateway.sock")
            self.assertEqual(gateway.info()["problem_paths"], {
                "alpha": str(service.problem_paths["alpha"]),
            })

    def test_incomplete_not_skipped(self):
        self.write(self.root / "one", "alpha")
        other = self.root / "two"
        other.mkdir()
        (other / "definition.json").write_text(json.dumps(problem_spec("beta").definition))
        with self.assertRaisesRegex(FileNotFoundError, "two/workloads.jsonl"):
            service_module.load_problems(self.root, backend=self.backend)

    def test_duplicate_names_rejected(self):
        self.write(self.root / "one", "same")
        self.write(self.root / "two", "same")
        with self.assertRaisesRegex(ValueError, "duplicate problem"):
            service_module.load_problems(self.root, backend=self.backend)

    def test_empty_cases_and_target(self):
        directory = self.write(self.root / "one", "alpha")
        (directory / "workloads.jsonl").write_text("")
        (directory / "target.json").write_text(json.dumps({
            "name": "renamed", "language": "cuda", "platform": "sm120",
        }))
        selected = service_module.load_problems(directory, backend=self.backend)
        self.assertEqual(list(selected), ["renamed"])
        self.assertEqual(load_problem(selected["renamed"]).workloads, ())
        with self.assertRaisesRegex(ValueError, "duplicate problem"):
            service_module.load_problems((directory, directory), backend=self.backend)

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

    def test_single_selects_itself(self):
        problem = problem_spec("alpha")
        service = Service(self.root / "server", self.backend, {"alpha": problem}, runtime_factory=FakeRuntime)
        with service.start() as running:
            client = Client(running.endpoint)
            self.assertEqual(client.problem(), problem)
            result = client.evaluate(make_kernel("alpha", problem, {"kernel.cu": "// source"}), self.root / "work")
            self.assertEqual(result.problem_id, "alpha")

    def test_unbound_gateway_set(self):
        problems = {name: problem_spec(name) for name in ("alpha", "beta")}
        service = Service(self.root / "server", self.backend, problems, runtime_factory=FakeRuntime)
        with service.start() as running:
            upstream = Client(running.endpoint)
            with ClientGateway(upstream, self.root / "gateway.sock") as gateway:
                client = Client(gateway.endpoint)
                self.assertEqual(client.info()["problems"], ["alpha", "beta"])
                self.assertEqual(client.problem("beta"), problems["beta"])
                candidate = make_kernel("beta", problems["beta"], {"kernel.cu": "// source"})
                result = client.evaluate(candidate, self.root / "work", problem_id="beta")
                self.assertEqual(result.problem_id, "beta")
                self.assertTrue(client.result(result.job_id, work=self.root / "work").accepted)
                with self.assertRaisesRegex(ValueError, "problem_id"):
                    client.problem()
                with self.assertRaises(ServerError):
                    client.problem("outside")
                with self.assertRaises(ServerError):
                    client.request("POST", "/evaluate", {"problem_id": "outside", "submission": submission(candidate)})
                payload = json.loads(to_json(result)) | {"problem_id": "outside"}
                with self.assertRaisesRegex(ValueError, "another problem"):
                    gateway.checked_result(payload)


if __name__ == "__main__":
    unittest.main()

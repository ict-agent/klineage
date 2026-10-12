"""Generation produces a measured kernel directly from a Problem."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kernel_fixtures import make_kernel, measured
from problem_fixtures import problem_spec

from klineage.action import Action, Generation
from klineage.action.action import input_paths
from klineage.artifact import Artifact, ArtifactKind, Kernel, Node, load_problem, nested
from klineage.artifact.kernel import latest_kernel
from klineage.backend import get_backend
from klineage.errors import StructuredOutputError
from klineage.server import Client, Service


class GenerationActionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.work = self.root / "work"
        self.problem = problem_spec()
        backend = get_backend("cuda")
        self.enterContext(patch.object(type(backend), "info", return_value={"available": True}))
        service = Service(
            self.root / "server", backend, {self.problem.name: self.problem},
            runtime_factory=lambda **options: SimpleNamespace(
                evaluate=lambda candidate, **kwargs: measured(candidate),
                info=lambda backend, **kwargs: backend.info(),
            ),
        )
        server = self.enterContext(service.start())
        self.client = Client(server.endpoint, self.problem.name)
        self.runner = Mock(return_value=SimpleNamespace(final_message="done"))
        self.enterContext(patch("klineage.action.action.create_runner", return_value=self.runner))

    def action(self, problem=None):
        return Generation(
            problem if problem is not None else self.problem,
            workdir=self.work, evaluation=self.client, max_retries=0,
        )

    def test_returns_generation_artifact(self):
        candidate = make_kernel("generated", self.problem, {"kernel.cu": "// generated"})
        self.runner.side_effect = lambda *args, **kwargs: self.client.evaluate(candidate, self.work)
        result = self.action().run()
        self.assertEqual(result.kind, ArtifactKind.GENERATION)
        kernel = result.part("kernel", Kernel)
        self.assertEqual(load_problem(kernel), self.problem)
        self.assertTrue(kernel.validation.accepted)
        self.assertEqual({node.name for node in result.tree.children}, {"kernel"})
        self.assertTrue(input_paths(result))
        self.runner.mount_memory.assert_not_called()

    def test_requires_problem(self):
        for value in (self.root, make_kernel("kernel", self.problem)):
            with self.subTest(value=type(value).__name__), self.assertRaisesRegex(TypeError, "Problem"):
                self.action(value)
        self.runner.assert_not_called()

    def test_requires_workloads(self):
        with self.assertRaisesRegex(ValueError, "at least one workload"):
            self.action(self.problem.replace(workloads=()))
        self.runner.assert_not_called()

    def test_problem_change_rejected(self):
        changed = make_kernel("other", self.problem.replace(name="other"))
        result = Artifact(ArtifactKind.GENERATION, Node.directory(".", (nested("kernel", changed),)))
        with (
            patch.object(Action, "run", return_value=result),
            self.assertRaisesRegex(StructuredOutputError, "generation changed the problem"),
        ):
            self.action().run()

    def test_can_select_earlier_trial(self):
        candidate = make_kernel("generated", self.problem, {"kernel.cu": "// first"})

        def generate(*args, **kwargs):
            first = self.client.evaluate(candidate, self.work)
            self.client.evaluate(make_kernel("generated", self.problem, {"kernel.cu": "// later"}), self.work)
            self.client.result(first.job_id, work=self.work)

        self.runner.side_effect = generate
        result = self.action().run()
        self.assertEqual(latest_kernel(self.work).name, "2")
        self.assertEqual(result.part("kernel", Kernel).find("submission"), candidate.find("submission"))

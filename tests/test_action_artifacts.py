from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kernel_fixtures import edit_kernel, kernel, measured, skill, write_kernel

from klineage.action import Action, Apply, Decompose, Init, same_implementation
from klineage.artifact import (
    Artifact,
    Kernel,
    Node,
    json_node,
    load_artifact,
    source_files,
)
from klineage.artifact.kernel import latest_kernel
from klineage.backend import get_backend
from klineage.errors import StructuredOutputError
from klineage.memory import SkillCard, save_skill
from klineage.serialization import raw_text
from klineage.server import Client, Service


class Program(Artifact):
    artifact_kind = "device.example"
    schema = Node.directory(
        ".",
        description="Custom device artifact.",
        children=(Node.file("program.txt", description="Device program."),),
    )


class Report(Artifact):
    artifact_kind = "device.report"
    schema = Node.directory(".", (
        Node.file("result.txt"),
        Node.directory("build", (Node.file("proof.txt"),)),
    ))


class ActionArtifactTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.work = self.root / "work"
        self.work.mkdir()
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / "expert.cu").write_text("expert source")
        backend = get_backend("cuda")
        self.enterContext(patch.object(type(backend), "info", return_value={"available": True}))
        self.problem = Path(json.loads(kernel().find("metadata.json").content)["problem"]["path"])
        service = Service(
            self.root / "server", backend, {"gemm": self.problem},
            runtime_factory=lambda **_: SimpleNamespace(
                evaluate=lambda candidate, **kwargs: measured(candidate),
                info=lambda backend, **kwargs: backend.info(),
            ),
        )
        running = self.enterContext(service.start())
        self.client = Client(running.endpoint, "gemm")
        self.enterContext(patch("klineage.action.action.Client", return_value=self.client))
        self.current = self.measure(kernel())
        shutil.rmtree(self.work / "versions")
        self.runner = Mock(return_value=SimpleNamespace(final_message="done"))
        self.enterContext(
            patch("klineage.action.action.create_runner", return_value=self.runner)
        )

    def test_base_returns_artifact(self):
        result = Action("write files", self.work).run()
        self.assertIsInstance(result, Artifact)
        self.assertEqual(result.kind, "action.action")
        self.assertFalse((self.work / "artifact.json").exists())
        self.assertEqual(self.runner.call_count, 1)

    def test_init_returns_kernel_tree(self):
        self.runner.side_effect = lambda *args, **kwargs: self.emit(self.current)
        result = Init(self.problem, self.repo, "expert.cu", workdir=self.work).run()
        self.assert_kernel_result(result.part("kernel", Kernel), self.current)
        saved = load_artifact(latest_kernel(self.work), artifact_type=type(self.current))
        self.assertEqual(source_files(saved.tree), source_files(result.part("kernel", Kernel).tree))
        self.assertFalse((self.work / ".klineage/kernel.json").exists())
        staged = self.work / "workspace/repository"
        self.assertEqual((staged / "expert.cu").read_text(), "expert source")
        self.assertIn(staged, self.runner.read_only)

    def measure(self, candidate):
        self.reply = self.client.evaluate(candidate, self.work)
        return self.reply.kernel

    def assert_kernel_result(self, actual, expected):
        self.assertIs(type(actual), type(expected))
        self.assertTrue(same_implementation(actual, expected))
        self.assertEqual(actual.validation, expected.validation)
        self.assertEqual(
            json.loads(actual.find("metadata.json").content)["name"],
            json.loads(expected.find("metadata.json").content)["name"],
        )

    def emit(self, candidate, card=None):
        self.measure(candidate)
        if card is not None:
            save_skill(card, self.work / "SKILL.md")
        return SimpleNamespace(final_message="done")

    def test_decompose_returns_recipe(self):
        candidate = self.measure(edit_kernel(self.current, source_files={"kernel.cu": "simpler"}))
        card = skill("tiling")
        self.runner.side_effect = lambda *args, **kwargs: self.emit(candidate, card)
        result = Decompose(self.current, workdir=self.work).run()
        self.assert_kernel_result(result.part("kernel", Kernel), candidate)
        self.assertEqual(result.part("skillcard", SkillCard), card)

    def test_removal_needs_card(self):
        candidate = edit_kernel(self.current, source_files={"kernel.cu": "simpler"})
        self.runner.side_effect = lambda *args, **kwargs: self.emit(candidate)
        with self.assertRaises(StructuredOutputError):
            Decompose(self.current, workdir=self.work, max_retries=0).run()
        self.assertFalse((self.work / "artifact.json").exists())

    def test_changed_problem(self):
        metadata = json.loads(self.current.find("metadata.json").content)
        metadata["problem"]["name"] = "different"
        candidate = type(self.current)(self.current.tree.with_child(json_node("metadata.json", metadata)))
        self.runner.side_effect = lambda *args, **kwargs: write_kernel(candidate, self.work / "versions/0")
        with self.assertRaisesRegex(ValueError, "problem reference differs"):
            Apply(self.current, workdir=self.work, max_retries=0).run()

    def test_output_retry_resumes(self):
        trace = self.work / "trace.jsonl"
        trace.write_text(json.dumps({"type": "thread.started", "thread_id": "original-session"}) + "\n")
        def generate(*args, **kwargs):
            if self.runner.call_count == 1:
                return SimpleNamespace(final_message="no files", trace_path=trace)
            return self.emit(self.current)

        self.runner.side_effect = generate
        with patch(
            "klineage.action.action.create_runner",
            return_value=self.runner,
        ) as factory:
            result = Apply(self.current, workdir=self.work, max_retries=1).run()

        self.assert_kernel_result(result.part("kernel", Kernel), self.current)
        factory.assert_called_once()
        self.assertEqual(self.runner.resume_id, "original-session")
        self.assertEqual(self.runner.call_count, 2)

    def test_custom_action_format(self):
        def generate(*args, **kwargs):
            (self.work / "program.txt").write_text("device code")
            return SimpleNamespace(final_message="done")

        self.runner.side_effect = generate
        action = Action("write device code", self.work, output_type=Program)
        result = action.run()
        self.assertEqual(result.kind, "device.example")
        self.assertEqual(result.find("program.txt").content, "device code")
        self.assertIn("from test_action_artifacts import Program", action.prompt)
        self.assertIn("Program.schema", action.prompt)
        self.assertNotIn(raw_text(Program.schema), action.prompt)

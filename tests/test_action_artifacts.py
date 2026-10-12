import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kernel_fixtures import edit_kernel, kernel, measured, skill, write_kernel

from klineage.artifact import Kernel
from klineage.memory import SkillCard
from klineage.action import Action, Apply, Decompose, Init, same_implementation
from klineage.action.action import input_paths
from klineage.artifact import (
    Artifact,
    Node,
    json_node,
    load_artifact,
    save_artifact,
    source_files,
)
from klineage.artifact.kernel import latest_kernel
from klineage.artifact.problem import load_problem
from klineage.backend import get_backend
from klineage.errors import StructuredOutputError
from klineage.memory import save_skill
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

    def test_artifact_handoff(self):
        incoming = self.current
        self.runner.side_effect = lambda *args, **kwargs: self.emit(self.current)
        result = Apply(incoming, workdir=self.work).run()
        self.assert_kernel_result(result.part("kernel", Kernel), self.current)
        self.runner.mount_memory.assert_called_once_with(None, exclude_skills=())

    def test_action_directory_input(self):
        previous = self.root / "previous"
        self.client.evaluate(self.current, previous)
        path = latest_kernel(previous)
        self.assertEqual(input_paths(previous), input_paths(path))
        self.assertNotIn(previous, input_paths(previous))
        self.runner.side_effect = lambda *args, **kwargs: self.emit(self.current)
        result = Apply(previous, workdir=self.work, max_retries=0).run()
        self.assert_kernel_result(result.part("kernel", Kernel), self.current)

    def test_kernel_problem_directory(self):
        problem = self.root / "input-problem"
        problem.mkdir()
        alias = self.root / "problem-link"
        alias.symlink_to(problem, target_is_directory=True)
        metadata = json.loads(self.current.find("metadata.json").content)
        metadata["problem"] = {"path": str(alias), **load_problem(self.current).target()}
        current = type(self.current)(self.current.tree.with_child(json_node("metadata.json", metadata)))
        self.assertEqual(input_paths(current), (problem.resolve(),))

    def test_changed_problem(self):
        metadata = json.loads(self.current.find("metadata.json").content)
        metadata["problem"]["name"] = "different"
        candidate = type(self.current)(self.current.tree.with_child(json_node("metadata.json", metadata)))
        self.runner.side_effect = lambda *args, **kwargs: write_kernel(candidate, self.work / "versions/0")
        with self.assertRaisesRegex(ValueError, "problem reference differs"):
            Apply(self.current, workdir=self.work, max_retries=0).run()

    def test_output_retry(self):
        calls = []

        def generate(*args, **kwargs):
            calls.append(args[0])
            if len(calls) == 1:
                return SimpleNamespace(final_message="no files")
            return self.emit(self.current)

        self.runner.side_effect = generate
        result = Apply(self.current, workdir=self.work, max_retries=1).run()
        self.assert_kernel_result(result.part("kernel", Kernel), self.current)
        self.assertIn("Previous attempt failed", calls[1])
        self.assertIn("klineage.result(job_id)", calls[1])
        self.assertNotIn("client.result", calls[1])

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

    def test_init_accepts_problem(self):
        self.runner.side_effect = lambda *args, **kwargs: self.emit(self.current)
        problem = load_problem(self.current)
        result = Init(problem, self.repo, "expert.cu", workdir=self.work).run()
        self.assert_kernel_result(result.part("kernel", Kernel), self.current)

    def test_init_checks_problem(self):
        problem = load_problem(self.current)
        wrong = problem.replace(name="other")
        candidate = edit_kernel(self.current, problem=wrong)
        service = Service(
            self.root / "multiple", get_backend("cuda"),
            {"gemm": problem, "other": wrong},
            runtime_factory=lambda **_: SimpleNamespace(
                evaluate=lambda current, **kwargs: measured(current),
                info=lambda backend, **kwargs: backend.info(),
            ),
        )
        running = self.enterContext(service.start())
        client = Client(running.endpoint)

        def submit_other(*args, **kwargs):
            client.evaluate(candidate, self.work, problem_id="other")
            return SimpleNamespace(final_message="done")

        self.runner.side_effect = submit_other
        for expected in (self.problem, problem):
            with (
                self.subTest(input=type(expected).__name__),
                self.assertRaisesRegex(StructuredOutputError, "init changed the problem"),
            ):
                Init(
                    expected, self.repo, "expert.cu", workdir=self.work,
                    evaluation=client, max_retries=0,
                ).run()

    def test_init_needs_no_local_gpu(self):
        directory = self.root / "without-target"
        shutil.copytree(self.problem, directory)
        (directory / "target.json").unlink()
        self.runner.side_effect = lambda *args, **kwargs: self.emit(self.current)
        with (
            patch("klineage.backend.detect_backend", side_effect=AssertionError("local device probe")),
            patch("klineage.artifact.problem.trace_module", side_effect=AssertionError("reference executed")),
        ):
            result = Init(directory, self.repo, "expert.cu", workdir=self.work).run()
        self.assert_kernel_result(result.part("kernel", Kernel), self.current)

    def test_init_accepts_problem_json(self):
        path = save_artifact(load_problem(self.current), self.root / "input.json")
        self.runner.side_effect = lambda *args, **kwargs: self.emit(self.current)
        with patch("klineage.backend.detect_backend", side_effect=AssertionError("local device probe")):
            result = Init(path, self.repo, "expert.cu", workdir=self.work).run()
        self.assert_kernel_result(result.part("kernel", Kernel), self.current)

    def test_init_respects_target(self):
        directory = self.root / "different-target"
        shutil.copytree(self.problem, directory)
        target = load_problem(self.current).target()
        target["platform"] = "different-platform"
        (directory / "target.json").write_text(json.dumps(target))
        self.runner.side_effect = lambda *args, **kwargs: self.emit(self.current)
        with self.assertRaisesRegex(StructuredOutputError, "init changed the problem"):
            Init(directory, self.repo, "expert.cu", workdir=self.work).run()

    def test_problem_input_references(self):
        problem = load_problem(self.current)
        path = self.root / "inputs.safetensors"
        path.write_bytes(b"fixed inputs")
        cases = list(problem.resolve_inputs())
        cases[0]["inputs"]["a"] = {
            "type": "safetensors", "path": str(path), "tensor_key": "a",
        }
        problem = problem.replace(workloads=cases)
        self.assertEqual(input_paths(problem), (path,))

    def test_init_expands_problem_home(self):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory(dir=Path.home())))
        path = directory / "problem"
        shutil.copytree(self.problem, path)
        argument = "~/" + str(path.relative_to(Path.home()))
        self.runner.side_effect = lambda *args, **kwargs: self.emit(self.current)
        Init(argument, self.repo, "expert.cu", workdir=self.work, max_retries=0).run()
        self.assertIn(path, self.runner.read_only)

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

    def test_local_output_rejected(self):
        class LocalProgram(Program):
            pass

        with self.assertRaisesRegex(ValueError, "importable"):
            Action("write device code", self.work, output_type=LocalProgram)

    def test_collects_declared_build(self):
        (self.work / "result.txt").write_text("result")
        (self.work / "build").mkdir()
        (self.work / "build/proof.txt").write_text("proof")
        result = Action("collect files", self.work, output_type=Report).run()
        self.assertEqual(result.find("build").find("proof.txt").content, "proof")

    def test_metadata_evidence_inline(self):
        self.runner.side_effect = lambda *args, **kwargs: self.emit(self.current)
        result = Apply(self.current, workdir=self.work, max_retries=0).run()
        metadata = json.loads(result.part("kernel", Kernel).find("metadata.json").content)
        self.assertEqual(
            metadata["evaluation"],
            {"job_id": self.reply.job_id, "error": self.reply.error, "diagnostics": self.reply.diagnostics},
        )
        self.assertEqual(result.part("kernel", Kernel).validation, self.reply.validation)
        self.assertFalse((self.work / "evaluations").exists())
        self.assertNotIn("evaluations", {node.name for node in result.tree.children})

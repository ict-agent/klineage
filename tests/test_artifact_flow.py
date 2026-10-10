import importlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kernel_fixtures import kernel, measured, skill
from prompt_fixtures import prompt_inputs

from klineage.action import same_implementation
from klineage.artifact import Artifact, Kernel, save_artifact
from klineage.backend import get_backend
from klineage.cli.init_memory import init_memory
from klineage.cli.optimize import optimize
from klineage.memory import SkillCard, save_skill
from klineage.serialization import from_json, restore
from klineage.server import Client, Service


class ArtifactFlowTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.dict(os.environ, {"CODEX_HOME": str(self.root / "codex-home")}))
        self.naive = kernel(source="naive")
        self.problem = Path(json.loads(self.naive.node("metadata.json").content)["problem"]["path"])
        self.expert = kernel(source="expert")
        self.optimized = kernel(source="optimized")
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / "expert.cu").write_text("expert")
        backend = get_backend("cuda")
        self.enterContext(patch.object(type(backend), "info", return_value={"available": True}))
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
        self.enterContext(
            patch(
                "klineage.harness.codex_runner.resolve_executable",
                return_value="/bin/true",
            )
        )
        self.enterContext(
            patch(
                "klineage.harness.codex_runner.CodexRunner.__call__",
                autospec=True,
                side_effect=self.generate,
            )
        )

    def generate(self, runner, prompt, *, run_id):
        stage = run_id.split("-", 1)[0]
        values = prompt_inputs(prompt)
        if stage == "Init":
            result = self.expert
        elif stage == "Decompose":
            current = restore(values["input_kernel"], Artifact)
            if not isinstance(current, Kernel):
                current = current.part("kernel", Kernel)
            result = self.naive
            if not same_implementation(current, result):
                save_skill(skill("tiling"), runner.work_dir / "SKILL.md")
        elif stage == "Apply":
            result = self.optimized
        else:
            self.fail(stage)
        self.client.evaluate(result, runner.work_dir)
        return SimpleNamespace(final_message="done")

    def test_memory_returns_tree(self):
        result = init_memory(
            self.problem,
            self.repo,
            "expert.cu",
            workdir=self.root / "run",
            memory_dir=self.root / "memory",
        )
        self.assertEqual(result.kind, "memory")
        self.assertEqual(len(result.tree.children), 1)
        card = result.part(result.tree.children[0].name, SkillCard)
        self.assertEqual(card, skill("tiling"))

    def test_optimize_manifest(self):
        start = self.root / "start"
        start.mkdir()
        path = save_artifact(self.naive, start / "input.json")
        result = optimize(path, workdir=self.root / "optimized")
        self.assertTrue(same_implementation(result.part("kernel", Kernel), self.optimized))
        self.assertEqual(result.part("kernel", Kernel).validation, measured(result.part("kernel", Kernel)))
        replay = optimize(result, workdir=self.root / "replay", max_apply_step=0)
        self.assertEqual(replay.part("kernel", Kernel), result.part("kernel", Kernel))

    def test_cli_artifact_json(self):
        current = self.naive
        cases = (
            (
                "init_memory",
                [
                    "--problem",
                    "p",
                    "--repo",
                    "r",
                    "--expert-kernel",
                    "k",
                    "--memory-dir",
                    "m",
                ],
            ),
            ("optimize", ["--start-kernel", "k", "--workdir", "w"]),
        )
        for name, arguments in cases:
            module = importlib.import_module("klineage.cli." + name)
            output = io.StringIO()
            with (
                self.subTest(cli=name),
                patch.object(module, name, return_value=current),
                redirect_stdout(output),
            ):
                module.main(arguments)
            self.assertEqual(from_json(output.getvalue(), type(current)), current)

    def test_invalid_step_limits(self):
        for limit in (-1, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                optimize(
                    self.naive,
                    workdir=self.root / "work",
                    max_apply_step=limit,
                )

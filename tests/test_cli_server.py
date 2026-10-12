from __future__ import annotations

import argparse
import importlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kernel_fixtures import kernel, measured, skill
from prompt_fixtures import prompt_inputs

from klineage.action import same_implementation
from klineage.artifact import (
    Artifact,
    ArtifactKind,
    Kernel,
    Node,
    nested,
    save_artifact,
)
from klineage.backend import get_backend
from klineage.cli.common import action_arguments, add_run_options
from klineage.cli.init_memory import decompose_steps, init_memory
from klineage.cli.optimize import optimize
from klineage.memory import SkillCard, save_skill
from klineage.serialization import from_json, restore
from klineage.server import Client, Service


class CliServerTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_cli_forwards_client(self):
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
                patch.object(module, name, return_value=kernel()) as invoke,
                redirect_stdout(output),
            ):
                module.main(
                    arguments
                    + [
                        "--server",
                        "http://evaluation.example:9000",
                        "--problem-id",
                        "gemm",
                        "--timeout",
                        "37",
                    ]
                )
            client = invoke.call_args.kwargs["evaluation"]
            self.assertEqual(client.endpoint, "http://evaluation.example:9000")
            self.assertEqual(client.problem_id, "gemm")
            self.assertEqual(client.timeout, 37)
            self.assertEqual(invoke.call_args.kwargs["timeout"], 37)
            self.assertNotIn("server", invoke.call_args.kwargs)

    def test_steps_forward_client(self):
        current = kernel()
        artifact = Artifact.from_tree(
            ArtifactKind.DECOMPOSE, Node.directory(".", (nested("kernel", current),))
        )
        client = object()
        agent = object()
        with (
            patch("klineage.cli.init_memory.Init") as init,
            patch("klineage.cli.init_memory.Decompose") as decompose,
        ):
            init.return_value.run.return_value = artifact
            decompose.return_value.run.return_value = artifact
            list(
                decompose_steps(
                    "problem",
                    "repo",
                    "expert",
                    2,
                    workdir=self.root / "work",
                    timeout=30,
                    max_retries=0,
                    evaluation=client,
                    agent=agent,
                )
            )
        self.assertIs(init.call_args.kwargs["evaluation"], client)
        self.assertIs(decompose.call_args.kwargs["evaluation"], client)
        self.assertIs(init.call_args.kwargs["agent"], agent)
        self.assertIs(decompose.call_args.kwargs["agent"], agent)
        with patch("klineage.cli.optimize.Apply") as apply:
            apply.return_value.run.return_value = artifact
            optimize(
                current,
                workdir=self.root / "optimized",
                max_apply_step=1,
                evaluation=client,
                agent=agent,
            )
        self.assertIs(apply.call_args.kwargs["evaluation"], client)
        self.assertIs(apply.call_args.kwargs["agent"], agent)

    def test_memory_forwards_client(self):
        current = kernel()
        artifact = Artifact.from_tree(
            ArtifactKind.DECOMPOSE, Node.directory(".", (nested("kernel", current),))
        )
        client = object()
        agent = object()
        memory = importlib.import_module("klineage.cli.init_memory")
        with patch.object(
            memory, "decompose_steps", return_value=iter((artifact,))
        ) as decompose:
            memory.init_memory(
                "p",
                "r",
                "k",
                workdir=self.root / "extract",
                memory_dir=self.root / "memory",
                evaluation=client,
                agent=agent,
            )
        self.assertIs(decompose.call_args.kwargs["evaluation"], client)
        self.assertIs(decompose.call_args.kwargs["agent"], agent)

    def test_agent_options_without_server(self):

        parser = argparse.ArgumentParser()
        add_run_options(parser)
        self.assertNotIn("agent", action_arguments(parser.parse_args([])))
        settings = action_arguments(
            parser.parse_args(
                [
                    "--agent-python",
                    sys.executable,
                    "--agent-device-node",
                    "/dev/null",
                    "--agent-device-node",
                    "/dev/zero",
                    "--agent-read-only",
                    str(self.root),
                    "--agent-env",
                    "CUDA_VISIBLE_DEVICES=0",
                    "--agent-env",
                    "CXX_FLAGS=-DVALUE=1",
                ]
            )
        )
        agent = settings["agent"]
        self.assertEqual(agent.python, Path(sys.executable).absolute())
        self.assertEqual(
            tuple(agent.device_nodes), (Path("/dev/null"), Path("/dev/zero"))
        )
        self.assertEqual(tuple(agent.read_only), (self.root,))
        self.assertEqual(
            dict(agent.environment),
            {"CUDA_VISIBLE_DEVICES": "0", "CXX_FLAGS": "-DVALUE=1"},
        )
        self.assertFalse(any(key.startswith("agent_") for key in settings))
        self.assertNotIn("evaluation", settings)
        with self.assertRaisesRegex(ValueError, "NAME=VALUE"):
            action_arguments(parser.parse_args(["--agent-env", "missing-value"]))


class ArtifactFlowTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.dict(os.environ, {"CODEX_HOME": str(self.root / "codex-home")}))
        self.naive = kernel(source="naive")
        self.problem = Path(json.loads(self.naive.find("metadata.json").content)["problem"]["path"])
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
                "klineage.harness.runner.resolve_executable",
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

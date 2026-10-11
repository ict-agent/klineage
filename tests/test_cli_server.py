"""CLI service configuration and device-free input capture."""

import argparse
import importlib
import io
import json
import os
import sys
import tempfile
import tomllib
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kernel_fixtures import kernel, write_kernel

from klineage.action import same_implementation
from klineage.action.action import input_paths
from klineage.artifact import Artifact, ArtifactKind, Node, nested
from klineage.backend.cuda import CudaKernel
from klineage.cli.common import action_arguments, add_run_options
from klineage.cli.init_memory import decompose_steps
from klineage.cli.optimize import optimize


class CliServerTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_env_forwards_timeout(self):
        parser = argparse.ArgumentParser()
        add_run_options(parser)
        with patch.dict(os.environ, {"KLINEAGE_SERVER": "http://evaluation.example:9100"}):
            options = action_arguments(parser.parse_args(["--timeout", "37"]))
        self.assertEqual(options["evaluation"].endpoint, "http://evaluation.example:9100")
        self.assertEqual(options["evaluation"].timeout, 37)

    def test_readonly_directory_no_build(self):
        original = kernel()
        directory = write_kernel(original, self.root / "kernel")
        with patch(
            "klineage.artifact.Kernel.load_kernel",
            side_effect=AssertionError("local build"),
        ):
            result = optimize(directory, workdir=self.root / "read", max_apply_step=0)
        self.assertTrue(same_implementation(result, original))
        self.assertIsNone(result.function)

    def test_extended_kernel_directory(self):
        class ExtendedKernel(CudaKernel):
            artifact_kind = "kernel.cuda.test_cli"

            @classmethod
            def kernel_spec(cls):
                schema = super().kernel_spec()
                return replace(schema, children=(
                    *schema.children, Node.file("notes.txt", description="Backend notes."),
                ))


        original = kernel()
        directory = write_kernel(original, self.root / "kernel")
        notes = directory / "notes.txt"
        notes.write_text("platform metadata")
        problem = Path(json.loads(original.find("metadata.json").content)["problem"]["path"])
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("klineage.backend.get_backend", return_value=SimpleNamespace(kernel_type=ExtendedKernel)),
            patch.object(ExtendedKernel, "build_kernel", side_effect=AssertionError("local build")),
        ):
            with self.subTest(entry="input_paths"):
                self.assertEqual(input_paths(directory), (directory, problem))
            with self.subTest(entry="optimize"):
                result = optimize(directory, workdir=self.root / "read", max_apply_step=0)
                self.assertIsInstance(result, ExtendedKernel)
                self.assertEqual(result.find("notes.txt").content, notes.read_text())
            notes.unlink()
            with self.assertRaisesRegex(ValueError, "notes.txt: missing"):
                optimize(directory, workdir=self.root / "invalid", max_apply_step=0)

    def test_env_server_owns_directory_kind(self):
        directory = write_kernel(kernel(), self.root / "kernel")
        path = directory / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["problem"]["path"] = "/remote-only/problem.json"
        path.write_text(json.dumps(metadata))
        with (
            patch.dict(
                os.environ, {"KLINEAGE_SERVER": "http://evaluation.example:9000"}
            ),
            patch("klineage.cli.optimize.Client") as client,
        ):
            client.return_value.info.return_value = {"kernel_kind": "kernel.cuda"}
            result = optimize(directory, workdir=self.root / "read", max_apply_step=0)
        self.assertEqual(result.kind, "kernel.cuda")
        client.return_value.info.assert_called_once_with()

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
        import argparse

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

    def test_server_script_registered(self):
        project = Path(__file__).resolve().parents[1] / "pyproject.toml"
        scripts = tomllib.loads(project.read_text())["project"]["scripts"]
        self.assertEqual(scripts["klineage-server"], "klineage.server.__main__:main")


if __name__ == "__main__":
    unittest.main()

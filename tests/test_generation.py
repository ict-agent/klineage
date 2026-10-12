"""Problem-to-kernel generation with a local evaluation service."""

import io
import json
import os
import tempfile
import tomllib
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kernel_fixtures import make_kernel, measured, write_kernel
from problem_fixtures import problem_spec
from prompt_fixtures import prompt_inputs

from klineage.artifact import (
    Kernel,
    json_node,
    load_artifact,
    load_problem,
    save_artifact,
    source_files,
)
from klineage.artifact.kernel import latest_kernel
from klineage.artifact.problem import DEFINITION_FILE, WORKLOADS_FILE, problem_reference
from klineage.backend import backends, get_backend
from klineage.cli.generation import generate, main
from klineage.harness import AgentRuntime
from klineage.server import Client, Service
from klineage.server.client import SERVER_ENV


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.problem = problem_spec()
        self.path = self.root / "problem"
        self.path.mkdir()
        self.write_problem(self.problem)
        self.work = self.root / "generated"
        self.runtime = AgentRuntime()
        self.local_runtime = self.enterContext(
            patch.object(AgentRuntime, "local", return_value=self.runtime)
        )
        self.enterContext(patch.dict(os.environ, {SERVER_ENV: ""}))
        self.services = []
        self.registrations = []

        def service(*args, **kwargs):
            self.registrations.append(args[2])
            result = Service(*args, **kwargs)
            self.services.append(result)
            return result

        self.enterContext(patch("klineage.cli.generation.Service", side_effect=service))
        self.worker = self.enterContext(
            patch(
                "klineage.cli.generation.WorkerRuntime",
                side_effect=lambda **_: SimpleNamespace(
                    info=lambda backend, **kwargs: backend.info(),
                    evaluate=lambda candidate, **kwargs: measured(candidate),
                ),
            )
        )
        self.runner = self.enterContext(patch("klineage.action.action.create_runner"))
        self.runner.return_value.side_effect = self.emit

    def write_problem(self, problem):
        (self.path / DEFINITION_FILE).write_text(json.dumps(problem.definition))
        (self.path / WORKLOADS_FILE).write_text(
            "".join(
                json.dumps({"definition": problem.name, "workload": case}) + "\n"
                for case in problem.resolve_inputs()
            )
        )

    def emit(self, prompt, **kwargs):
        client = self.runner.call_args.kwargs["evaluation"]
        work = self.runner.call_args.args[0]
        self.assertTrue(client.endpoint.startswith(("unix://", "http://")))
        self.assertFalse(self.services[-1].root.is_relative_to(work))
        link = work / "problems" / client.problem_id
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.resolve(), self.services[-1].problem_paths[client.problem_id])
        self.assertFalse((work / ".klineage/input-problem.json").exists())
        spec = client.problem()
        backend = get_backend(spec.language, spec.platform)
        candidate = make_kernel(
            spec.name,
            spec,
            {backend.kernel_type.kernel_spec().find("submission").find("solution").children[0].name: "// generated"},
        )
        # An agent may retain the supplied problem reference in its source tree.
        metadata = json.loads(candidate.find("metadata.json").content)
        metadata["problem"] = problem_reference(link.resolve(), spec)
        candidate = type(candidate)(
            candidate.tree.with_child(json_node("metadata.json", metadata))
        )
        directory = write_kernel(candidate, work / "current_kernel")
        client.evaluate(directory, work)
        return SimpleNamespace(final_message="done")

    def test_detected_backend_and_output(self):
        for backend in backends():
            with (
                self.subTest(backend=backend.kind),
                patch("klineage.cli.generation.detect_backend", return_value=backend),
                patch.object(
                    type(backend),
                    "info",
                    return_value={
                        "available": True,
                        "platform": str(backend.kind),
                    },
                ),
            ):
                work = self.root / str(backend.kind)
                result = generate(self.path, workdir=work, timeout=30, max_retries=0)
                self.assertEqual(result.kind, backend.kernel_type.artifact_kind)
                self.assertTrue(result.validation.accepted)
                spec = load_problem(result)
                self.assertEqual(spec.language, backend.language)
                self.assertEqual(spec.platform, str(backend.kind))
                self.assertEqual(spec.definition, self.problem.definition)
                self.assertEqual(spec.workloads, self.problem.workloads)
                saved = load_artifact(latest_kernel(work), artifact_type=type(result))
                self.assertEqual(source_files(saved.tree), source_files(result.tree))
                self.assertFalse((work / ".klineage/kernel.json").exists())
                self.assertFalse(self.services[-1].root.exists())
                self.assertIs(self.runner.call_args.kwargs["agent"], self.runtime)
                self.assertEqual(self.registrations[-1], {self.problem.name: self.path})
                reference = json.loads(result.find("metadata.json").content)["problem"]
                self.assertEqual(Path(reference["path"]).resolve(), self.path)
                self.assertEqual(set(reference), {"path", "name", "language", "platform"})
                settings = self.worker.call_args.kwargs
                self.assertEqual(settings["python"], self.runtime.python)
                self.assertEqual(settings["device_nodes"], self.runtime.device_nodes)

    def test_all_workloads_retained(self):
        workloads = [dict(case) for case in self.problem.resolve_inputs()]
        workloads.append(workloads[0] | {"uuid": "second-case"})
        self.write_problem(self.problem.replace(workloads=workloads))
        backend = get_backend("cuda")
        with (
            patch("klineage.cli.generation.detect_backend", return_value=backend),
            patch.object(type(backend), "info", return_value={"platform": "nvidia"}),
        ):
            result = generate(self.path, workdir=self.work, max_retries=0)
        self.assertEqual(
            set(result.validation.workloads), {case["uuid"] for case in workloads}
        )

    def test_runs_generation_action(self):
        backend = get_backend("cuda")
        with (
            patch("klineage.cli.generation.detect_backend", return_value=backend),
            patch.object(type(backend), "info", return_value={"platform": "sm90"}),
        ):
            generate(self.path, workdir=self.work, max_retries=0)
        call = self.runner.return_value.call_args
        self.assertTrue(call.kwargs["run_id"].startswith("Generation-"))
        values = prompt_inputs(call.args[0])
        self.assertEqual(values["problem"]["artifact_type"], "problem")
        self.assertNotIn("current_kernel", values)
        self.runner.return_value.mount_memory.assert_not_called()

    def test_declared_target(self):
        target = {"name": "custom_gemm", "language": "cuda", "platform": "sm90"}
        (self.path / "target.json").write_text(json.dumps(target))
        backend = get_backend("cuda")
        with (
            patch("klineage.cli.generation.detect_backend", return_value=backend),
            patch.object(type(backend), "info", return_value={"platform": "sm120"}),
        ):
            result = generate(self.path, workdir=self.work, max_retries=0)
        self.assertEqual(load_problem(result).target(), target)
        self.assertEqual(self.registrations[-1], {"custom_gemm": self.path})

    def test_workloads_file_required(self):
        (self.path / WORKLOADS_FILE).unlink()
        backend = get_backend("cuda")
        with (
            patch("klineage.cli.generation.detect_backend", return_value=backend),
            patch.object(type(backend), "info", return_value={"platform": "nvidia"}),
            self.assertRaisesRegex(FileNotFoundError, "workloads.jsonl"),
        ):
            generate(self.path, workdir=self.work, max_retries=0)
        self.runner.assert_not_called()
        self.assertFalse(self.work.exists())

    def test_missing_backend_stops(self):
        with (
            patch(
                "klineage.cli.generation.detect_backend",
                side_effect=RuntimeError("no device"),
            ),
            self.assertRaisesRegex(RuntimeError, "no device"),
        ):
            generate(self.path, workdir=self.work)
        self.assertFalse(self.work.exists())
        self.runner.assert_not_called()

    def test_failure_closes_service(self):
        self.runner.return_value.side_effect = RuntimeError("agent failed")
        backend = get_backend("cuda")
        with (
            patch("klineage.cli.generation.detect_backend", return_value=backend),
            patch.object(type(backend), "info", return_value={"platform": "nvidia"}),
            self.assertRaisesRegex(RuntimeError, "agent failed"),
        ):
            generate(self.path, workdir=self.work, max_retries=0)
        self.assertFalse(self.services[-1].root.exists())

    def test_directory_survives_exit(self):
        backend = get_backend("cuda")
        with (
            patch("klineage.cli.generation.detect_backend", return_value=backend),
            patch.object(type(backend), "info", return_value={"platform": "nvidia"}),
        ):
            result = generate(self.path, workdir=self.work, max_retries=0)
        with patch.object(
            backend.kernel_type, "build_kernel", return_value=lambda: None
        ):
            restored = Kernel.load_kernel(self.work / "current_kernel")
        self.assertEqual(load_problem(restored), load_problem(result))

    def test_stdout_ignores_diagnostics(self):
        backend = get_backend("cuda")
        output, error = io.StringIO(), io.StringIO()

        def info():
            print("runtime diagnostic")
            return {"platform": "nvidia"}

        with (
            patch("klineage.cli.generation.detect_backend", return_value=backend),
            patch.object(type(backend), "info", side_effect=info),
            redirect_stdout(output),
            redirect_stderr(error),
        ):
            main([str(self.path), "--workdir", str(self.work), "--max-retries", "0"])
        self.assertEqual(json.loads(output.getvalue())["artifact_type"], "kernel.cuda")
        self.assertIn("runtime diagnostic", error.getvalue())

    def test_missing_workloads(self):
        self.write_problem(self.problem.replace(workloads=()))
        backend = get_backend("cuda")
        with (
            patch("klineage.cli.generation.detect_backend", return_value=backend),
            patch.object(type(backend), "info", return_value={"platform": "nvidia"}),
            self.assertRaisesRegex(ValueError, "at least one workload"),
        ):
            generate(self.path, workdir=self.work)
        self.runner.assert_not_called()
        self.assertFalse(self.work.exists())

    def test_settings_checked_before_run(self):
        for options in ({"timeout": 0}, {"max_retries": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                generate(self.path, workdir=self.work, **options)
        self.runner.assert_not_called()

    def test_file_input_rejected(self):
        serialized = save_artifact(self.problem, self.root / "problem.json")
        for path in (serialized, self.path / DEFINITION_FILE):
            with (
                self.subTest(path=path),
                patch("klineage.cli.generation.detect_backend") as detect,
                self.assertRaises(NotADirectoryError),
            ):
                generate(path, workdir=self.root / path.stem, max_retries=0)
            detect.assert_not_called()

    def test_existing_workdir_preserved(self):
        self.work.mkdir()
        marker = self.work / "keep.txt"
        marker.write_text("keep")
        backend = get_backend("cuda")
        with (
            patch("klineage.cli.generation.detect_backend", return_value=backend),
            patch.object(type(backend), "info", return_value={"platform": "nvidia"}),
            self.assertRaises(FileExistsError),
        ):
            generate(self.path, workdir=self.work)
        self.assertEqual(marker.read_text(), "keep")
        self.runner.assert_not_called()

    def test_cli_emits_kernel_json(self):
        output = io.StringIO()
        candidate = make_kernel("generated", self.problem)
        with (
            patch("klineage.cli.generation.generate", return_value=candidate) as invoke,
            redirect_stdout(output),
        ):
            main(
                [
                    str(self.path),
                    "--workdir",
                    str(self.work),
                    "--timeout",
                    "600",
                    "--max-retries",
                    "0",
                ]
            )
        self.assertEqual(json.loads(output.getvalue())["artifact_type"], "kernel.cuda")
        invoke.assert_called_once_with(
            problem=self.path, workdir=self.work, timeout=600, max_retries=0
        )

    def remote_client(self, problems=None):
        backend = get_backend("cuda")
        with patch.object(type(backend), "info", return_value={"platform": self.problem.platform}):
            service = Service(
                self.root / "remote", backend, problems or {self.problem.name: self.path},
                runtime_factory=lambda **options: SimpleNamespace(
                    info=lambda backend, **kwargs: {"available": True, "platform": self.problem.platform},
                    evaluate=lambda candidate, **kwargs: measured(candidate),
                ),
            )
        self.services.append(service)
        server = self.enterContext(service.start())
        return Client(server.endpoint, timeout=30)

    def test_remote_without_device(self):
        client = self.remote_client()
        with (
            patch("klineage.cli.generation.detect_backend", side_effect=AssertionError("local GPU probe")),
            patch.object(type(get_backend("cuda")), "info", side_effect=AssertionError("local runtime query")),
        ):
            result = generate(self.path, workdir=self.work, evaluation=client, max_retries=0)
        self.assertTrue(result.validation.accepted)
        self.assertEqual(self.runner.call_args.kwargs["agent"].device_nodes, ())
        self.assertEqual(self.runner.call_args.kwargs["evaluation"].endpoint, client.endpoint)
        self.assertEqual(client.health(), {"status": "ok"})
        self.local_runtime.assert_not_called()
        self.worker.assert_not_called()

    def test_server_environment(self):
        client = self.remote_client()
        with (
            patch.dict(os.environ, {SERVER_ENV: client.endpoint}),
            patch("klineage.cli.generation.detect_backend", side_effect=AssertionError("local GPU probe")),
        ):
            result = generate(self.path, workdir=self.work, max_retries=0)
        self.assertTrue(result.validation.accepted)
        self.local_runtime.assert_not_called()

    def test_remote_selects_problem(self):
        client = self.remote_client({
            self.problem.name: self.path,
            "other": self.problem.replace(name="other"),
        })
        generate(self.path, workdir=self.work, evaluation=client, max_retries=0)
        self.assertEqual(self.runner.call_args.kwargs["evaluation"].problem_id, self.problem.name)

    def test_remote_explicit_id(self):
        client = self.remote_client({"alias": self.path, "other": self.problem.replace(name="other")})
        selected = Client(client.endpoint, "alias", timeout=30)
        generate(self.path, workdir=self.work, evaluation=selected, max_retries=0)
        self.assertEqual(self.runner.call_args.kwargs["evaluation"].problem_id, "alias")

    def test_remote_mirrored_inputs(self):
        local_input, remote_input = self.root / "local.bin", self.root / "remote.bin"
        local_input.write_bytes(b"fixed input")
        remote_input.write_bytes(local_input.read_bytes())
        cases = list(self.problem.workloads)
        cases[0]["inputs"]["a"] = {
            "type": "safetensors", "path": str(local_input), "tensor_key": "a",
        }
        self.write_problem(self.problem.replace(workloads=cases))
        cases[0]["inputs"]["a"]["path"] = str(remote_input)
        registered = self.problem.replace(workloads=cases)
        client = self.remote_client({self.problem.name: registered})
        result = generate(self.path, workdir=self.work, evaluation=client, max_retries=0)
        self.assertEqual(load_problem(result), registered)
        self.assertTrue(result.validation.accepted)

    def test_remote_target_mismatch(self):
        client = self.remote_client()
        (self.path / "target.json").write_text(json.dumps({
            "name": self.problem.name, "language": "hip", "platform": "hygon",
        }))
        with self.assertRaisesRegex(ValueError, "registered problem"):
            generate(self.path, workdir=self.work, evaluation=client, max_retries=0)
        self.assertFalse(self.work.exists())
        self.runner.assert_not_called()

    def test_remote_path_required(self):
        client = self.remote_client()
        info = client.info() | {"problem_paths": {self.problem.name: str(self.root / "unmounted")}}
        with (
            patch.object(client, "info", return_value=info),
            self.assertRaisesRegex(ValueError, "accessible locally"),
        ):
            generate(self.path, workdir=self.work, evaluation=client, max_retries=0)
        self.assertFalse(self.work.exists())
        self.runner.assert_not_called()

    def test_remote_problem_mismatch(self):
        client = self.remote_client()
        cases = list(self.problem.workloads)
        cases[0]["axes"]["M"] //= 2
        self.write_problem(self.problem.replace(workloads=cases))
        with self.assertRaisesRegex(ValueError, "registered problem"):
            generate(self.path, workdir=self.work, evaluation=client, max_retries=0)
        self.assertFalse(self.work.exists())
        self.runner.assert_not_called()

    def test_cli_forwards_server(self):
        candidate = make_kernel("generated", self.problem)
        with (
            patch.dict(os.environ, {SERVER_ENV: "http://ignored.example:9100"}),
            patch("klineage.cli.generation.generate", return_value=candidate) as invoke,
            redirect_stdout(io.StringIO()),
        ):
            main([
                str(self.path), "--server", "http://evaluation.example:9100",
                "--problem-id", "registered-task", "--timeout", "37",
            ])
        client = invoke.call_args.kwargs["evaluation"]
        self.assertEqual(client.endpoint, "http://evaluation.example:9100")
        self.assertEqual(client.problem_id, "registered-task")
        self.assertEqual(client.timeout, 37)

    def test_public_entrypoint(self):
        from klineage.generation import generate as public_generate

        self.assertIs(public_generate, generate)
        project = Path(__file__).resolve().parents[1] / "pyproject.toml"
        scripts = tomllib.loads(project.read_text())["project"]["scripts"]
        self.assertEqual(scripts["klineage-generation"], "klineage.generation:main")


if __name__ == "__main__":
    unittest.main()

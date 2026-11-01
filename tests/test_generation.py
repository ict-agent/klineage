from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kernel_fixtures import make_kernel, measured, write_kernel
from problem_fixtures import problem_spec

from klineage.action import Action, Generation
from klineage.artifact import (
    Artifact,
    ArtifactKind,
    Node,
    json_node,
    load_artifact,
    load_problem,
    nested,
    source_files,
)
from klineage.artifact.kernel import latest_kernel
from klineage.artifact.problem import DEFINITION_FILE, WORKLOADS_FILE, problem_reference
from klineage.backend import backends, get_backend
from klineage.cli.generation import generate, main
from klineage.errors import StructuredOutputError
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
                    info=lambda backend, **kwargs: {"available": True, **backend.info()},
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

    def test_settings_checked_before_run(self):
        for options in ({"timeout": 0}, {"max_retries": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                generate(self.path, workdir=self.work, **options)
        self.runner.assert_not_called()

    def test_worker_unavailable(self):
        backend = get_backend("ascendc")
        self.worker.side_effect = lambda **_: SimpleNamespace(
            info=lambda *args, **kwargs: {"available": False},
            evaluate=lambda candidate, **kwargs: measured(candidate),
        )
        with (
            patch("klineage.cli.generation.detect_backend", return_value=backend),
            patch.object(type(backend), "info", return_value={"available": True, "platform": "ascend"}),
            self.assertRaisesRegex(RuntimeError, "evaluation worker.*unavailable"),
        ):
            generate(self.path, workdir=self.work, max_retries=0)
        self.runner.assert_not_called()

    def test_ascend_device_owner(self):
        backend = get_backend("ascendc")
        self.runtime = AgentRuntime(device_nodes=(Path("/dev/null"),))
        self.local_runtime.return_value = self.runtime
        with (
            patch("klineage.cli.generation.detect_backend", return_value=backend),
            patch.object(type(backend), "info", return_value={"available": True, "platform": "ascend"}),
        ):
            result = generate(self.path, workdir=self.work, max_retries=0)
        self.assertTrue(result.validation.accepted)
        self.assertEqual(self.runner.call_args.kwargs["agent"].device_nodes, ())
        self.assertEqual(self.worker.call_args.kwargs["device_nodes"], self.runtime.device_nodes)

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

    def test_remote_problem_mismatch(self):
        client = self.remote_client()
        cases = list(self.problem.workloads)
        cases[0]["axes"]["M"] //= 2
        self.write_problem(self.problem.replace(workloads=cases))
        with self.assertRaisesRegex(ValueError, "registered problem"):
            generate(self.path, workdir=self.work, evaluation=client, max_retries=0)
        self.assertFalse(self.work.exists())
        self.runner.assert_not_called()


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


    def test_problem_change_rejected(self):
        changed = make_kernel("other", self.problem.replace(name="other"))
        result = Artifact(ArtifactKind.GENERATION, Node.directory(".", (nested("kernel", changed),)))
        with (
            patch.object(Action, "run", return_value=result),
            self.assertRaisesRegex(StructuredOutputError, "generation changed the problem"),
        ):
            self.action().run()

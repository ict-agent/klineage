"""The worker environment must select the service's backend."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from problem_fixtures import problem_spec

from klineage.artifact import load_problem
from klineage.backend import get_backend
from klineage.server.__main__ import main


class ServerCliTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def write_problem(self, language="cuda", platform="sm120"):
        path = self.root / "problem"
        spec = problem_spec(language=language, platform=platform)
        path.mkdir(exist_ok=True)
        (path / "target.json").write_text(json.dumps(spec.target()))
        (path / "definition.json").write_text(json.dumps(spec.definition))
        (path / "workloads.jsonl").write_text(json.dumps({
            "definition": spec.name, "workload": spec.workloads[0],
        }) + "\n")
        return path

    def invoke(self, language, platform, extra=(), *, problem=None):
        path = problem or self.write_problem(language, platform)
        argv = [
            "klineage-server",
            "--root",
            str(self.root / "service"),
            "--backend",
            language,
            "--problem",
            str(path),
            *extra,
        ]
        self.output = io.StringIO()
        with (
            patch("sys.argv", argv),
            patch("klineage.server.__main__.Service") as service,
            patch("klineage.server.__main__.threading.Event"),
            patch("klineage.server.__main__.signal.signal"),
            redirect_stdout(self.output),
        ):
            service.return_value.start.return_value.__enter__.return_value.endpoint = (
                "http://localhost:1"
            )
            main()
        self.listen = service.return_value.start.call_args
        return service.call_args

    def test_listen_aliases(self):
        endpoint = "http://127.0.0.1:8123"
        for option in ("--server", "--listen"):
            with self.subTest(option=option):
                self.invoke("cuda", "sm120", (option, endpoint))
                self.assertEqual(self.listen.args, (endpoint,))

    def test_listen_ignores_client_env(self):
        with patch.dict(os.environ, {"KLINEAGE_SERVER": "http://remote.example:8123"}):
            self.invoke("cuda", "sm120")
        self.assertEqual(self.listen.args, ("http://127.0.0.1:0",))

    def test_canonical_worker_backend(self):
        for language, kind in (
            ("cuda", "cuda"),
            ("hip", "hygon"),
            ("ascendc", "ascend"),
        ):
            with self.subTest(language=language):
                arguments = self.invoke(language, kind)
                environment = arguments.kwargs["runtime_factory"].keywords[
                    "environment"
                ]
                self.assertEqual(environment["KLINEAGE_BACKEND"], kind)

    def test_environment_default(self):
        path = self.write_problem()
        argv = ["klineage-server", "--root", str(self.root / "service"), "--backend", "cuda"]
        with (
            patch.dict(os.environ, {"KLINEAGE_PROBLEM": str(path)}),
            patch("sys.argv", argv),
            patch("klineage.server.__main__.Service") as service,
            patch("klineage.server.__main__.threading.Event"),
            patch("klineage.server.__main__.signal.signal"),
            redirect_stdout(io.StringIO()),
        ):
            service.return_value.start.return_value.__enter__.return_value.endpoint = "http://localhost:1"
            main()
        self.assertEqual(list(service.call_args.args[2]), ["gemm"])

    def test_explicit_overrides_env(self):
        with patch.dict(os.environ, {"KLINEAGE_PROBLEM": str(self.root / "missing")}):
            arguments = self.invoke("cuda", "sm120")
        self.assertEqual(list(arguments.args[2]), ["gemm"])

    def test_alias_is_normalized(self):
        arguments = self.invoke("hip", "hygon", ("--env", "KLINEAGE_BACKEND=hip"))
        environment = arguments.kwargs["runtime_factory"].keywords["environment"]
        self.assertEqual(environment["KLINEAGE_BACKEND"], "hygon")

    def test_backend_override_rejected(self):
        with self.assertRaises(SystemExit):
            self.invoke("hip", "hygon", ("--env", "KLINEAGE_BACKEND=cuda"))

    def test_directory_uses_backend(self):
        problem = problem_spec()
        directory = self.root / "task"
        directory.mkdir()
        (directory / "definition.json").write_text(json.dumps(problem.definition))
        (directory / "workloads.jsonl").write_text(json.dumps({
            "definition": problem.name, "workload": problem.workloads[0],
        }) + "\n")
        backend = get_backend("hip")
        with (
            patch("klineage.backend.detect_backend", side_effect=AssertionError("implicit backend detection")),
            patch.object(type(backend), "info", return_value={"platform": "hygon"}),
        ):
            arguments = self.invoke("hip", "hygon", problem=directory)
            selected = load_problem(arguments.args[2]["gemm"], backend=backend)
        self.assertIs(arguments.args[1], backend)
        self.assertEqual(arguments.args[2]["gemm"], directory)
        self.assertEqual((selected.language, selected.platform), ("hip", "hygon"))

    def test_reference_stdout_is_json(self):
        problem = problem_spec()
        directory = self.root / "task"
        directory.mkdir()
        definition = problem.definition
        definition["reference"] = "print('reference diagnostic')\n" + definition["reference"]
        (directory / "definition.json").write_text(json.dumps(definition))
        (directory / "workloads.jsonl").write_text("")
        backend = get_backend("cuda")
        with patch.object(type(backend), "info", return_value={"platform": "sm120"}):
            self.invoke("cuda", "sm120", problem=directory)
        self.assertEqual(json.loads(self.output.getvalue())["problems"], ["gemm"])


if __name__ == "__main__":
    unittest.main()

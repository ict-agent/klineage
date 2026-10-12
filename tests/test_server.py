from __future__ import annotations

import http.client
import importlib
import io
import json
import os
import socket
import subprocess
import sys
import sysconfig
import tempfile
import threading
import unittest
import uuid
import venv
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlsplit

from kernel_fixtures import (
    accepted,
    edit_kernel,
    kernel,
    make_kernel,
    measured,
    write_kernel,
)
from problem_fixtures import problem_spec
from runtime_fixtures import FakeRuntime

from klineage.artifact import (
    ValidationResult,
    json_node,
    load_artifact,
    load_problem,
    source_files,
)
from klineage.backend import get_backend
from klineage.serialization import from_json, raw_text, restore, to_json
from klineage.server import Client, ClientGateway, ServerError, Service
from klineage.server.__main__ import main
from klineage.server.protocol import Evaluation, submission
from klineage.server.worker import WorkerRuntime


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.backend = get_backend("cuda")
        self.info = patch.object(
            type(self.backend),
            "info",
            return_value={"available": True, "platform": "sm120"},
        )
        self.info.start()
        self.addCleanup(self.info.stop)
        FakeRuntime.calls = []
        FakeRuntime.active = FakeRuntime.peak = FakeRuntime.delay = 0
        self.service = Service(
            self.root / "server",
            self.backend,
            {"gemm": problem_spec()},
            runtime_factory=FakeRuntime,
            timeout=2,
        )
        self.running = self.service.start()
        self.addCleanup(self.running.close)
        self.client = Client(self.running.endpoint, "gemm", timeout=2)

    def test_roundtrip_records(self):
        work = self.root / "work"
        reply = self.client.evaluate(kernel(validation=accepted(99)), work)
        self.assertTrue(reply.accepted)
        self.assertEqual(reply.validation.latency_ms, 1)
        self.assertEqual(load_problem(reply.kernel), problem_spec())
        self.assertEqual(
            json.loads(raw_text(reply.kernel.find("metadata.json")))["problem"][
                "path"
            ],
            str(self.service.problem_paths["gemm"]),
        )
        metadata = json.loads((work / "versions/0/metadata.json").read_text())
        self.assertEqual(metadata["evaluation"]["job_id"], reply.job_id)
        self.assertEqual(metadata["validation"], json.loads(to_json(reply.validation)))
        self.assertIn("measured timing evidence", "".join(metadata["evaluation"]["diagnostics"].values()))
        self.assertFalse((work / "evaluations").exists())
        self.assertFalse((work / ".klineage/evaluation.json").exists())
        self.client.verify(reply.kernel, reply.job_id)
        self.assertIsNone(FakeRuntime.calls[0][0].validation)
        self.assertEqual(self.client.health(), {"status": "ok"})

    def test_numbered_snapshots(self):
        work = self.root / "work"
        first = self.client.evaluate(kernel(), work)
        second = self.client.evaluate(kernel(source="changed"), work)
        restored = self.client.result(first.job_id, work=work)
        versions = work / "versions"
        self.assertEqual(sorted(path.name for path in versions.iterdir()), ["0", "1", "2"])
        for number, result in enumerate((first, second, restored)):
            with self.subTest(number=number):
                saved = load_artifact(versions / str(number), artifact_type=type(result.kernel))
                self.assertEqual(source_files(saved.tree), source_files(result.kernel.tree))
                self.assertTrue(self.client.verify(saved).accepted)
        self.assertFalse((work / ".klineage/kernel.json").exists())

    def test_failed_job_snapshot(self):
        work = self.root / "work"
        with patch.object(FakeRuntime, "evaluate", side_effect=RuntimeError("worker failed")):
            result = self.client.evaluate(kernel(), work)
        saved = load_artifact(work / "versions/0", artifact_type=type(result.kernel))
        self.assertEqual(saved.validation, result.validation)
        self.assertFalse(saved.validation.accepted)

    def test_unix_socket_transport(self):
        socket = self.root / "eval.sock"
        with self.service.start("unix://" + str(socket)) as running:
            client = Client(running.endpoint, "gemm")
            self.assertEqual(client.problem(), problem_spec())
            self.assertTrue(client.evaluate(kernel(), self.root / "unix-work").accepted)
            self.assertEqual(socket.stat().st_mode & 0o777, 0o600)
        self.assertFalse(socket.exists())

    def test_rejects_policy_and_paths(self):
        base = {"problem_id": "gemm", "submission": submission(kernel())}
        for field in (
            "seed",
            "timing",
            "oracle",
            "include_paths",
            "build_root",
            "problem",
        ):
            with self.subTest(field=field), self.assertRaises(ServerError):
                self.client.request("POST", "/evaluate", base | {field: "forged"})
        for name in ("../escape.cu", "/etc/passwd", "a/../../b", "a\\b"):
            value = submission(kernel())
            value["sources"][name] = "x"
            with self.subTest(name=name), self.assertRaises(ServerError):
                self.client.request(
                    "POST", "/evaluate", {"problem_id": "gemm", "submission": value}
                )
        self.assertEqual(FakeRuntime.calls, [])

    def test_baseline_and_serial_device(self):
        FakeRuntime.delay = 0.03
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda n: self.client.evaluate(kernel(), self.root / str(n)),
                    range(2),
                )
            )
        self.assertEqual(FakeRuntime.peak, 1)
        self.client.evaluate(kernel(), self.root / "next", reference=results[0].job_id)
        self.assertIsNotNone(FakeRuntime.calls[-1][2]["reference"])
        self.client.evaluate(kernel(), self.root / "other", reference=kernel())
        self.assertIsNotNone(FakeRuntime.calls[-1][2]["reference"])

    def test_rejects_forged_handoff(self):
        work = self.root / "work"
        reply = self.client.evaluate(kernel(), work)
        changed = edit_kernel(reply.kernel, source_files={"kernel.cu": "changed"})
        # Preserve the expected local problem reference while changing sources.
        original_meta = json.loads(raw_text(reply.kernel.find("metadata.json")))
        changed = type(changed)(
            changed.tree.with_child(json_node("metadata.json", original_meta))
        )
        with self.assertRaisesRegex(ValueError, "sources"):
            self.client.verify(changed, reply.job_id)
        original_meta["validation"] = json.loads(to_json(accepted(2)))
        forged = type(reply.kernel)(
            reply.kernel.tree.with_child(json_node("metadata.json", original_meta))
        )
        with self.assertRaisesRegex(ValueError, "validation"):
            self.client.verify(forged, reply.job_id)

    def test_restart_preserves_job_authority(self):
        reply = self.client.evaluate(kernel(), self.root / "work")
        restarted = Service(
            self.root / "server",
            self.backend,
            {"gemm": problem_spec()},
            runtime_factory=FakeRuntime,
        )
        self.assertEqual(restarted.result(reply.job_id).validation, reply.validation)
        with self.assertRaisesRegex(ValueError, "cannot change"):
            Service(
                self.root / "server",
                self.backend,
                {"gemm": problem_spec(size=64)},
                runtime_factory=FakeRuntime,
            )

    def test_gateway_scopes_task(self):
        with ClientGateway(self.client, self.root / "gateway.sock") as gateway:
            client = Client(gateway.endpoint, "gemm")
            reply = client.evaluate(kernel(), self.root / "work")
            self.assertTrue(reply.accepted)
            self.assertEqual(client.info()["problems"], ["gemm"])
            self.assertEqual(
                client.result(reply.job_id, work=self.root / "work").digest,
                reply.digest,
            )
            with self.assertRaises(ServerError):
                client.problem("another-problem")
            with self.assertRaises(ServerError):
                client.request("POST", "/evaluate", {"problem_id": "another-problem"})
        self.assertFalse((self.root / "gateway.sock").exists())

    def test_gateway_caps_timeout(self):
        host = Client(self.running.endpoint, "gemm", timeout=1)
        with ClientGateway(host, self.root / "gateway.sock") as gateway:
            agent = Client(gateway.endpoint, "gemm", timeout=30)
            for requested, expected in ((30, 1), (0.25, 0.25), (None, 1)):
                with self.subTest(timeout=requested):
                    request = {"problem_id": "gemm", "submission": submission(kernel())}
                    if requested is not None:
                        request["timeout"] = requested
                    with patch.object(host, "request", wraps=host.request) as forward:
                        reply = agent.request("POST", "/evaluate", request)
                    self.assertIsNone(reply["error"])
                    self.assertEqual(forward.call_args.args[2]["timeout"], expected)
                    self.assertEqual(forward.call_args.kwargs["timeout"], expected)
                    self.assertGreater(FakeRuntime.calls[-1][1], 0)
                    self.assertLessEqual(FakeRuntime.calls[-1][1], expected)


class ServerBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.backend = get_backend("cuda")
        self.enterContext(
            patch.object(type(self.backend), "info", return_value={"available": True})
        )
        FakeRuntime.calls = []
        FakeRuntime.delay = 0
        self.service = Service(
            self.root / "service",
            self.backend,
            {"first": problem_spec(), "second": problem_spec(size=64)},
            runtime_factory=FakeRuntime,
        )
        self.running = self.enterContext(self.service.start())
        self.client = Client(self.running.endpoint, "first")

    def request(self):
        return {"problem_id": "first", "submission": submission(kernel())}

    def raw_request(self, body, headers=()):
        target = urlsplit(self.running.endpoint)
        header = b"POST /evaluate HTTP/1.1\r\nHost: localhost\r\n"
        header += b"\r\n".join(headers) + b"\r\n\r\n"
        with socket.create_connection(
            (target.hostname, target.port), timeout=2
        ) as connection:
            connection.sendall(header + body)
            connection.shutdown(socket.SHUT_WR)
            response = http.client.HTTPResponse(connection)
            response.begin()
            return response.status, response.read()

    def test_duplicate_lengths_rejected(self):
        body = json.dumps(self.request()).encode()
        length = b"Content-Length: " + str(len(body)).encode()
        status, _ = self.raw_request(body, (length, length))
        self.assertEqual(status, http.client.BAD_REQUEST)
        self.assertFalse(FakeRuntime.calls)
        self.assertEqual(self.client.health(), {"status": "ok"})

    def test_duplicate_json_rejected(self):
        body = b'{"problem_id":"second",' + json.dumps(self.request()).encode()[1:]
        status, _ = self.raw_request(
            body, (b"Content-Length: " + str(len(body)).encode(),)
        )
        self.assertEqual(status, http.client.BAD_REQUEST)
        self.assertFalse(FakeRuntime.calls)

    def test_response_paths_rejected(self):
        reply = self.service.evaluate(self.request())
        payload = json.loads(to_json(reply))
        payload["job_id"] = "../../escape"
        with self.assertRaises(ValueError):
            restore(payload, Evaluation)
        payload = json.loads(to_json(reply))
        payload["diagnostics"] = {"../../escape": "unexpected"}
        with self.assertRaises(ValueError):
            restore(payload, Evaluation)

    def test_source_fifo_rejected(self):
        work = self.root / "agent"
        source = write_kernel(kernel(), self.root / "source")
        (source / "metadata.json").unlink()
        os.mkfifo(source / "metadata.json")
        code = """
from pathlib import Path
import sys
from klineage.server import Client
try:
    Client(sys.argv[1], 'first').evaluate(Path(sys.argv[2]), Path(sys.argv[3]))
except (OSError, ValueError):
    pass
else:
    raise AssertionError('FIFO accepted')
"""
        subprocess.run(
            [sys.executable, "-c", code, self.running.endpoint, str(source), str(work)],
            check=True, timeout=2,
        )
        self.assertFalse(FakeRuntime.calls)

    def test_close_drains_requests(self):
        for endpoint in ("http://127.0.0.1:0", "unix://" + str(self.root / "drain.sock")):
            entered, release, closed = (threading.Event() for _ in range(3))
            results, errors = [], []

            def evaluate(candidate, **options):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test worker was not released")
                return measured(candidate)

            running = self.service.start(endpoint)
            client = Client(running.endpoint, "first")

            def submit():
                try:
                    results.append(client.request("POST", "/evaluate", self.request()))
                except Exception as error:
                    errors.append(error)

            def close():
                running.close()
                closed.set()

            request = threading.Thread(target=submit)
            closing = threading.Thread(target=close)
            with self.subTest(endpoint=endpoint), patch.object(FakeRuntime, "evaluate", side_effect=evaluate):
                try:
                    request.start()
                    self.assertTrue(entered.wait(2))
                    closing.start()
                    self.assertFalse(closed.wait(0.75), "close returned while the worker was active")
                finally:
                    release.set()
                    request.join(3)
                    if closing.ident is not None:
                        closing.join(3)
                    running.close()

                self.assertFalse(request.is_alive())
                self.assertFalse(closing.is_alive())
                self.assertFalse(errors)
                result = self.service.result(results[0]["job_id"])
                self.assertTrue(result.accepted)
                self.assertEqual(result.digest, results[0]["digest"])

    def test_gateway_stats_failures(self):
        logs = self.root / "attempt"
        gateway = ClientGateway(self.client, self.root / "gateway.sock", directory=logs)
        with patch.object(self.client, "request", side_effect=ServerError("upstream unavailable")):
            with self.assertRaisesRegex(ServerError, "upstream unavailable"):
                gateway.evaluate(self.request())
        events = [json.loads(line) for line in (logs / "stats.jsonl").read_text().splitlines()]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event"], "evaluate")
        self.assertEqual(events[0]["problem_id"], "first")
        self.assertFalse(events[0]["accepted"])
        self.assertEqual(events[0]["error"], "ServerError: upstream unavailable")
        self.assertLessEqual(events[0]["at"], events[0]["finished_at"])
        self.assertNotIn("submission", events[0])

    def test_gateway_blocks_other_jobs(self):
        other = self.service.evaluate(self.request() | {"problem_id": "second"})
        with ClientGateway(self.client, self.root / "gateway.sock") as gateway:
            client = Client(gateway.endpoint, "first")
            with self.assertRaises(ServerError):
                client.result(other.job_id, work=self.root / "agent")
            with self.assertRaises(ServerError):
                client.evaluate(kernel(), self.root / "agent", reference=other.job_id)
        self.assertEqual(len(FakeRuntime.calls), 1)


class ServerWorkloadTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        base = problem_spec()
        first = base.workloads[0] | {"uuid": "small"}
        second = first | {"uuid": "large", "axes": {"M": 256, "N": 256, "K": 256}}
        self.problem = base.replace(workloads=(first, second))

    def service(self, problem, result):
        return Service(
            self.root / "server", get_backend("cuda"), {"task": problem},
            runtime_factory=lambda **kwargs: SimpleNamespace(evaluate=lambda *args, **kwargs: result),
        )

    def evaluate(self, service, problem):
        kernel = make_kernel("test", problem, {"kernel.cu": "test"})
        return service.evaluate({"problem_id": "task", "submission": submission(kernel)})

    def test_missing_case_cannot_pass(self):
        result = ValidationResult.combine({"small": ValidationResult(True, True, True, 1.0)})
        reply = self.evaluate(self.service(self.problem, result), self.problem)
        self.assertFalse(reply.accepted)
        self.assertIn("workload", reply.error)
        self.assertEqual(set(reply.validation.workloads), {"small", "large"})

    def test_extra_case_cannot_pass(self):
        result = ValidationResult.combine({
            name: ValidationResult(True, True, True, 1.0)
            for name in ("small", "large", "unregistered")
        })
        reply = self.evaluate(self.service(self.problem, result), self.problem)
        self.assertFalse(reply.accepted)
        self.assertIn("workload", reply.error)

    def test_complete_report_roundtrip(self):
        result = ValidationResult.combine({
            "small": ValidationResult(True, True, True, 1.0),
            "large": ValidationResult(True, True, True, 9.0),
        })
        reply = self.evaluate(self.service(self.problem, result), self.problem)
        self.assertTrue(reply.accepted)
        self.assertEqual(from_json(to_json(reply), Evaluation).validation, result)
        self.assertIsNone(reply.validation.latency_ms)

    def test_definition_only_cannot_pass(self):
        problem = self.problem.replace(workloads=())
        optimistic = ValidationResult(True, True, True, 1.0)
        reply = self.evaluate(self.service(problem, optimistic), problem)
        self.assertFalse(reply.accepted)
        self.assertIn("workload", reply.error)

    def test_late_report_not_accepted(self):
        clock = [0.0]
        report = ValidationResult.combine({
            case["uuid"]: ValidationResult(True, True, True, 1.0)
            for case in self.problem.workloads
        })

        def evaluate(*args, **kwargs):
            clock[0] = 6.0
            return report

        service = Service(
            self.root / "server", get_backend("cuda"), {"task": self.problem},
            timeout=5,
            runtime_factory=lambda **kwargs: SimpleNamespace(evaluate=evaluate),
        )
        with patch("time.monotonic", side_effect=lambda: clock[0]):
            result = self.evaluate(service, self.problem)
        self.assertFalse(result.accepted)
        self.assertIn("deadline", result.error)
        self.assertEqual(result.validation.workloads, report.workloads)


class ServerInfoTests(unittest.TestCase):
    def test_configured_worker_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = root / "python-env"
            venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
            python = environment / "bin/python"
            secret = root / "host-secret"
            secret.write_text("private")
            device = root / "selected-device"
            device.symlink_to("/dev/null")
            name = "info_backend_" + uuid.uuid4().hex
            (root / (name + ".py")).write_text(f"""
import os,sys,uuid
from pathlib import Path
from klineage.backend import register_backend
from klineage.backend.cuda import CudaBackend,CudaKernel
class InfoKernel(CudaKernel):
    artifact_kind = 'kernel.' + {name!r}
class InfoBackend(CudaBackend):
    kind = {name!r}
    language = {name!r}
    platform_prefixes = ({name!r},)
    kernel_type = InfoKernel
    def info(self):
        return {{'available': True, 'prefix': sys.prefix,
                'setting': os.environ.get('DEVICE_SETTING'),
                'device_visible': Path({str(device)!r}).is_char_device(),
                'host_secret_visible': Path({str(secret)!r}).exists(),
                'probe_id': uuid.uuid4().hex}}
register_backend(InfoBackend())
""")
            try:
                with patch.object(sys, "path", [str(root), *sys.path]):
                    importlib.invalidate_caches()
                    importlib.import_module(name)
                    factory = partial(
                        WorkerRuntime, python=python, worker_plugins=(name,),
                        device_nodes=(device,), read_only=(Path(sys.prefix),),
                        environment={"DEVICE_SETTING": "assigned", "PYTHONPATH": sysconfig.get_path("purelib")},
                    )
                    service = Service(
                        root / "service", get_backend(name),
                        {"task": problem_spec(language=name, platform=name)},
                        runtime_factory=factory,
                    )
                    with service.start() as running:
                        client = Client(running.endpoint, "task")
                        first = client.info()["device"]
                        second = client.info()["device"]
                    self.assertEqual(first["prefix"], str(environment))
                    self.assertEqual(first["setting"], "assigned")
                    self.assertTrue(first["device_visible"])
                    self.assertFalse(first["host_secret_visible"])
                    self.assertEqual(first["probe_id"], second["probe_id"])
                    record = json.loads((root / "service/info/evaluations/info-process.json").read_text())
                    self.assertEqual(Path(record["command"][0]).name, "bwrap")
                    self.assertIn(str(python), record["command"])
            finally:
                from klineage.backend.registry import REGISTERED_BACKENDS

                REGISTERED_BACKENDS[:] = [item for item in REGISTERED_BACKENDS if item.kind != name]
                sys.modules.pop(name, None)


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

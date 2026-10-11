"""Exercise transport boundaries with malformed and filesystem-backed inputs."""

import errno
import http.client
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

from kernel_fixtures import kernel, measured, write_kernel
from problem_fixtures import problem_spec
from test_server import FakeRuntime

from klineage.artifact import file_ref
from klineage.artifact.kernel import latest_kernel
from klineage.backend import get_backend
from klineage.errors import StructuredOutputError
from klineage.serialization import restore, to_json
from klineage.server import Client, ClientGateway, ServerError, Service
from klineage.server.protocol import REQUEST_LIMIT, Evaluation, submission
from klineage.server.service import RequestHandler, RunningServer
from klineage.server.worker import WorkerRuntime


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

    def test_listener_rejects_url_parts(self):
        endpoints = (
            "http://user:password@127.0.0.1:0",
            "http://127.0.0.1:0/evaluation",
            "http://127.0.0.1:0?option=value",
            "http://127.0.0.1:0#fragment",
            "http://127.0.0.1:0\n",
            "unix://host/socket",
            "unix:relative.sock",
            "unix:///",
            "unix:///socket%00",
        )
        for endpoint in endpoints:
            with (
                self.subTest(endpoint=endpoint),
                patch("klineage.server.service.ThreadingHTTPServer") as server,
                patch("klineage.server.service.UnixHTTPServer") as unix,
            ):
                server.return_value.server_address = ("127.0.0.1", 0)
                with self.assertRaisesRegex(ValueError, "endpoint"):
                    RunningServer(self.service, endpoint)
                server.assert_not_called()
                unix.assert_not_called()

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

    def test_worker_log_symlink_not_read(self):
        secret = self.root / "server-secret"
        secret.write_text("private-server-data")

        def evaluate(candidate, **options):
            work = options["build_root"].parent
            work.mkdir(parents=True)
            (work / "leak.log").symlink_to(secret)
            return measured(candidate)

        with patch.object(FakeRuntime, "evaluate", side_effect=evaluate):
            reply = self.client.evaluate(kernel(), self.root / "agent")
        self.assertNotIn("private-server-data", to_json(reply, indent=None))
        self.assertNotIn("work/leak.log", reply.diagnostics)

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

    def test_bad_framing_keeps_service_alive(self):
        cases = (
            (b"{", (b"Content-Length: 1",)),
            (b"{}", (b"Content-Length: 10",)),
            (b"", (b"Content-Length: " + str(REQUEST_LIMIT + 1).encode(),)),
            (b"{}", (b"Content-Length: 2", b"Transfer-Encoding: chunked")),
        )
        for body, headers in cases:
            with self.subTest(headers=headers):
                status, data = self.raw_request(body, headers)
                self.assertIn(
                    status,
                    (http.client.BAD_REQUEST, http.client.REQUEST_ENTITY_TOO_LARGE),
                )
                self.assertEqual(self.client.health(), {"status": "ok"})
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises((ValueError, ServerError)):
                self.client.request(
                    "POST", "/evaluate", self.request() | {"timeout": value}
                )
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

    def test_problem_response_cannot_read(self):
        payload = {"problem": json.loads(to_json(problem_spec()))}
        target = payload["problem"]["tree"]["children"][0]
        local = self.root / "target.json"
        local.write_text(target["content"])
        target["content"] = None
        target["reference"] = json.loads(to_json(file_ref(local)))
        with (
            patch.object(self.client, "request", return_value=payload),
            patch(
                "klineage.serialization.FileRefHandler.raw_text",
                side_effect=AssertionError("untrusted FileRef read"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "inline"):
                self.client.problem()

    def test_problem_link_guard(self):
        work = self.root / "agent"
        work.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        (work / "problems").symlink_to(outside, target_is_directory=True)
        with self.assertRaises((OSError, ValueError)):
            self.client.evaluate(kernel(), work)
        self.assertEqual(list(outside.iterdir()), [])
        self.assertFalse((work / "versions").exists())

    def test_old_hardlink_untouched(self):
        work = self.root / "agent"
        previous = work / "versions/0"
        previous.mkdir(parents=True)
        outside = self.root / "outside"
        outside.write_text("preserve")
        (previous / "metadata.json").hardlink_to(outside)
        reply = self.client.evaluate(kernel(), work)
        self.assertEqual(outside.read_text(), "preserve")
        self.assertTrue((previous / "metadata.json").samefile(outside))
        self.assertEqual(latest_kernel(work), work / "versions/1")
        metadata = json.loads((latest_kernel(work) / "metadata.json").read_text())
        self.assertEqual(metadata["evaluation"]["job_id"], reply.job_id)

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

    def test_client_validates_endpoint(self):
        for endpoint in (
            "http://", "https:///api", "http://host:bad", "http://host:70000",
            "http://host:0", "http://white space", "unix://host/path", "unix://",
            "unix:///", "unix:///tmp/socket%00",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                Client(endpoint, "first")

    def test_registered_problem_stays_pinned(self):
        path = self.service.problem_paths["first"] / "workloads.jsonl"
        before = path.read_text()
        path.write_text(problem_spec(size=64).find("workloads.jsonl").content)
        with self.assertRaisesRegex(ServerError, "changed"):
            self.client.evaluate(kernel(), self.root / "agent")
        self.assertFalse(FakeRuntime.calls)
        path.write_text(before)
        self.assertTrue(self.client.evaluate(kernel(), self.root / "agent").accepted)

    def test_close_keeps_new_socket(self):
        path = self.root / "rebound.sock"
        first = self.service.start("unix://" + str(path))
        first.close()
        with self.service.start("unix://" + str(path)) as second:
            first.close()
            self.assertTrue(path.is_socket())
            client = Client(second.endpoint, "first")
            self.assertEqual(client.health(), {"status": "ok"})

    def test_close_preserves_replacement(self):
        path = self.root / "replaced.sock"
        running = self.service.start("unix://" + str(path))
        path.unlink()
        path.write_text("replacement")
        running.close()
        self.assertEqual(path.read_text(), "replacement")

    def test_close_before_start(self):
        code = """
from types import SimpleNamespace
from klineage.server.service import RunningServer
server = RunningServer(SimpleNamespace(), 'http://127.0.0.1:0')
server.close()
server.close()
"""
        subprocess.run([sys.executable, "-c", code], check=True, timeout=2)

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

    def test_close_bounds_partial_read(self):
        entered = threading.Event()
        setup = RequestHandler.setup

        def start(handler):
            setup(handler)
            entered.set()

        with (
            patch("klineage.server.service.HTTP_READ_TIMEOUT", 0.05),
            patch.object(RequestHandler, "setup", start),
        ):
            running = self.service.start()
            target = urlsplit(running.endpoint)
            connection = socket.create_connection((target.hostname, target.port), timeout=2)
            try:
                connection.sendall(b"GET /health HTTP/1.1\r\nHost: localhost")
                self.assertTrue(entered.wait(2))
                closing = threading.Thread(target=running.close)
                closing.start()
                closing.join(2)
                self.assertFalse(closing.is_alive(), "partial request blocked shutdown")
            finally:
                connection.close()
                running.close()

    def test_source_directory_symlink_rejected(self):
        directory = write_kernel(kernel(), self.root / "kernel")
        link = self.root / "linked"
        link.symlink_to(directory, target_is_directory=True)
        with self.assertRaises(OSError) as error:
            self.client.evaluate(link, self.root / "agent")
        self.assertIn(error.exception.errno, (errno.ELOOP, errno.ENOTDIR))
        self.assertFalse(FakeRuntime.calls)
        source = directory / "submission/solution/kernel.cu"
        source.unlink()
        source.symlink_to(self.root / "absent")
        with self.assertRaisesRegex(StructuredOutputError, "symlink"):
            self.client.evaluate(directory, self.root / "agent")

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

    def test_gateway_failed_result(self):
        logs = self.root / "attempt"
        gateway = ClientGateway(self.client, self.root / "gateway.sock", directory=logs)
        with patch.object(FakeRuntime, "evaluate", side_effect=RuntimeError("compiler failed")):
            result = gateway.evaluate(self.request())
        self.assertFalse(result.accepted)
        events = [json.loads(line) for line in (logs / "stats.jsonl").read_text().splitlines()]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["job_id"], result.job_id)
        self.assertFalse(events[0]["accepted"])
        self.assertFalse(events[0]["compile_passed"])
        self.assertIn("compiler failed", events[0]["error"])

    def test_parallel_gateway_stats(self):
        gateways = [ClientGateway(
            Client(self.running.endpoint, name), self.root / f"{name}.sock",
            directory=self.root / name,
        ) for name in ("first", "second")]
        with patch.dict(os.environ, {"KLINEAGE_LOG_DIR": str(self.root / "unrelated")}):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(gateway.evaluate, self.request() | {"problem_id": name})
                           for gateway, name in zip(gateways, ("first", "second"), strict=True)]
                results = [future.result() for future in futures]
        for result in results:
            events = [json.loads(line) for line in
                      (self.root / result.problem_id / "stats.jsonl").read_text().splitlines()]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["problem_id"], result.problem_id)
            self.assertEqual(events[0]["job_id"], result.job_id)
        self.assertFalse((self.root / "unrelated").exists())

    def test_gateway_result_no_event(self):
        logs = self.root / "attempt"
        gateway = ClientGateway(self.client, self.root / "gateway.sock", directory=logs)
        result = gateway.evaluate(self.request())
        self.assertTrue(result.accepted)
        gateway.result(result.job_id)
        events = [json.loads(line) for line in (logs / "stats.jsonl").read_text().splitlines()]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["job_id"], result.job_id)
        self.assertTrue(events[0]["accepted"])

    def test_gateway_blocks_other_jobs(self):
        other = self.service.evaluate(self.request() | {"problem_id": "second"})
        with ClientGateway(self.client, self.root / "gateway.sock") as gateway:
            client = Client(gateway.endpoint, "first")
            with self.assertRaises(ServerError):
                client.result(other.job_id, work=self.root / "agent")
            with self.assertRaises(ServerError):
                client.evaluate(kernel(), self.root / "agent", reference=other.job_id)
        self.assertEqual(len(FakeRuntime.calls), 1)

    def test_canonical_problem_is_readonly(self):
        candidate = self.service.kernel("first", submission(kernel()))
        problem_path = self.service.problem_paths["first"]
        definition = problem_path / "definition.json"
        before = definition.read_bytes()
        secret = self.service.root / "private-state"
        secret.write_text("private-server-data")
        build = self.root / "isolated" / "work" / "build"
        build.mkdir(parents=True)
        logs = self.root / "isolated" / "logs"
        logs.mkdir()
        code = f"""
import errno,json,pathlib
problem=pathlib.Path({str(problem_path)!r})
assert problem.is_dir()
assert not pathlib.Path({str(secret)!r}).exists()
try:
    (problem/'definition.json').write_text('tampered')
except OSError as error:
    assert error.errno == errno.EROFS
else:
    raise AssertionError('worker modified the registered oracle')
print(json.dumps({{'readonly': True}}))
"""
        request = {
            "operation": "evaluate",
            "kernel": json.loads(to_json(candidate)),
            "config": {"build_root": str(build), "include_paths": []},
        }
        with patch("klineage.harness.eval.WORKER_ENTRY", code):
            result = WorkerRuntime(timeout=10).invoke(request, logs, "readonly")
        self.assertEqual(result, {"readonly": True})
        self.assertEqual(definition.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

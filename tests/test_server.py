"""Source transport and authoritative evaluation records."""

import http.client
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

from kernel_fixtures import accepted, edit_kernel, kernel, measured, write_kernel
from problem_fixtures import problem_spec

from klineage.artifact import FileRef, Node, json_node, load_artifact, load_problem, source_files
from klineage.backend import get_backend
from klineage.serialization import raw_text, to_json
from klineage.server import Client, ClientGateway, ServerError, Service
from klineage.server.protocol import submission


class FakeRuntime:
    calls = []
    active = 0
    peak = 0
    delay = 0

    def __init__(self, *, timeout):
        self.timeout = timeout

    def info(self, backend, **options):
        return backend.info()

    def evaluate(self, candidate, **options):
        type(self).active += 1
        type(self).peak = max(type(self).peak, type(self).active)
        try:
            time.sleep(self.delay)
            self.calls.append((candidate, self.timeout, options))
            logs = options["log_dir"]
            logs.mkdir(parents=True)
            (logs / "eval-0001-stderr.log").write_text("measured timing evidence\n")
            return measured(candidate)
        finally:
            type(self).active -= 1


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
            json.loads(raw_text(reply.kernel.node("metadata.json")))["problem"][
                "path"
            ],
            str(self.service.problem_paths["gemm"]),
        )
        metadata = json.loads((work / "kernels/0/metadata.json").read_text())
        self.assertEqual(metadata["evaluation"]["job_id"], reply.job_id)
        self.assertEqual(metadata["validation"], json.loads(to_json(reply.validation)))
        self.assertIn("measured timing evidence", "".join(metadata["evaluation"]["diagnostics"].values()))
        self.assertFalse((work / "evaluations").exists())
        self.assertFalse((work / ".klineage/evaluation.json").exists())
        self.client.verify(reply.kernel, reply.job_id)
        self.assertIsNone(FakeRuntime.calls[0][0].validation)
        self.assertEqual(self.client.health(), {"status": "ok"})

    def test_info_uses_cached_worker(self):
        metadata = {"available": True, "device": "worker-device", "torch_version": "worker-version"}
        with patch.object(FakeRuntime, "info", return_value=metadata) as probe:
            service = Service(
                self.root / "metadata", self.backend, {"gemm": problem_spec()},
                runtime_factory=FakeRuntime,
            )
            with service.start() as running:
                client = Client(running.endpoint, "gemm")
                self.assertEqual(client.info()["device"], metadata)
                self.assertEqual(client.info()["device"], metadata)
        probe.assert_called_once()

    def test_numbered_snapshots(self):
        work = self.root / "work"
        first = self.client.evaluate(kernel(), work)
        second = self.client.evaluate(kernel(source="changed"), work)
        restored = self.client.result(first.job_id, work=work)
        versions = work / "kernels"
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
        saved = load_artifact(work / "kernels/0", artifact_type=type(result.kernel))
        self.assertEqual(saved.validation, result.validation)
        self.assertFalse(saved.validation.accepted)

    def test_problem_directory_link(self):
        work = self.root / "work"
        reply = self.client.evaluate(kernel(), work)
        link = work / "problems/gemm"
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.resolve(), self.service.problem_paths["gemm"].resolve())
        self.assertEqual(load_problem(link), load_problem(reply.kernel))
        self.assertFalse((work / ".klineage/problems").exists())
        self.assertFalse((work / ".klineage/input-problem.json").exists())
        self.assertFalse((self.service.root / "jobs" / reply.job_id / "candidate.json").exists())

    def test_unix_socket_transport(self):
        socket = self.root / "eval.sock"
        with self.service.start("unix://" + str(socket)) as running:
            client = Client(running.endpoint, "gemm")
            self.assertEqual(client.problem(), problem_spec())
            self.assertTrue(client.evaluate(kernel(), self.root / "unix-work").accepted)
            self.assertEqual(socket.stat().st_mode & 0o777, 0o600)
        self.assertFalse(socket.exists())

    def test_directory_never_builds_locally(self):
        source = write_kernel(kernel(), self.root / "source")
        with patch(
            "klineage.artifact.kernel.Kernel.load_kernel",
            side_effect=AssertionError("local build"),
        ):
            self.assertTrue(self.client.evaluate(source, self.root / "work").accepted)

    def test_http_without_auth(self):
        target = urlsplit(self.running.endpoint)
        connection = http.client.HTTPConnection(target.hostname, target.port)
        self.addCleanup(connection.close)
        connection.request("GET", "/health")
        reply = connection.getresponse()
        self.addCleanup(reply.close)
        self.assertEqual(reply.status, http.client.OK)
        self.assertEqual(json.loads(reply.read()), {"status": "ok"})

    def test_client_omits_auth(self):
        request = http.client.HTTPConnection.request
        headers = []

        def capture(connection, *args, **kwargs):
            headers.append(kwargs.get("headers", {}))
            return request(connection, *args, **kwargs)

        with patch.object(http.client.HTTPConnection, "request", capture):
            self.client.health()
        self.assertNotIn("Authorization", headers[0])

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

    def test_rejects_source_fileref(self):
        candidate = kernel()
        reference = FileRef("/etc/passwd", "0" * 64, 100)
        tree = candidate.tree.with_child(
            Node.directory(
                "submission", (
                    Node.file("config.toml", reference=reference),
                    candidate.node("submission/solution"),
                )
            )
        )
        with self.assertRaisesRegex(ValueError, "FileRefs"):
            self.client.evaluate(type(candidate)(tree), self.root / "work")

    def test_rejects_wrong_backend(self):
        with self.assertRaisesRegex(ServerError, "backend"):
            self.client.evaluate(
                kernel(language="hip", platform="hygon"), self.root / "work"
            )

    def test_oracle_is_owned_by_service(self):
        candidate = kernel()
        metadata = json.loads(raw_text(candidate.node("metadata.json")))
        metadata["problem"] = {"path": "/etc/passwd", "sha256": "0" * 64, "size": 100}
        candidate = type(candidate)(
            candidate.tree.with_child(json_node("metadata.json", metadata))
        )
        reply = self.client.evaluate(candidate, self.root / "work")
        self.assertTrue(reply.accepted)
        self.assertEqual(load_problem(FakeRuntime.calls[0][0]), problem_spec())

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

    def test_queue_time_counts_against_budget(self):
        self.service.lock.acquire()
        try:
            reply = self.client.evaluate(kernel(), self.root / "work", timeout=0.01)
        finally:
            self.service.lock.release()
        self.assertFalse(reply.accepted)
        self.assertIn("waiting for device", reply.error)
        self.assertEqual(FakeRuntime.calls, [])

    def test_worker_failure_is_recorded(self):
        with patch.object(
            FakeRuntime, "evaluate", side_effect=RuntimeError("worker failed")
        ):
            reply = self.client.evaluate(kernel(), self.root / "work")
        self.assertFalse(reply.accepted)
        self.assertIn("worker failed", reply.error)
        self.assertIn("error.log", reply.diagnostics)
        self.assertEqual(self.service.result(reply.job_id).validation, reply.validation)
        self.assertTrue(self.client.evaluate(kernel(), self.root / "next").accepted)

    def test_rejects_forged_handoff(self):
        work = self.root / "work"
        reply = self.client.evaluate(kernel(), work)
        changed = edit_kernel(reply.kernel, source_files={"kernel.cu": "changed"})
        # Preserve the expected local problem reference while changing sources.
        original_meta = json.loads(raw_text(reply.kernel.node("metadata.json")))
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

    def test_verify_checks_ref_before_read(self):
        work = self.root / "work"
        reply = self.client.evaluate(kernel(), work)
        metadata = json.loads(raw_text(reply.kernel.node("metadata.json")))
        metadata["problem"]["path"] = "/etc/passwd"
        forged = type(reply.kernel)(
            reply.kernel.tree.with_child(json_node("metadata.json", metadata))
        )
        with patch(
            "klineage.server.client.load_problem",
            side_effect=AssertionError("untrusted read"),
        ):
            with self.assertRaisesRegex(ValueError, "reference"):
                self.client.verify(forged, reply.job_id)

    def test_verify_does_not_repair_tampering(self):
        work = self.root / "work"
        reply = self.client.evaluate(kernel(), work)
        path = self.service.problem_paths["gemm"] / "definition.json"
        path.write_text("tampered")
        with self.assertRaises(ServerError):
            self.client.verify(reply.kernel, reply.job_id)
        self.assertEqual(path.read_text(), "tampered")

    def test_client_import_does_not_load_torch(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from klineage.server import Client; assert 'torch' not in sys.modules",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_environment_configuration(self):
        with patch.dict(
            os.environ,
            {
                "KLINEAGE_SERVER": self.running.endpoint,
                "KLINEAGE_PROBLEM": str(self.root / "mounted-problems"),
            },
        ):
            self.assertEqual(Client().problem(), problem_spec())

    def test_backend_protocol_is_shared(self):
        for language, platform in (("hip", "hygon"), ("ascendc", "ascend")):
            with self.subTest(language=language):
                backend = get_backend(language)
                service = Service(
                    self.root / language,
                    backend,
                    {"target": problem_spec(language=language, platform=platform)},
                    runtime_factory=FakeRuntime,
                )
                with patch.object(
                    type(backend), "info", return_value={"available": True}
                ):
                    with service.start() as running:
                        client = Client(running.endpoint, "target")
                        reply = client.evaluate(
                            kernel(language=language, platform=platform),
                            self.root / (language + "-work"),
                        )
                self.assertTrue(reply.accepted)
                self.assertIsInstance(FakeRuntime.calls[-1][0], backend.kernel_type)

    def test_baseline_cannot_cross_problems(self):
        service = Service(
            self.root / "multi",
            self.backend,
            {"first": problem_spec(), "second": problem_spec(size=64)},
            runtime_factory=FakeRuntime,
        )
        with service.start() as running:
            first = Client(running.endpoint, "first")
            second = Client(running.endpoint, "second")
            baseline = first.evaluate(kernel(), self.root / "first")
            with self.assertRaisesRegex(ServerError, "baseline"):
                second.evaluate(
                    kernel(), self.root / "second", reference=baseline.job_id
                )

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

    def test_restore_result(self):
        work = self.root / "work"
        first = self.client.evaluate(kernel(), work)
        self.client.evaluate(kernel(source="changed"), work)
        restored = self.client.result(first.job_id, work=work)
        self.assertEqual(restored.digest, first.digest)
        self.assertEqual(
            json.loads((work / "kernels/2/metadata.json").read_text())["evaluation"]["job_id"],
            first.job_id,
        )

    def test_matching_job_acceptance(self):
        work = self.root / "work"
        first, second = sorted(
            (self.client.evaluate(kernel(), work), self.client.evaluate(kernel(), work)),
            key=lambda result: result.job_id,
        )
        # A local snapshot cannot override the server's job-level error.
        path = self.service.root / "jobs" / first.job_id / "result.json"
        path.write_text(to_json(replace(first, error="evaluation deadline expired")))
        result = self.client.verify(second.kernel)
        self.assertEqual(result.job_id, second.job_id)
        self.assertTrue(result.accepted)
        with self.assertRaisesRegex(ValueError, "evaluation metadata differs"):
            self.client.verify(first.kernel)

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


if __name__ == "__main__":
    unittest.main()

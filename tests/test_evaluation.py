from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from contextlib import chdir
from pathlib import Path
from unittest.mock import patch

from kernel_fixtures import kernel, measured, write_kernel

import klineage
from klineage.artifact import load_problem
from klineage.backend import get_backend
from klineage.harness import AgentRuntime
from klineage.serialization import to_json
from klineage.server import Client, ServerError, Service


class CpuRuntime:
    calls = []

    def __init__(self, *, timeout, **settings):
        self.timeout = timeout

    def info(self, backend, **options):
        return backend.info()

    def evaluate(self, candidate, **options):
        self.calls.append((candidate, options))
        logs = options["log_dir"]
        logs.mkdir(parents=True)
        (logs / "worker.log").write_text("CPU fixture evidence")
        return measured(candidate)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.work = self.root / "work"
        self.work.mkdir()
        self.enterContext(chdir(self.work))
        self.enterContext(patch.dict(os.environ, {"KLINEAGE_SERVER": ""}))
        self.backend = get_backend("cuda")
        self.enterContext(patch.object(type(self.backend), "info", return_value={
            "available": True, "platform": "sm120",
        }))
        self.candidate = kernel()
        self.reference = json.loads(self.candidate.find("metadata.json").content)["problem"]
        self.problem_path = Path(self.reference["path"])
        CpuRuntime.calls = []

    def local_runtime(self):
        self.enterContext(patch("klineage.evaluation.WorkerRuntime", CpuRuntime))
        self.enterContext(patch("klineage.evaluation.AgentRuntime.local", return_value=AgentRuntime(python=Path(sys.executable))))

    def start_server(self, problems=None):
        service = Service(self.root / "server", self.backend,
                          problems or {"alias": self.problem_path}, runtime_factory=CpuRuntime)
        running = self.enterContext(service.start())
        os.environ["KLINEAGE_SERVER"] = running.endpoint
        return service, running

    def test_local_directory(self):
        self.local_runtime()
        path = write_kernel(self.candidate, self.work / "current_kernel")
        with patch.object(Service, "start", side_effect=AssertionError("no listener")):
            reply = klineage.evaluate(path)
        self.assertTrue(reply.accepted)
        self.assertEqual(load_problem(reply.kernel), load_problem(self.candidate))
        metadata = json.loads((self.work / "versions/0/metadata.json").read_text())
        self.assertEqual(metadata["problem"]["path"], str(self.problem_path))
        self.assertTrue(Path(metadata["problem"]["path"]).is_dir())
        self.assertEqual(metadata["evaluation"]["job_id"], reply.job_id)
        self.assertEqual(metadata["validation"], json.loads(to_json(reply.validation)))
        self.assertIn("CPU fixture evidence", "".join(reply.diagnostics.values()))
        self.assertEqual(len(CpuRuntime.calls), 1)

    def test_local_baseline_result(self):
        self.local_runtime()
        first = klineage.evaluate(self.candidate)
        klineage.evaluate(self.candidate, reference=first)
        self.assertEqual(CpuRuntime.calls[1][1]["reference"].find("submission"), first.kernel.find("submission"))
        restored = klineage.result(first)
        self.assertEqual(restored.job_id, first.job_id)
        self.assertEqual(len(CpuRuntime.calls), 2)
        self.assertTrue((self.work / "versions/2/metadata.json").is_file())
        with self.assertRaisesRegex(ValueError, "Evaluation"):
            klineage.result(first.job_id)

    def test_local_failure_is_saved(self):
        self.local_runtime()
        with patch.object(CpuRuntime, "evaluate", side_effect=RuntimeError("compile failed")):
            reply = klineage.evaluate(self.candidate)
        self.assertFalse(reply.accepted)
        self.assertIn("compile failed", reply.error)
        self.assertTrue((self.work / "versions/0/metadata.json").exists())

    def test_remote_alias(self):
        other = kernel(case="other")
        other_path = Path(json.loads(other.find("metadata.json").content)["problem"]["path"])
        service, running = self.start_server({"first": other_path, "gemm-alias": self.problem_path})
        path = write_kernel(self.candidate, self.work / "current_kernel")
        with patch("klineage.evaluation.detect_backend", side_effect=AssertionError("no local detection")), patch("klineage.evaluation.AgentRuntime.local", side_effect=AssertionError("no local devices")):
            reply = klineage.evaluate(path)
            self.assertEqual(klineage.info()["problems"], ["first", "gemm-alias"])
            self.assertEqual(klineage.problem("gemm-alias").name, "gemm")
            selected = klineage.result(reply)
        self.assertEqual(reply.problem_id, "gemm-alias")
        self.assertEqual(reply.job_id, selected.job_id)
        self.assertEqual(len(CpuRuntime.calls), 1)
        Client(running.endpoint, "gemm-alias").verify(selected.kernel)
        self.assertEqual(service.problem_paths["gemm-alias"], self.problem_path)

    def test_remote_failure(self):
        self.start_server()
        with patch.object(Client, "evaluate", side_effect=ServerError("offline")), patch("klineage.evaluation.AgentRuntime.local", side_effect=AssertionError("no fallback")):
            with self.assertRaisesRegex(ServerError, "offline"):
                klineage.evaluate(self.candidate)
        self.assertFalse((self.work / "versions").exists())

    def test_local_info_problem_links(self):
        links = self.work / "problems"
        links.mkdir()
        (links / "alias").symlink_to(self.problem_path, target_is_directory=True)
        with patch("klineage.evaluation.detect_backend", return_value=self.backend):
            self.assertEqual(klineage.info()["problems"], ["alias"])
            self.assertEqual(klineage.problem(), load_problem(self.candidate))
            self.assertEqual(klineage.problem("alias"), load_problem(self.candidate))
            self.assertEqual(klineage.problem(self.problem_path), load_problem(self.candidate))

"""Action handoffs must match service evidence."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kernel_fixtures import accepted, kernel, measured, write_kernel

from klineage.artifact import Kernel
from klineage.action import Apply, same_implementation
from klineage.artifact import (
    Node,
    ValidationResult,
    json_node,
    load_problem,
    source_tree,
)
from klineage.backend import get_backend
from klineage.artifact.kernel import latest_kernel
from klineage.artifact.tree import MAX_INLINE_BYTES
from klineage.errors import StructuredOutputError
from klineage.serialization import to_json
from klineage.server import PROBLEM_ENV, Client, ServerError, Service
from klineage.server.client import SERVER_ENV
from klineage.server.protocol import DIAGNOSTIC_LIMIT


class ActionServerTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.work = self.root / "work"
        self.work.mkdir()
        self.initial = kernel(validation=accepted(5))
        self.validation = accepted()
        backend = get_backend("cuda")
        self.enterContext(patch.object(type(backend), "info", return_value={"available": True}))
        self.service = Service(
            self.root / "server", backend, {"gemm": load_problem(self.initial)},
            runtime_factory=lambda **_: SimpleNamespace(
                evaluate=lambda candidate, **kwargs: measured(candidate, self.validation),
                info=lambda backend, **kwargs: backend.info(),
            ),
        )
        running = self.enterContext(self.service.start())
        self.client = Client(running.endpoint, "gemm")
        self.runner = Mock()
        self.enterContext(patch("klineage.action.action.CodexRunner", return_value=self.runner))

    def action(self, **kwargs):
        return Apply(
            self.initial, workdir=self.work, evaluation=self.client,
            max_retries=0, **kwargs,
        )

    def emit(self, change=None):
        measured = self.client.evaluate(self.initial, self.work)
        if change:
            write_kernel(change(measured.kernel), latest_kernel(self.work))
        return SimpleNamespace(final_message="done")

    def metadata(self, current, **changes):
        metadata = json.loads(current.node("metadata.json").content)
        return type(current)(current.tree.with_child(json_node("metadata.json", metadata | changes)))

    def test_measured_unchanged_handoff(self):
        self.runner.side_effect = lambda *args, **kwargs: self.emit()
        result = self.action().run().part("kernel", Kernel)
        self.assertTrue(same_implementation(result, self.initial))
        self.assertEqual(result.validation, measured(result))
        self.assertNotEqual(result.validation, self.initial.validation)
        self.assertIs(self.runner.evaluation, self.client)
        self.assertTrue(self.client.verify(result).accepted)

    def test_forged_validation_rejected(self):
        self.runner.side_effect = lambda *args, **kwargs: self.emit(
            lambda current: self.metadata(current, validation=json.loads(to_json(accepted(0.00001))))
        )
        with self.assertRaisesRegex(ValueError, "validation differs"):
            self.action().run()

    def test_changed_sources_rejected(self):
        def change(current):
            sources = source_tree({"config.toml": "changed", "solution/kernel.cu": "changed"})
            return type(current)(current.tree.with_child(Node.directory("submission", sources.children)))

        self.runner.side_effect = lambda *args, **kwargs: self.emit(change)
        with self.assertRaisesRegex(ValueError, "sources differ"):
            self.action().run()

    def test_problem_ref_rejected_first(self):
        reference = {"path": "/etc/shadow", "name": "gemm", "language": "cuda", "platform": "sm120"}
        self.runner.side_effect = lambda *args, **kwargs: self.emit(
            lambda current: self.metadata(current, problem=reference)
        )
        with patch("klineage.server.client.load_problem") as read:
            with self.assertRaisesRegex(ValueError, "problem reference differs"):
                self.action().run()
            read.assert_not_called()

    def test_changed_problem_file(self):
        def generate(*args, **kwargs):
            result = self.emit()
            (self.service.problem_paths["gemm"] / "definition.json").write_text("forged oracle")
            return result

        self.runner.side_effect = generate
        with self.assertRaises(ServerError):
            self.action().run()

    def test_failed_measurement_rejected(self):
        self.validation = ValidationResult(True, False, False)
        self.runner.side_effect = lambda *args, **kwargs: self.emit()
        with self.assertRaisesRegex(StructuredOutputError, "did not pass server evaluation"):
            self.action().run()

    def test_missing_job_id_rejected(self):
        def generate(*args, **kwargs):
            result = self.emit()
            path = latest_kernel(self.work) / "metadata.json"
            metadata = json.loads(path.read_text())
            del metadata["evaluation"]
            path.write_text(json.dumps(metadata))
            return result

        self.runner.side_effect = generate
        with self.assertRaisesRegex(ValueError, "job ID"):
            self.action().run()

    def test_symlink_metadata_rejected(self):
        def generate(*args, **kwargs):
            result = self.emit()
            record = latest_kernel(self.work) / "metadata.json"
            external = self.root / "external-result.json"
            record.rename(external)
            record.symlink_to(external)
            return result

        self.runner.side_effect = generate
        with self.assertRaises((ValueError, OSError)):
            self.action().run()

    def test_earlier_result_handoff(self):
        def generate(*args, **kwargs):
            earlier = self.client.evaluate(self.initial, self.work)
            self.client.evaluate(kernel(source="different candidate"), self.work)
            self.client.result(earlier.job_id, work=self.work)
            return SimpleNamespace(final_message="done")

        self.runner.side_effect = generate
        result = self.action().run().part("kernel", Kernel)
        self.assertTrue(same_implementation(result, self.initial))

    def test_forged_record_rejected(self):
        def generate(*args, **kwargs):
            reply = self.client.evaluate(self.initial, self.work)
            validation = measured(reply.kernel, accepted(0.00001))
            forged = self.metadata(reply.kernel, validation=json.loads(to_json(validation)))
            write_kernel(forged, latest_kernel(self.work))
            return SimpleNamespace(final_message="done")

        self.runner.side_effect = generate
        with self.assertRaisesRegex(ValueError, "validation differs"):
            self.action().run()

    def test_no_server_no_fallback(self):
        with patch.dict(os.environ, {SERVER_ENV: "", PROBLEM_ENV: ""}):
            with self.assertRaisesRegex(ValueError, "endpoint is required"):
                Apply(self.initial, workdir=self.work, max_retries=0).run()
        self.runner.assert_not_called()

    def test_terminal_name_preserved(self):
        renamed = self.metadata(self.initial, name="renamed")

        def generate(*args, **kwargs):
            self.client.evaluate(renamed, self.work)
            return SimpleNamespace(final_message="done")

        self.runner.side_effect = generate
        with self.assertRaisesRegex(StructuredOutputError, "metadata"):
            self.action().run()

    def test_latest_failure_rejected(self):
        def generate(*args, **kwargs):
            self.client.evaluate(self.initial, self.work)
            self.validation = ValidationResult(True, False, False)
            self.client.evaluate(self.initial, self.work)
            return SimpleNamespace(final_message="done")

        self.runner.side_effect = generate
        with self.assertRaisesRegex(StructuredOutputError, "did not pass server evaluation"):
            self.action().run()

    def test_scratch_sources_ignored(self):
        def generate(*args, **kwargs):
            write_kernel(kernel(source="unfinished trial"), self.work / "kernel")
            return self.emit()

        self.runner.side_effect = generate
        result = self.action().run().part("kernel", Kernel)
        self.assertTrue(same_implementation(result, self.initial))

    def test_large_diagnostics_handoff(self):
        def evaluate(candidate, **options):
            logs = options["log_dir"]
            logs.mkdir(parents=True)
            for index in range(MAX_INLINE_BYTES // DIAGNOSTIC_LIMIT + 1):
                (logs / f"{index}.log").write_text("x" * DIAGNOSTIC_LIMIT)
            return measured(candidate)

        runtime = SimpleNamespace(
            info=lambda backend, **options: backend.info(), evaluate=evaluate,
        )
        self.runner.side_effect = lambda *args, **kwargs: self.emit()
        with patch.object(self.service, "runtime_factory", return_value=runtime):
            result = self.action().run().part("kernel", Kernel)
        metadata = latest_kernel(self.work) / "metadata.json"
        self.assertGreater(metadata.stat().st_size, MAX_INLINE_BYTES)
        self.assertIsNotNone(result.node("metadata.json").content)
        self.assertTrue(self.client.verify(result).accepted)
        self.assertFalse((self.work / "evaluations").exists())

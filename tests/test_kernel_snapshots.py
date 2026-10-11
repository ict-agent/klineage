import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from kernel_fixtures import edit_kernel, kernel

from klineage.artifact import Node, json_node, load_artifact, source_files
from klineage.artifact import kernel as storage
from klineage.artifact.tree import MAX_INLINE_BYTES, SOURCE_BYTE_LIMIT
from klineage.backend.cuda import CudaKernel
from klineage.errors import StructuredOutputError


class KernelSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.work = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.current = kernel()

    def test_snapshot_writes_tree(self):
        path = storage.snapshot_kernel(self.current, self.work)
        self.assertEqual(path, self.work / "versions/0")
        self.assertEqual({child.name for child in path.iterdir()}, {"metadata.json", "submission"})
        restored = load_artifact(path, artifact_type=type(self.current))
        self.assertEqual(source_files(restored.tree), source_files(self.current.tree))
        self.assertIsNone(restored.function)
        self.assertEqual(storage.latest_kernel(self.work), path)
        self.assertEqual(storage.next_version(self.work), 1)

    def test_versions_keep_own_sources(self):
        first = storage.snapshot_kernel(self.current, self.work)
        changed = edit_kernel(self.current, source_files={"other.cu": "changed"})
        second = storage.snapshot_kernel(changed, self.work)
        self.assertEqual(second, self.work / "versions/1")
        self.assertTrue((first / "submission/solution/kernel.cu").is_file())
        self.assertFalse((second / "submission/solution/kernel.cu").exists())
        self.assertEqual((second / "submission/solution/other.cu").read_text(), "changed")

    def test_extra_nodes_and_empty_dirs(self):
        class ExtendedKernel(CudaKernel):
            artifact_kind = "kernel.cuda.test_snapshot"

            @classmethod
            def kernel_spec(cls):
                schema = super().kernel_spec()
                return replace(schema, children=(
                    *schema.children,
                    Node.file("notes.txt", description="Backend notes."),
                    Node.directory("assets", description="Backend assets."),
                ))


        tree = self.current.tree.with_child(Node.file("notes.txt", "backend notes"))
        current = ExtendedKernel(replace(tree.with_child(Node.directory("assets")), artifact_type=None))
        path = storage.snapshot_kernel(current, self.work)
        self.assertEqual((path / "notes.txt").read_text(), "backend notes")
        self.assertTrue((path / "assets").is_dir())
        restored = storage.capture_kernel(path)
        self.assertEqual(restored.find("notes.txt").content, "backend notes")
        self.assertEqual(restored.find("assets").children, ())

    def test_binary_reference_snapshot(self):
        from klineage.artifact import file_ref

        data = self.work / "weights.bin"
        data.write_bytes(b"\x00\xff\x80\r\n")
        tree = self.current.tree.with_child(Node.file("weights.bin", reference=file_ref(data)))
        class BinaryKernel(CudaKernel):
            @classmethod
            def kernel_spec(cls):
                return super().kernel_spec().with_child(Node.file("weights.bin"))

        current = BinaryKernel(tree)
        saved = storage.snapshot_kernel(current, self.work)
        self.assertEqual((saved / "weights.bin").read_bytes(), data.read_bytes())
        data.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            storage.snapshot_kernel(current, self.work)
        self.assertEqual(storage.next_version(self.work), 1)

    def test_large_text_stays_inline(self):
        current = edit_kernel(self.current, source_files={"large.cu": "x" * (MAX_INLINE_BYTES + 1)})
        metadata = json.loads(current.find("metadata.json").content)
        metadata["evaluation"] = {"diagnostics": {"build.log": "x" * (MAX_INLINE_BYTES + 1)}}
        current = type(current)(current.tree.with_child(json_node("metadata.json", metadata)))
        path = storage.snapshot_kernel(current, self.work)
        tree = storage.capture_kernel(path)
        self.assertEqual(tree.find("metadata.json").content, current.find("metadata.json").content)
        self.assertEqual(tree.find("submission").find("solution").find("large.cu").content, "x" * (MAX_INLINE_BYTES + 1))
        restored = load_artifact(path, artifact_type=type(current))
        self.assertIsNotNone(restored.find("metadata.json").content)
        self.assertIsNotNone(restored.find("submission").find("solution").find("large.cu").content)

    def test_metadata_not_source_budget(self):
        path = storage.snapshot_kernel(self.current, self.work)
        (path / "metadata.json").write_text("x" * (SOURCE_BYTE_LIMIT + 1))
        self.assertEqual(len(storage.capture_kernel(path).find("metadata.json").content), SOURCE_BYTE_LIMIT + 1)
        (path / "submission/solution/kernel.cu").write_text("x" * (SOURCE_BYTE_LIMIT + 1))
        with self.assertRaisesRegex(StructuredOutputError, "source bundle exceeds"):
            storage.capture_kernel(path)

    def test_capture_rejects_links(self):
        path = storage.snapshot_kernel(self.current, self.work)
        outside = self.work / "outside"
        outside.write_text("private")
        (path / "metadata.json").unlink()
        (path / "metadata.json").symlink_to(outside)
        with self.assertRaises((OSError, ValueError)):
            storage.capture_kernel(path)

    def test_version_order_is_numeric(self):
        self.assertEqual(storage.next_version(self.work), 0)
        with self.assertRaises(FileNotFoundError):
            storage.latest_kernel(self.work)
        for name in ("2", "10", "notes"):
            (self.work / "versions" / name).mkdir(parents=True)
        self.assertEqual(storage.latest_kernel(self.work), self.work / "versions/10")
        self.assertEqual(storage.next_version(self.work), 11)

    def test_failed_write_is_not_a_version(self):
        first = storage.snapshot_kernel(self.current, self.work)
        with (
            patch.object(storage, "safe_file", side_effect=OSError("disk full")),
            self.assertRaisesRegex(OSError, "disk full"),
        ):
            storage.snapshot_kernel(self.current, self.work)
        self.assertEqual(storage.latest_kernel(self.work), first)
        self.assertEqual(list((self.work / "versions").iterdir()), [first])

    def test_symlink_paths_rejected(self):
        external = self.work / "external"
        external.mkdir()
        for relative in ("work", "work/versions", "work/versions/0"):
            with self.subTest(relative=relative):
                work = self.work / relative.replace("/", "-")
                target = work if relative == "work" else work / relative.removeprefix("work/")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(external, target_is_directory=True)
                with self.assertRaises((OSError, ValueError)):
                    storage.snapshot_kernel(self.current, work)
                self.assertEqual(list(external.iterdir()), [])

    def test_concurrent_versions_are_whole(self):
        count = 8
        with ThreadPoolExecutor(max_workers=count) as executor:
            paths = tuple(executor.map(
                lambda _: storage.snapshot_kernel(self.current, self.work), range(count),
            ))
        self.assertEqual({path.name for path in paths}, {str(i) for i in range(count)})
        for path in paths:
            restored = load_artifact(path, artifact_type=type(self.current))
            self.assertEqual(source_files(restored.tree), source_files(self.current.tree))

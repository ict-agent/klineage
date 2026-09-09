import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import chdir
from pathlib import Path

from klineage.errors import StructuredOutputError
from klineage.harness.artifacts import read_source_tree, snapshot


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.bundle = self.root / "submission"
        self.bundle.mkdir()

    def test_relative_directory(self):
        (self.bundle / "kernel.cu").write_text("kernel")
        with chdir(self.root):
            self.assertEqual(
                read_source_tree(Path("submission"), "bundle"),
                {"kernel.cu": "kernel"},
            )

    def test_preserves_source_text(self):
        source = b"// source\r\nvoid kernel() {}\r\n"
        (self.bundle / "kernel.cu").write_bytes(source)
        self.assertEqual(
            read_source_tree(self.bundle, "bundle")["kernel.cu"].encode(), source
        )

    def test_rejects_symlink_root(self):
        (self.bundle / "kernel.cu").write_text("kernel")
        link = self.root / "link"
        link.symlink_to(self.bundle, target_is_directory=True)
        with self.assertRaises(StructuredOutputError):
            read_source_tree(link, "bundle")

    def test_rejects_invalid_files(self):
        source = self.bundle / "kernel.cu"
        for contents in (b"\xff", b"invalid\x00text"):
            with self.subTest(contents=contents):
                source.write_bytes(contents)
                with self.assertRaises(StructuredOutputError):
                    read_source_tree(self.bundle, "bundle")
        source.unlink()
        source.symlink_to(self.root / "missing")
        with self.assertRaises(StructuredOutputError):
            read_source_tree(self.bundle, "bundle")


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.work = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_snapshot_freezes_sources(self):
        (self.work / "kernel.cu").write_text("stale predecessor")
        sources = {"kernel.cu": "current kernel\n", "nested/helper.cuh": "helper\n"}
        first = Path(snapshot(self.work, sources))
        second = Path(snapshot(self.work, {"kernel.cu": "next kernel\n"}))
        self.assertNotEqual(first, second)
        self.assertEqual(
            {
                p.relative_to(first).as_posix(): p.read_text()
                for p in first.rglob("*")
                if p.is_file()
            },
            sources,
        )
        self.assertEqual((second / "kernel.cu").read_text(), "next kernel\n")
        self.assertEqual((self.work / "kernel.cu").read_text(), "stale predecessor")

    def test_snapshot_rejects_escape(self):
        for name in ("../outside.cu", "/tmp/outside.cu", "nested/../../outside.cu"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                snapshot(self.work, {name: "kernel"})

    def test_reuses_identical_sources(self):
        sources = {"kernel.cu": "kernel\r\n", "helper.cuh": "header\n"}
        first = Path(snapshot(self.work, sources))
        timestamp = (first / "kernel.cu").stat().st_mtime_ns
        second = Path(snapshot(self.work, dict(reversed(sources.items()))))
        self.assertEqual(first, second)
        self.assertEqual((first / "kernel.cu").stat().st_mtime_ns, timestamp)
        self.assertEqual((first / "kernel.cu").read_bytes(), b"kernel\r\n")
        self.assertEqual(list(self.work.iterdir()), [first])

    def test_concurrent_snapshot(self):
        sources = {"kernel.cu": "kernel", "include/header.cuh": "header"}
        with ThreadPoolExecutor(max_workers=4) as pool:
            paths = list(pool.map(lambda _: snapshot(self.work, sources), range(8)))
        self.assertEqual(len(set(paths)), 1)
        self.assertEqual(len(list(self.work.iterdir())), 1)
        self.assertEqual((Path(paths[0]) / "include/header.cuh").read_text(), "header")

    def test_rejects_changed_snapshot(self):
        sources = {"kernel.cu": "kernel"}
        root = Path(snapshot(self.work, sources))
        (root / "kernel.cu").write_text("changed")
        with self.assertRaisesRegex(StructuredOutputError, "snapshot"):
            snapshot(self.work, sources)

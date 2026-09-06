import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_action_sandbox import FakeRunner

from klineage.action._sandbox import _Kind, _new


class AuditSourceTests(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        problem = root / "reference.py"
        problem.write_text("def torch_ref(): pass\n")
        self.enterContext(patch("klineage.action._sandbox._project_root", return_value=root))
        self.enterContext(patch("klineage.action._sandbox.CodexRunner", FakeRunner))
        self.sandbox = self.enterContext(_new(_Kind.DECOMPOSE, problem))

    def test_snapshot_freezes_sources(self):
        sandbox = self.sandbox
        (sandbox._work / "kernel.cu").write_text("stale predecessor")
        sources = {"kernel.cu": "current kernel\n", "nested/helper.cuh": "helper\n"}

        first = Path(sandbox._snapshot(sources))
        second = Path(sandbox._snapshot({"kernel.cu": "next kernel\n"}))

        self.assertNotEqual(first, second)
        self.assertFalse(first.is_relative_to(sandbox._work))
        self.assertTrue(first.is_relative_to(sandbox._runner.read_roots[0]))
        self.assertEqual({p.relative_to(first).as_posix(): p.read_text()
                          for p in first.rglob("*") if p.is_file()}, sources)
        self.assertEqual((second / "kernel.cu").read_text(), "next kernel\n")
        self.assertEqual((sandbox._work / "kernel.cu").read_text(), "stale predecessor")

    def test_snapshot_rejects_escape(self):
        for name in ("../outside.cu", "/tmp/outside.cu", "nested/../../outside.cu"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.sandbox._snapshot({name: "kernel"})

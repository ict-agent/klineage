import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from problem_fixtures import io_problem

from klineage.contract import ABIValue, OutputStyle
from klineage.kernel import Kernel

SOURCES = {
    "config.toml": """[solution]
name = "increment"
definition = "increment"
author = "test"
[build]
language = "python"
entry_point = "kernel.py::run"
""",
    "solution/kernel.py": "def run(x): return x + 1\n",
}


class KernelBuildTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.problem = io_problem(
            inputs=(ABIValue("x"),),
            outputs=(ABIValue("result"),),
            language="python",
            platform="cpu",
        )

    def test_build_returns_callable(self):
        current = Kernel.from_sources(SOURCES, self.problem, build_root=self.root)
        self.assertEqual(current(3), 4)
        self.assertEqual(current.language, "python")
        self.assertEqual(current.entry_point, "kernel.py::run")
        self.assertEqual(current.source_path, "solution/kernel.py")
        self.assertEqual(current.symbol, "run")
        self.assertIs(current.output_style, OutputStyle.RETURN)

    def test_restore_requires_build(self):
        current = Kernel.from_sources(SOURCES, self.problem, build_root=self.root)
        saved = current.to_dict()
        self.assertEqual(set(saved), {"name", "problem", "source_files", "validation"})
        restored = Kernel.from_dict(saved)
        self.assertIsNone(restored.function)
        with self.assertRaisesRegex(RuntimeError, "built"):
            restored(3)
        restored.build(self.root)
        self.assertEqual(restored(3), current(3))
        self.assertEqual(restored.fingerprint, current.fingerprint)

    def test_changed_sources_need_build(self):
        current = Kernel.from_sources(SOURCES, self.problem, build_root=self.root)
        changed = replace(
            current,
            source_files={**SOURCES, "solution/kernel.py": "def run(x): return x + 2"},
        )
        self.assertIsNone(changed.function)
        changed.build(self.root)
        self.assertEqual((current(3), changed(3)), (4, 5))

    def test_build_failure_propagates(self):
        sources = {**SOURCES, "solution/kernel.py": "def run(:"}
        with self.assertRaises(SyntaxError):
            Kernel.from_sources(sources, self.problem, build_root=self.root)

import inspect
import json
import tempfile
import unittest
from contextlib import chdir
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kernel_fixtures import make_kernel, write_kernel
from problem_fixtures import io_problem, problem_spec

from klineage.artifact import (
    Kernel,
    Node,
    load_artifact,
    load_problem,
    save_artifact,
)
from klineage.backend import backends
from klineage.backend.cuda import CudaKernel
from klineage.serialization import from_json, to_json

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
            inputs={"x": {}},
            outputs={"result": {}},
            language="python",
            platform="sm120",
        )
        self.current = make_kernel("increment", self.problem, SOURCES)
        self.directory = write_kernel(self.current, self.root / "current_kernel")

    def test_abstract_interfaces(self):
        self.assertEqual(Kernel.__abstractmethods__, {"kernel_spec", "build_kernel"})
        self.assertFalse(hasattr(Kernel, "save_kernel"))
        self.assertEqual(
            set(inspect.signature(CudaKernel.load_kernel).parameters),
            {"build_root", "include_paths"},
        )
        with self.assertRaises(TypeError):
            Kernel(self.current.tree)

    def test_directory_load(self):
        current = CudaKernel.load_kernel(build_root=self.directory)
        self.assertEqual(current(3), 4)
        self.assertEqual(load_problem(current), self.problem)
        self.assertEqual(
            {p.name for p in self.directory.iterdir()}, {"metadata.json", "submission"}
        )
        self.assertNotIn("problem", {node.name for node in current.tree.children})

    def test_default_directory_load(self):
        with chdir(self.root):
            current = Kernel.load_kernel()
        self.assertIsInstance(current, CudaKernel)
        self.assertEqual(current(3), 4)
        self.assertEqual(load_problem(current), self.problem)

    def test_json_is_separate_data(self):
        current = CudaKernel.load_kernel(self.directory)
        path = save_artifact(current, self.root / "snapshot.json")
        restored = load_artifact(path, type(self.current))
        self.assertIsInstance(restored, CudaKernel)
        self.assertIsNone(restored.function)
        self.assertEqual(restored, current)
        self.assertEqual(from_json(to_json(current), type(current)), current)
        self.assertFalse((self.directory / "artifact.json").exists())

    def test_load_changed_files(self):
        current = CudaKernel.load_kernel(self.directory)
        (self.directory / "submission/solution/kernel.py").write_text(
            "def run(x): return x + 2"
        )
        loaded = Kernel.load_kernel(self.directory)
        self.assertEqual((current(3), loaded(3)), (4, 5))

    def test_failed_reload_is_isolated(self):
        current = CudaKernel.load_kernel(self.directory)
        (self.directory / "submission/solution/kernel.py").write_text("def run(:")
        with self.assertRaises(SyntaxError):
            CudaKernel.load_kernel(self.directory)
        self.assertEqual(current(3), 4)

    def test_subtree_type_error(self):
        entry = self.directory / "submission/solution/kernel.py"
        entry.unlink()
        solution = entry.parent
        solution.rmdir()
        solution.write_text("not a directory")
        with (
            patch.object(CudaKernel, "build_kernel") as build,
            self.assertRaisesRegex(
                ValueError, "submission/solution: expected directory"
            ),
        ):
            CudaKernel.load_kernel(self.directory)
        build.assert_not_called()

    def test_config_error_path(self):
        (self.directory / "submission/config.toml").write_text("[broken")
        with self.assertRaisesRegex(ValueError, "submission/config.toml"):
            CudaKernel.load_kernel(self.directory)

    def test_missing_entry_path(self):
        (self.directory / "submission/solution/kernel.py").unlink()
        with self.assertRaisesRegex(
            ValueError, "submission/solution/kernel.py: missing"
        ):
            CudaKernel.load_kernel(self.directory)

    def test_problem_reference_error(self):
        metadata = json.loads((self.directory / "metadata.json").read_text())
        metadata["problem"]["path"] = str(self.root / "missing-problem")
        (self.directory / "metadata.json").write_text(json.dumps(metadata))
        with (
            patch.object(CudaKernel, "build_kernel") as build,
            self.assertRaisesRegex(ValueError, "metadata.json/problem:"),
        ):
            CudaKernel.load_kernel(self.directory)
        build.assert_not_called()

    def test_backend_interfaces(self):
        for backend in backends():
            cls = backend.kernel_type
            with self.subTest(backend=backend.kind):
                self.assertIsInstance(cls.kernel_spec(), Node)
                current = make_kernel(
                    "test",
                    problem_spec(language=backend.language, platform=backend.kind),
                    {cls.kernel_spec().find("submission").find("solution").children[0].name: "// source"},
                )
                directory = write_kernel(current, self.root / backend.kind)
                with patch.object(
                    cls, "build_kernel", return_value=lambda x: x + 1
                ) as build:
                    loaded = cls.load_kernel(directory)
                build.assert_called_once_with(
                    self.root / "workspace/.klineage/build", include_paths=()
                )
                self.assertEqual(loaded(3), 4)

    def test_generic_backend_spec(self):
        class ExtendedKernel(CudaKernel):
            artifact_kind = "kernel.cuda.test_build"

            @classmethod
            def kernel_spec(cls):
                schema = super().kernel_spec()
                return replace(schema, children=(
                    *schema.children, Node.file("notes.txt", description="Backend notes."),
                ))


        notes = self.directory / "notes.txt"
        notes.write_text("platform-specific metadata")
        backend = SimpleNamespace(kernel_type=ExtendedKernel)
        with (
            patch("klineage.backend.get_backend", return_value=backend),
            patch.object(ExtendedKernel, "build_kernel", return_value=lambda x: x + 1),
        ):
            loaded = Kernel.load_kernel(self.directory)
            self.assertIsInstance(loaded, ExtendedKernel)
            self.assertEqual(loaded.find("notes.txt").content, notes.read_text())
            self.assertEqual(loaded(3), 4)
            notes.unlink()
            with self.assertRaisesRegex(ValueError, "notes.txt: missing"):
                Kernel.load_kernel(self.directory)

    def test_wrong_backend_rejected(self):
        from klineage.backend.ascend import AscendKernel

        with self.assertRaisesRegex(ValueError, "metadata.json/problem: target"):
            AscendKernel.load_kernel(self.directory)

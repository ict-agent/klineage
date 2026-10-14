from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from kernel_fixtures import edit_kernel, kernel, make_kernel, write_kernel
from problem_fixtures import io_problem

from klineage.artifact import (
    Kernel,
    Node,
    Problem,
    json_node,
    load_artifact,
    load_problem,
    source_files,
)
from klineage.artifact import kernel as storage
from klineage.artifact.tree import read_source_tree
from klineage.backend import get_backend
from klineage.backend.cuda import CudaKernel
from klineage.backend.runtime import BundleCallable, OutputStyle, bundle_entry, snapshot
from klineage.errors import StructuredOutputError

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
    def test_validation(self):
        original = kernel()
        metadata = json.loads(original.find("metadata.json").content)
        metadata.pop("validation")
        unmeasured = type(original)(original.tree.with_child(json_node("metadata.json", metadata)))
        self.assertIsNone(unmeasured.validation)

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

    def test_directory_load(self):
        current = CudaKernel.load_kernel(build_root=self.directory)
        self.assertEqual(current(3), 4)
        self.assertEqual(load_problem(current), self.problem)
        self.assertEqual(
            {p.name for p in self.directory.iterdir()}, {"metadata.json", "submission"}
        )
        self.assertNotIn("problem", {node.name for node in current.tree.children})

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

    def test_version_order_is_numeric(self):
        self.assertEqual(storage.next_version(self.work), 0)
        with self.assertRaises(FileNotFoundError):
            storage.latest_kernel(self.work)
        for name in ("2", "10", "notes"):
            (self.work / "versions" / name).mkdir(parents=True)
        self.assertEqual(storage.latest_kernel(self.work), self.work / "versions/10")
        self.assertEqual(storage.next_version(self.work), 11)

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


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.bundle = self.root / "submission"
        self.bundle.mkdir()

    def test_preserves_source_text(self):
        source = b"// source\r\nvoid kernel() {}\r\n"
        (self.bundle / "kernel.cu").write_bytes(source)
        self.assertEqual(
            read_source_tree(self.bundle, "bundle")["kernel.cu"].encode(), source
        )

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

    def test_reuses_identical_sources(self):
        sources = {"kernel.cu": "kernel\r\n", "helper.cuh": "header\n"}
        first = Path(snapshot(self.work, sources))
        timestamp = (first / "kernel.cu").stat().st_mtime_ns
        second = Path(snapshot(self.work, dict(reversed(sources.items()))))
        self.assertEqual(first, second)
        self.assertEqual((first / "kernel.cu").stat().st_mtime_ns, timestamp)
        self.assertEqual((first / "kernel.cu").read_bytes(), b"kernel\r\n")
        self.assertEqual(list(self.work.iterdir()), [first])


def workload(rows):
    return {
        "uuid": f"rows-{rows}",
        "axes": {"N": rows, "M": rows + 1},
        "inputs": {name: {"type": "random"} for name in ("x", "y")},
    }


def dynamic_problem(workloads=(), output_axis="N"):
    return Problem(
        "add",
        {
            "name": "add",
            "op_type": "add",
            "axes": {
                "N": {"type": "var"},
                "M": {"type": "var"},
                "C": {"type": "const", "value": 2},
            },
            "inputs": {
                name: {"dtype": "float32", "shape": ["N", "C"]} for name in ("x", "y")
            },
            "outputs": {"out": {"dtype": "float32", "shape": [output_axis, "C"]}},
            "reference": "def run(x, y): return x + y\n",
        },
        workloads,
        language="cuda",
        platform="cuda",
    )


class DynamicABITests(unittest.TestCase):
    def setUp(self):
        self.problem = dynamic_problem()
        self.backend = SimpleNamespace(kind="cuda", device_type="cpu")
        self.enterContext(
            patch("klineage.backend.runtime.get_backend", return_value=self.backend)
        )
        self.enterContext(
            patch("klineage.backend.runtime.load_torch", return_value=torch)
        )
        self.enterContext(
            patch(
                "klineage.backend.runtime.device_runtime",
                return_value=SimpleNamespace(device=lambda device: nullcontext()),
            )
        )

    def test_dynamic_destinations(self):
        def add(x, y, out):
            out.copy_(x + y)

        fn = BundleCallable(self.problem, add, OutputStyle.DESTINATION)
        for rows in (3, 7, 0):
            x, y = torch.ones(rows, 2), torch.ones(rows, 2)
            result = fn(x, y)
            torch.testing.assert_close(result, x + y)

    def test_dynamic_return(self):
        fn = BundleCallable(self.problem, lambda x, y: x + y, OutputStyle.RETURN)
        for rows in (3, 7):
            torch.testing.assert_close(
                fn(torch.ones(rows, 2), torch.ones(rows, 2)), torch.full((rows, 2), 2.0)
            )

    def test_destination_return(self):
        def add(x, y, out):
            out.copy_(x + y)
            return out

        fn = BundleCallable(self.problem, add, OutputStyle.DESTINATION)
        with self.assertRaisesRegex(ValueError, "must return None"):
            fn(torch.ones(3, 2), torch.ones(3, 2))

    def test_destination_changes(self):
        def replace_data(x, y, out):
            out.data = x + y

        def replace_storage(x, y, out):
            out.set_(x + y)

        def change_stride(x, y, out):
            out.as_strided_(out.shape, (1, out.shape[0]))

        def change_shape(x, y, out):
            out.resize_(2, 3)

        def change_dtype(x, y, out):
            out.data = out.to(torch.float64)

        def grow_storage(x, y, out):
            out.untyped_storage().resize_(out.untyped_storage().nbytes() * 2)

        for function in (
            replace_data, replace_storage, change_stride, change_shape,
            change_dtype, grow_storage,
        ):
            with self.subTest(function=function.__name__):
                fn = BundleCallable(self.problem, function, OutputStyle.DESTINATION)
                with self.assertRaisesRegex(ValueError, "output 'out'.*storage or metadata"):
                    fn(torch.ones(3, 2), torch.ones(3, 2))

    def test_empty_storage_rejected(self):
        def replace_storage(x, y, out):
            out.set_(torch.empty_like(out))

        fn = BundleCallable(self.problem, replace_storage, OutputStyle.DESTINATION)
        with self.assertRaisesRegex(ValueError, "output 'out'.*storage or metadata"):
            fn(torch.empty(0, 2), torch.empty(0, 2))

    def test_destination_offset(self):
        out = torch.empty(8)[:6].view(3, 2)

        def change_offset(x, y, out):
            out.as_strided_(out.shape, out.stride(), 1)

        fn = BundleCallable(self.problem, change_offset, OutputStyle.DESTINATION)
        with patch("klineage.backend.runtime.allocate", return_value=out):
            with self.assertRaisesRegex(ValueError, "output 'out'.*storage or metadata"):
                fn(torch.ones(3, 2), torch.ones(3, 2))

    def test_ordered_destinations(self):
        spec = {"shape": [2], "dtype": "float32"}
        problem = io_problem(
            inputs={"z": spec, "a": spec}, outputs={"y": spec, "b": spec},
        )
        destinations = []

        def run(z, a, y, b):
            destinations.extend((y, b))
            y.copy_(z - a)
            b.copy_(z + a)

        fn = BundleCallable(problem, run, OutputStyle.DESTINATION)
        result = fn(torch.full((2,), 3.0), torch.ones(2))
        for actual, destination, expected in zip(result, destinations, (2.0, 4.0)):
            self.assertIs(actual, destination)
            torch.testing.assert_close(actual, torch.full((2,), expected))

    def test_repeated_axis_conflict(self):
        function = Mock()
        fn = BundleCallable(self.problem, function, OutputStyle.RETURN)
        with self.assertRaisesRegex(ValueError, "axis.*N|N.*axis"):
            fn(torch.ones(3, 2), torch.ones(4, 2))
        function.assert_not_called()

    def test_dynamic_input_contract(self):
        invalid = (
            ([1, 2], TypeError, "Tensor"),
            (torch.ones(3, 2, dtype=torch.int64), ValueError, "dtype"),
            (torch.ones(3, 2, 1), ValueError, "shape"),
            (torch.empty(3, 2, device="meta"), ValueError, "cpu"),
        )
        for value, error, message in invalid:
            with self.subTest(message=message):
                function = Mock()
                fn = BundleCallable(self.problem, function, OutputStyle.RETURN)
                with self.assertRaisesRegex(error, message):
                    fn(value, torch.ones(3, 2))
                function.assert_not_called()

    def test_dynamic_output_shape(self):
        fn = BundleCallable(self.problem, lambda x, y: x[:1], OutputStyle.RETURN)
        with self.assertRaisesRegex(ValueError, "output.*shape"):
            fn(torch.ones(3, 2), torch.ones(3, 2))

    def test_explicit_case_binding(self):
        cases = (workload(3), workload(7))
        problem = dynamic_problem(cases)
        fn = BundleCallable(
            problem, lambda x, y: x + y, OutputStyle.RETURN, workload=cases[1]
        )
        torch.testing.assert_close(
            fn(torch.ones(7, 2), torch.ones(7, 2)), torch.full((7, 2), 2.0)
        )
        with self.assertRaisesRegex(ValueError, "input.*shape"):
            fn(torch.ones(3, 2), torch.ones(3, 2))

    def test_case_specific_strides(self):
        def add(x, y, out):
            out.copy_(x + y)

        cases = (workload(3), workload(7))
        for case in cases:
            rows = case["axes"]["N"]
            with self.subTest(rows=rows):
                strides = {("output", "out"): (1, rows)}
                fn = BundleCallable(
                    dynamic_problem(cases),
                    add,
                    OutputStyle.DESTINATION,
                    strides,
                    workload=case,
                )
                result = fn(torch.ones(rows, 2), torch.ones(rows, 2))
                self.assertEqual(result.stride(), (1, rows))
                torch.testing.assert_close(result, torch.full((rows, 2), 2.0))

    def load_python(self, problem, source, root):
        sources = {
            "config.toml": '[solution]\nname="add"\ndefinition="add"\nauthor="test"\n[build]\nlanguage="python"\nentry_point="kernel.py::run"\n',
            "solution/kernel.py": source,
        }
        directory = write_kernel(make_kernel("add", problem, sources), root / "kernel")
        return Kernel.load_kernel(directory)


_CONFIG = """[solution]
name = "gemm"
definition = "bf16 GEMM"
author = "klineage"

[build]
language = "cuda"
entry_point = "kernel.cu::run"
destination_passing_style = true
"""


def source_bundle(config: str = _CONFIG) -> dict[str, str]:
    return {"config.toml": config, "solution/kernel.cu": "void run() {}"}


def kernel_metadata(sources):
    return bundle_entry(sources, get_backend("cuda"))


class CudaContractTests(unittest.TestCase):

    def test_reads_cuda_build(self):
        build = kernel_metadata(source_bundle())

        self.assertEqual(build.language, "cuda")
        self.assertEqual(build.source_path, "solution/kernel.cu")
        self.assertEqual(build.symbol, "run")
        self.assertIs(build.output_style, OutputStyle.DESTINATION)

    def test_python_entry_and_defaults(self):
        config = _CONFIG.replace("kernel.cu::run", "nested/kernel.py::run").replace(
            'language = "cuda"', 'language = "python"'
        )
        config = config.replace("destination_passing_style = true\n", "")
        sources = {
            "config.toml": config,
            "solution/nested/kernel.py": "def run(): pass",
        }
        build = kernel_metadata(sources)

        self.assertEqual(build.source_path, "solution/nested/kernel.py")
        self.assertEqual(build.language, "python")
        self.assertIs(build.output_style, OutputStyle.RETURN)

    def test_cuda_return_style(self):
        for config in (
            _CONFIG.replace(
                "destination_passing_style = true", "destination_passing_style = false"
            ),
            _CONFIG.replace("destination_passing_style = true\n", ""),
        ):
            with self.subTest(config=config):
                build = kernel_metadata(source_bundle(config))
                self.assertIs(build.output_style, OutputStyle.RETURN)

    def test_rejects_invalid_config(self):
        for config in (
            "[solution",
            "",
            _CONFIG.replace("[solution]", "[other]"),
            "solution = []\n" + _CONFIG[_CONFIG.index("[build]") :],
            _CONFIG.replace("[build]", "[other]"),
            _CONFIG.replace('name = "gemm"', "name = 1"),
            _CONFIG.replace('definition = "bf16 GEMM"', 'definition = " "'),
            _CONFIG.replace('author = "klineage"', "author = []"),
            _CONFIG.replace('author = "klineage"\n', ""),
            _CONFIG.replace('language = "cuda"', 'language = "triton"'),
            _CONFIG.replace('language = "cuda"', 'language = "python"'),
            _CONFIG.replace('language = "cuda"', "language = true"),
            _CONFIG.replace('entry_point = "kernel.cu::run"', "entry_point = 1"),
            _CONFIG.replace(
                "destination_passing_style = true", "destination_passing_style = 1"
            ),
            _CONFIG.replace(
                "destination_passing_style = true",
                'destination_passing_style = "false"',
            ),
        ):
            with (
                self.subTest(config=config),
                self.assertRaises((ValueError, TypeError, KeyError)),
            ):
                kernel_metadata(source_bundle(config))

    def test_rejects_bad_entry_paths(self):
        for entry in (
            "../kernel.cu::run",
            "/kernel.cu::run",
            "nested/../kernel.cu::run",
            "C:/kernel.cu::run",
            " kernel.cu::run",
            "kernel.cu::run ",
            "./kernel.cu::run",
            "nested//kernel.cu::run",
            "kernel.cu",
            "kernel.txt::run",
            "kernel.cu::",
            "kernel.cu::nested::run",
            "kernel.cu::bad-name",
            "kernel.cu::run()",
            "kernel.py::class",
        ):
            with (
                self.subTest(entry=entry),
                self.assertRaises((ValueError, TypeError, KeyError)),
            ):
                config = _CONFIG.replace("kernel.cu::run", entry)
                sources = source_bundle(config)
                sources["solution/" + entry.partition("::")[0]] = "entry"
                kernel_metadata(sources)

    def test_requires_entry_source(self):
        with self.assertRaises((ValueError, KeyError)):
            kernel_metadata({"config.toml": _CONFIG})

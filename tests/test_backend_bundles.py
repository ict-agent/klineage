from klineage.artifact.problem import load_problem
from kernel_fixtures import load_entry, edit_kernel, make_kernel
import fcntl
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kernel_fixtures import kernel

from klineage.backend.runtime import BundleLoader
from klineage.backend.runtime import BUILD_LOCK, compile_sources, native_sources
import importlib
from klineage.backend import backends, get_backend


class BackendBundleTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.loader = BundleLoader(self.root)
        self.compiles = []
        for backend in backends():
            self.enterContext(
                patch.object(type(backend), "info", return_value=self.info(backend))
            )
            module = importlib.import_module(type(backend).__module__)
            compiler = (
                "compile_ascend"
                if backend.language == "ascendc"
                else "compile_extension"
            )
            self.enterContext(patch.object(module, compiler, side_effect=self.compile))

    def info(self, backend):
        return {
            "language": backend.language,
            "architecture": "target",
            "compiler": "compiler",
        }

    def compile(self, name, sources, directory, includes, options):
        self.compiles.append((options["language"], name, sources, directory, options))
        return SimpleNamespace(run=lambda x: x + 1)

    def make_kernel(self, backend):
        module = importlib.import_module(type(backend).__module__)
        self.loader = BundleLoader(self.root, compiler=module.compile_native)
        original = kernel()
        problem = load_problem(original).replace(
            language=backend.language, platform=backend.kind
        )
        return make_kernel("test", problem, {("kernel" + backend.kernel_type.entry_suffixes[0]): "// native kernel"})

    def test_native_bundle_backend(self):
        for backend in backends():
            with self.subTest(backend=backend.kind):
                current = self.make_kernel(backend)
                config = (
                    '[solution]\nname="test"\ndefinition="gemm"\nauthor="test"\n'
                    f'[build]\nlanguage="{backend.language}"\n'
                    f'entry_point="{("kernel" + backend.kernel_type.entry_suffixes[0])}::run"\n'
                )
                current = edit_kernel(current, source_files={'config.toml': config, f'solution/{("kernel" + backend.kernel_type.entry_suffixes[0])}': '// entry', 'solution/helper.cpp': '// helper', 'solution/helper.h': '// header'})
                self.assertEqual(load_entry(self.loader, current)(2), 3)
                self.assertEqual(
                    {Path(path).name for path in self.compiles[-1][2]},
                    {("kernel" + backend.kernel_type.entry_suffixes[0]), "helper.cpp"},
                )

    def test_cache_backend_arch(self):
        for backend in backends():
            current = self.make_kernel(backend)
            load_entry(self.loader, current)
            load_entry(self.loader, current)
        self.assertEqual(len(self.compiles), len(backends()))
        self.assertEqual(len({item[1] for item in self.compiles}), len(backends()))
        with patch.object(type(backend), "info", return_value=self.info(backend) | {"architecture": "other"}):
            load_entry(self.loader, current)
        self.assertEqual(len(self.compiles), len(backends()) + 1)

    def test_compiler_preserves_sources(self):
        current = self.make_kernel(get_backend("hip"))

        def compile(name, sources, directory, includes, options):
            # HIP preprocessing can emit files beside its input sources.
            generated = Path(sources[0]).with_name("generated.hip")
            generated.write_text("// compiler output")
            return self.compile(name, sources, directory, includes, options)

        with patch("klineage.backend.hip.kernel.compile_extension", new=compile):
            load_entry(self.loader, current)
            load_entry(self.loader, current)

        self.assertEqual(len(self.compiles), 1)
        self.assertEqual(current.find("submission").find("solution").find("kernel.hip").content, "// native kernel")

    def test_build_lock_lifetime(self):
        with (self.root / BUILD_LOCK).open("a") as contender:
            with (
                native_sources(self.root, {"kernel.hip": "// source"}),
                self.assertRaises(BlockingIOError),
            ):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_source_cache_keeps_mtime(self):
        sources = {"solution/kernel.cu": "// source\r\n"}
        with native_sources(self.root, sources) as root:
            path = root / "solution/kernel.cu"
            os.utime(path, ns=(0, 0))

        with native_sources(self.root, sources):
            self.assertEqual(path.stat().st_mtime_ns, 0)
            self.assertEqual(path.read_bytes(), b"// source\r\n")

        path.write_text("// compiler changed its input")
        with native_sources(self.root, sources):
            self.assertEqual(path.read_bytes(), b"// source\r\n")

    def test_build_names_include_root(self):
        backend = get_backend("cuda")
        current = self.make_kernel(backend)
        load_entry(self.loader, current)
        self.loader.build_root = self.root / "worker"
        load_entry(self.loader, current)
        # PyTorch versions extensions by name across all build directories.
        self.assertNotEqual(self.compiles[0][1], self.compiles[1][1])

    def test_build_error_keeps_log(self):
        current = self.make_kernel(get_backend("cuda"))
        diagnostic = "kernel.cu: invalid conversion\n" + "compiler warning\n" * 200
        error = RuntimeError(diagnostic)
        with (
            patch("klineage.backend.cuda.kernel.compile_extension", side_effect=error),
            self.assertRaises(RuntimeError) as caught,
        ):
            load_entry(self.loader, current)

        logs = list(self.root.rglob("build-error.log"))
        self.assertEqual(len(logs), 1)
        self.assertIn(diagnostic, logs[0].read_text())
        self.assertIs(caught.exception, error)
        self.assertIn(str(logs[0]), "\n".join(error.__notes__))

    def test_python_registers_npu(self):
        current = self.make_kernel(get_backend("ascendc"))
        config = (
            '[solution]\nname="test"\ndefinition="gemm"\nauthor="test"\n'
            '[build]\nlanguage="python"\nentry_point="kernel.py::run"\n'
        )
        current = edit_kernel(current, source_files={'config.toml': config, 'solution/kernel.py': 'import torch\ndef run(x): return x + torch.npu.increment\n'})
        torch = SimpleNamespace()

        def register(backend):
            torch.npu = SimpleNamespace(increment=1)
            return torch

        with (
            patch.dict(sys.modules, {"torch": torch}),
            patch("klineage.backend.runtime.load_torch", side_effect=register),
        ):
            self.assertEqual(load_entry(self.loader, current)(2), 3)


class ExternalIncludeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.headers = self.root / "include"
        self.headers.mkdir()
        self.header = self.headers / "value.h"
        self.header.write_text("1")
        self.modules = {}
        self.calls = 0

    def compile(self, name, sources, directory, includes, options):
        # Model a compiler's own module-name cache, beyond compile_sources.
        if name in self.modules:
            return self.modules[name]
        self.calls += 1
        header = next(
            (Path(root) / "value.h" for root in includes if (Path(root) / "value.h").is_file()),
            None,
        )
        if header is None:
            raise RuntimeError("compiler cannot open value.h")
        value = int(header.read_text())
        module = SimpleNamespace(run=lambda: value)
        self.modules[name] = module
        return module

    def load(self, includes=()):
        sources = {"solution/kernel.cu": '#include "value.h"\n'}
        if not includes:
            sources["solution/value.h"] = "1"
        return compile_sources(
            sources, self.root / "build",
            includes, {"language": "cuda"}, (".cu",), self.compile,
        )

    def test_changed_header_rebuild(self):
        first = self.load((self.headers,))
        timestamp = self.header.stat().st_mtime_ns
        self.header.write_text("2")
        os.utime(self.header, ns=(timestamp, timestamp))
        second = self.load((self.headers,))
        self.assertEqual((first.run(), second.run()), (1, 2))
        self.assertEqual(self.calls, 2)

    def test_deleted_header_rebuild(self):
        self.assertEqual(self.load((self.headers,)).run(), 1)
        self.header.unlink()
        with self.assertRaisesRegex(RuntimeError, "compiler cannot open value.h") as caught:
            self.load((self.headers,))
        self.assertEqual(self.calls, 2)
        self.assertIn("build-error.log", "\n".join(caught.exception.__notes__))

    def test_no_include_keeps_cache(self):
        first = self.load()
        second = self.load()
        self.assertIs(first, second)
        self.assertEqual(second.run(), 1)
        self.assertEqual(self.calls, 1)


class NpuInputTests(unittest.TestCase):
    def test_random_inputs_use_cpu(self):
        import torch

        from klineage.artifact.problem import load_trace, trace_inputs

        module = load_trace(
            Path(__file__).parents[1] / "problems/gemm"
        )
        workload = module.workloads[0] | {"axes": {"M": 2, "N": 2, "K": 2}}
        transferred = torch.ones((2, 2), dtype=torch.bfloat16)
        with (
            patch("klineage.backend.runtime.load_torch", return_value=torch),
            patch.object(torch, "Generator", wraps=torch.Generator) as generator,
            patch.object(torch.Tensor, "to", return_value=transferred) as transfer,
        ):
            result = trace_inputs(module.definition, workload, device="npu:0")
        self.assertEqual(generator.call_args.kwargs["device"], "cpu")
        self.assertEqual(transfer.call_count, 2)
        self.assertIs(result["x"], transferred)

    def test_saved_inputs_use_cpu(self):
        import torch
        from safetensors import safe_open

        from klineage.artifact.problem import (
            trace_definition,
            trace_inputs,
            trace_workload,
        )

        tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3).T
        definition = trace_definition(
            "copy", "copy", "def torch_ref(x): return x", {"x": tensor}, {"out": tensor}
        )
        with tempfile.TemporaryDirectory() as temporary:
            workload = trace_workload(
                {"x": tensor}, Path(temporary) / "inputs.safetensors"
            )
            with (
                patch("klineage.backend.runtime.load_torch", return_value=torch),
                patch.object(torch.Tensor, "to", return_value=tensor),
                patch("safetensors.safe_open", wraps=safe_open) as reader,
            ):
                result = trace_inputs(definition, workload, device="npu:0")
        self.assertEqual(reader.call_args.kwargs["device"], "cpu")
        torch.testing.assert_close(result["x"], tensor, check_stride=True)


if __name__ == "__main__":
    unittest.main()

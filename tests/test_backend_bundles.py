import fcntl
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kernel_fixtures import kernel

from klineage.artifact.bundle import BUILD_LOCK, BundleLoader, native_sources
from klineage.artifact.kernel import Kernel
from klineage.backend import BACKENDS, Backend, get_backend


class BackendBundleTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.loader = BundleLoader(self.root)
        self.compiles = []
        self.enterContext(
            patch.object(Backend, "build_options", return_value={"arch": "target"})
        )
        self.enterContext(
            patch.object(Backend, "compile", autospec=True, side_effect=self.compile)
        )

    def compile(self, backend, name, sources, directory, includes, options):
        self.compiles.append((backend, name, sources, directory, options))
        return SimpleNamespace(launch=lambda *args: None, run=lambda x: x + 1)

    def make_kernel(self, backend):
        original = kernel()
        problem = replace(
            original.problem, language=backend.language, platform=backend.kind
        )
        return Kernel("test", problem, {backend.raw_source: "// native kernel"})

    def test_raw_backend_dispatch(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend.kind):
                current = self.make_kernel(backend)
                self.assertTrue(callable(self.loader.load(current)))
                selected, name, paths, directory, _ = self.compiles[-1]
                self.assertIs(selected, backend)
                self.assertTrue(name.startswith(f"klineage_{backend.kind}_"))
                self.assertEqual(
                    {Path(path).name for path in paths},
                    {backend.raw_source, "binding.cpp"},
                )
                binding = next(
                    Path(path) for path in paths if path.endswith("binding.cpp")
                )
                text = binding.read_text()
                self.assertIn(backend.raw_abi, text)
                self.assertNotIn("@", text)
                self.assertIn("DeviceGuard", text)
                self.assertTrue(directory.is_relative_to(self.root))

    def test_native_bundle_backend(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend.kind):
                current = self.make_kernel(backend)
                config = (
                    '[solution]\nname="test"\ndefinition="gemm"\nauthor="test"\n'
                    f'[build]\nlanguage="{backend.language}"\n'
                    f'entry_point="{backend.raw_source}::run"\n'
                )
                current = replace(
                    current,
                    source_files={
                        "config.toml": config,
                        f"solution/{backend.raw_source}": "// entry",
                        "solution/helper.cpp": "// helper",
                        "solution/helper.h": "// header",
                    },
                )
                self.assertEqual(self.loader.load(current)(2), 3)
                self.assertEqual(
                    {Path(path).name for path in self.compiles[-1][2]},
                    {backend.raw_source, "helper.cpp"},
                )

    def test_cache_backend_arch(self):
        for backend in BACKENDS:
            current = self.make_kernel(backend)
            self.loader.load(current)
            self.loader.load(current)
        self.assertEqual(len(self.compiles), len(BACKENDS))
        self.assertEqual(len({item[1] for item in self.compiles}), len(BACKENDS))
        with patch.object(Backend, "build_options", return_value={"arch": "other"}):
            self.loader.load(current)
        self.assertEqual(len(self.compiles), len(BACKENDS) + 1)

    def test_compiler_preserves_sources(self):
        current = self.make_kernel(get_backend("hip"))

        def compile(backend, name, sources, directory, includes, options):
            # HIP preprocessing can emit files beside its input sources.
            generated = Path(sources[0]).with_name("generated.hip")
            generated.write_text("// compiler output")
            return self.compile(backend, name, sources, directory, includes, options)

        with patch.object(Backend, "compile", new=compile):
            self.loader.load(current)
            self.loader.load(current)

        self.assertEqual(len(self.compiles), 1)
        self.assertEqual(list(self.root.glob("source-*/generated.hip")), [])

    def test_build_lock_lifetime(self):
        with (self.root / BUILD_LOCK).open("a") as contender:
            with (
                native_sources(self.root, {"kernel.hip": "// source"}),
                self.assertRaises(BlockingIOError),
            ):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_python_registers_npu(self):
        current = self.make_kernel(get_backend("ascendc"))
        config = (
            '[solution]\nname="test"\ndefinition="gemm"\nauthor="test"\n'
            '[build]\nlanguage="python"\nentry_point="kernel.py::run"\n'
        )
        current = replace(
            current,
            source_files={
                "config.toml": config,
                "solution/kernel.py": "import torch\ndef run(x): return x + torch.npu.increment\n",
            },
        )
        torch = SimpleNamespace()

        def register():
            torch.npu = SimpleNamespace(increment=1)
            return torch

        with (
            patch.dict(sys.modules, {"torch": torch}),
            patch.object(Backend, "torch", side_effect=register),
        ):
            self.assertEqual(self.loader.load(current)(2), 3)


class NpuInputTests(unittest.TestCase):
    def test_random_inputs_use_cpu(self):
        import torch

        from klineage.artifact.problem import load_trace, trace_inputs

        module = load_trace(
            Path(__file__).parents[1] / "problems/definitions/gemm.json"
        )
        module.workload["axes"] = {"M": 2, "N": 2, "K": 2}
        transferred = torch.ones((2, 2), dtype=torch.bfloat16)
        with (
            patch.object(Backend, "torch", return_value=torch),
            patch.object(torch, "Generator", wraps=torch.Generator) as generator,
            patch.object(torch.Tensor, "to", return_value=transferred) as transfer,
        ):
            result = trace_inputs(module.definition, module.workload, device="npu:0")
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
                patch.object(Backend, "torch", return_value=torch),
                patch.object(torch.Tensor, "to", return_value=tensor),
                patch("safetensors.safe_open", wraps=safe_open) as reader,
            ):
                result = trace_inputs(definition, workload, device="npu:0")
        self.assertEqual(reader.call_args.kwargs["device"], "cpu")
        torch.testing.assert_close(result["x"], tensor, check_stride=True)


if __name__ == "__main__":
    unittest.main()

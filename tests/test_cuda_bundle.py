from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from klineage.contract import KernelABI, OutputStyle, ProblemSpec
from klineage.errors import StructuredOutputError
from klineage.harness._cuda_bundle import _BundleLoader
from klineage.kernel import Kernel, TargetContext


_CONFIG = '''[solution]
name = "example"
definition = "example"
author = "klineage"
[build]
language = "{language}"
entry_point = "{entry}"
destination_passing_style = {dps}
'''
_CUDA = '''#include <tvm/ffi/function.h>
__global__ void unused() {{}}
int run(int value) {{ return value + {increment}; }}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
'''
_HAS_NATIVE = all(importlib.util.find_spec(name) for name in ("torch", "tvm_ffi"))


class _BundleCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.loader = _BundleLoader(self.root / "build")
        self.counter = 0

    def kernel(self, sources, entry="binding.py::run", dps="false", language=None):
        self.counter += 1
        root = self.root / str(self.counter)
        language = language or ("python" if entry.endswith(".py::run") else "cuda")
        files = {"config.toml": _CONFIG.format(entry=entry, dps=dps, language=language),
                 **{"solution/" + name: value for name, value in sources.items()}}
        for name, source in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        return Kernel(
            name="example", source=files["solution/" + entry.split("::")[0]],
            source_files=files, artifact_path=root, abi=KernelABI(),
            context=TargetContext("example", "cuda", "sm120"),
        )


class BundleTests(_BundleCase):
    def test_python_entry_skips_loader(self):
        kernel = self.kernel({"binding.py": "def load(): raise AssertionError()\ndef run(x): return x + 1\n"})
        function, build = self.loader.load(kernel)

        self.assertEqual(function(3), 4)
        self.assertIs(build.output_style, OutputStyle.RETURN)
        self.assertFalse((kernel.artifact_path / "solution/__pycache__").exists())
        self.assertEqual(self.loader.load(kernel)[0](4), 5)

    def test_python_module_isolation(self):
        foreign = types.ModuleType("bundle_helper")
        foreign.VALUE = 99
        self.enterContext(patch.dict(sys.modules, {"bundle_helper": foreign}))
        first = self.kernel({"binding.py": "import bundle_helper\ndef run(): return bundle_helper.VALUE\n",
                             "bundle_helper.py": "VALUE = 1\n"})
        second = self.kernel({"binding.py": "import bundle_helper\ndef run(): return bundle_helper.VALUE\n",
                              "bundle_helper.py": "VALUE = 2\n"})

        left, _ = self.loader.load(first)
        right, _ = self.loader.load(second)

        self.assertEqual((left(), right(), left()), (1, 2, 1))
        self.assertIs(sys.modules["bundle_helper"], foreign)

    def test_python_destination_entry(self):
        kernel = self.kernel({"binding.py": "def run(x, output): output.append(x)\n"}, dps="true")
        function, build = self.loader.load(kernel)
        output = []
        self.assertIsNone(function(7, output))
        self.assertEqual(output, [7])
        self.assertIs(build.output_style, OutputStyle.DESTINATION)

    def test_bad_symbol(self):
        for source in ("other = 1", "run = 1"):
            with self.subTest(source=source), self.assertRaises((AttributeError, TypeError)):
                self.loader.load(self.kernel({"binding.py": source}))

    def test_artifact_matches_sources(self):
        kernel = self.kernel({"binding.py": "def run(): return 1"})
        (kernel.artifact_path / "solution/binding.py").write_text("raise AssertionError('executed tampered source')")
        with self.assertRaisesRegex(ValueError, "match"):
            self.loader.load(kernel)

    def test_extra_file(self):
        kernel = self.kernel({"binding.py": "def run(): return 1"})
        (kernel.artifact_path / "solution/extra.py").write_text("unexpected = 1")
        with self.assertRaisesRegex(ValueError, "match"):
            self.loader.load(kernel)

    def test_symlink_is_rejected(self):
        kernel = self.kernel({"binding.py": "def run(): return 1"})
        path = kernel.artifact_path / "solution/binding.py"
        target = self.root / "outside.py"
        target.write_text(path.read_text())
        path.unlink()
        path.symlink_to(target)
        with self.assertRaises((ValueError, StructuredOutputError)):
            self.loader.load(kernel)


@unittest.skipUnless(_HAS_NATIVE, "Torch and TVM FFI are unavailable")
class NativeBundleTests(_BundleCase):
    def setUp(self):
        super().setUp()
        self.enterContext(patch.dict(os.environ, {"TVM_FFI_CUDA_ARCH_LIST": "12.0"}))

    def test_native_sources_and_cache(self):
        first = self.kernel({"kernel.cu": _CUDA.format(increment=1), "helper.cpp": "int helper() { return 0; }",
                             "helper.cuh": "// header"}, entry="kernel.cu::run", dps="true")
        second = self.kernel({"kernel.cu": _CUDA.format(increment=2)}, entry="kernel.cu::run")
        with patch("tvm_ffi.cpp.load", return_value=types.SimpleNamespace(run=lambda x: x)) as compile:
            function, build = self.loader.load(first)
            self.assertEqual(function(2), 2)
            self.assertIs(build.output_style, OutputStyle.DESTINATION)
            self.loader.load(second)
            with patch.dict(os.environ, {"TVM_FFI_CUDA_ARCH_LIST": "9.0"}):
                self.loader.load(second)

        calls = [call.kwargs for call in compile.call_args_list]
        self.assertEqual(len({call["name"] for call in calls}), 3)
        self.assertEqual({Path(p).suffix for p in calls[0]["sources"]}, {".cu", ".cpp"})
        self.assertTrue(Path(calls[0]["build_directory"]).is_relative_to(self.root / "build"))
        self.assertFalse(Path(calls[0]["build_directory"]).is_relative_to(first.artifact_path))
        self.assertTrue(any("torch/include" in p for p in calls[0]["extra_include_paths"]))
        self.assertIn("-lc10_cuda", calls[0]["extra_ldflags"])

    def test_cuda_python_autocompiles(self):
        source = '''#include <torch/library.h>
__global__ void unused() {}
int64_t increment(int64_t value) { return value + 1; }
TORCH_LIBRARY(klineage_bundle_auto, module) {
  module.def("increment(int value) -> int", increment);
}
'''
        binding = '''import torch
native = torch.ops.klineage_bundle_auto.increment
def run(value): return native(value)
'''
        kernel = self.kernel({"kernel.cu": source, "binding.py": binding}, language="cuda")

        function, _ = self.loader.load(kernel)

        self.assertEqual(function(8), 9)
        copied = self.kernel({"kernel.cu": source, "binding.py": binding.replace("def run", "def other")},
                             entry="binding.py::other", language="cuda")
        other, _ = self.loader.load(copied)
        self.assertEqual(other(9), 10)

    def test_reuses_native_copies(self):
        sources = {"kernel.cu": _CUDA.format(increment=1), "value.cuh": "#define VALUE 1",
                   "binding.py": "def run(value): return value + 1"}
        first = self.kernel(sources, language="cuda")
        changed_binding = {**sources, "binding.py": "def other(value): return value + 2"}
        second = self.kernel(changed_binding, entry="binding.py::other", language="cuda")
        second = replace(second, problem=ProblemSpec("other", "Other contract."))
        header_change = self.kernel({**sources, "value.cuh": "#define VALUE 2"}, language="cuda")

        with patch("tvm_ffi.cpp.load", return_value=types.SimpleNamespace()) as compile:
            left, _ = self.loader.load(first)
            right, _ = self.loader.load(second)
            self.assertEqual((left(3), right(3)), (4, 5))
            self.assertEqual(compile.call_count, 1)
            self.loader.load(header_change)
            self.assertEqual(compile.call_count, 2)

    def test_build_uses_hashed_arch(self):
        kernel = self.kernel({"kernel.cu": _CUDA.format(increment=1)}, entry="kernel.cu::run")
        observed = []

        def compile(**kwargs):
            observed.append((kwargs["name"], os.environ.get("TVM_FFI_CUDA_ARCH_LIST")))
            return types.SimpleNamespace(run=lambda value: value)

        with patch.dict(os.environ), patch("tvm_ffi.cpp.load", side_effect=compile):
            os.environ.pop("TVM_FFI_CUDA_ARCH_LIST", None)
            for architecture in ((9, 0), (12, 0)):
                with patch("torch.cuda.get_device_capability", return_value=architecture):
                    self.loader.load(kernel)
                self.assertNotIn("TVM_FFI_CUDA_ARCH_LIST", os.environ)

        self.assertEqual([item[1] for item in observed], ["9.0", "12.0"])
        self.assertNotEqual(observed[0][0], observed[1][0])

    def test_arch_restored_on_error(self):
        kernel = self.kernel({"kernel.cu": _CUDA.format(increment=1)}, entry="kernel.cu::run")

        def compile(**kwargs):
            self.assertEqual(os.environ.get("TVM_FFI_CUDA_ARCH_LIST"), "9.0")
            raise RuntimeError("compiler failed")

        with patch.dict(os.environ), patch("tvm_ffi.cpp.load", side_effect=compile):
            os.environ.pop("TVM_FFI_CUDA_ARCH_LIST", None)
            with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
                with self.assertRaisesRegex(RuntimeError, "compiler failed"):
                    self.loader.load(kernel)
            self.assertNotIn("TVM_FFI_CUDA_ARCH_LIST", os.environ)

    def test_cuda_uses_torch_abi(self):
        kernel = self.kernel({"kernel.cu": _CUDA.format(increment=1)}, entry="kernel.cu::run")
        with patch("torch._C._GLIBCXX_USE_CXX11_ABI", False):
            with patch("tvm_ffi.cpp.load", return_value=types.SimpleNamespace(run=lambda value: value)) as compile:
                self.loader.load(kernel)
        for flags in ("extra_cflags", "extra_cuda_cflags"):
            self.assertIn("-D_GLIBCXX_USE_CXX11_ABI=0", compile.call_args.kwargs[flags])

    def test_cuda_python_needs_native(self):
        kernel = self.kernel({"binding.py": "def run(): return 1"}, language="cuda")
        with self.assertRaisesRegex(ValueError, "native source"):
            self.loader.load(kernel)

    def test_native_build_failure(self):
        kernel = self.kernel({"kernel.cu": _CUDA.format(increment=1)}, entry="kernel.cu::run")
        with patch("tvm_ffi.cpp.load", side_effect=RuntimeError("compiler failed")):
            with self.assertRaisesRegex(RuntimeError, "compiler failed"):
                self.loader.load(kernel)

    def test_native_missing_symbol(self):
        kernel = self.kernel({"kernel.cu": _CUDA.format(increment=1)}, entry="kernel.cu::run")
        with patch("tvm_ffi.cpp.load", return_value=types.SimpleNamespace()):
            with self.assertRaises(AttributeError):
                self.loader.load(kernel)

    def test_real_native_load_on_cpu(self):
        kernel = self.kernel({"kernel.cu": _CUDA.format(increment=1)}, entry="kernel.cu::run")
        function, _ = self.loader.load(kernel)
        self.assertEqual(function(8), 9)


if __name__ == "__main__":
    unittest.main()

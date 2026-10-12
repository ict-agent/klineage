from __future__ import annotations

import importlib.util
import os
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from kernel_fixtures import load_entry
from runtime_fixtures import BundleCase

from klineage.backend.cuda.kernel import compile_native

_CUDA = """#include <pybind11/pybind11.h>
__global__ void unused() {{}}
int run(int value) {{ return value + {increment}; }}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {{ module.def("run", &run); }}
"""
_HAS_NATIVE = importlib.util.find_spec("torch") is not None
RUN_CUDA = os.environ.get("KLINEAGE_CUDA_TESTS") == "1"
THREAD_WAIT_SECONDS = 1


class BundleTests(BundleCase):
    def test_python_entry_skips_loader(self):
        kernel = self.kernel(
            {
                "binding.py": "def load(): raise AssertionError()\ndef run(x): return x + 1\n"
            }
        )
        function = load_entry(self.loader, kernel)

        self.assertEqual(function(3), 4)
        self.assertEqual(list(self.loader.build_root.rglob("__pycache__")), [])
        self.assertEqual(load_entry(self.loader, kernel)(4), 5)

    def test_python_module_isolation(self):
        foreign = types.ModuleType("bundle_helper")
        foreign.VALUE = 99
        self.enterContext(patch.dict(sys.modules, {"bundle_helper": foreign}))
        first = self.kernel(
            {
                "binding.py": "import bundle_helper\ndef run(): return bundle_helper.VALUE\n",
                "bundle_helper.py": "VALUE = 1\n",
            }
        )
        second = self.kernel(
            {
                "binding.py": "import bundle_helper\ndef run(): return bundle_helper.VALUE\n",
                "bundle_helper.py": "VALUE = 2\n",
            }
        )

        left = load_entry(self.loader, first)
        right = load_entry(self.loader, second)

        self.assertEqual((left(), right(), left()), (1, 2, 1))
        self.assertIs(sys.modules["bundle_helper"], foreign)

    def test_python_lazy_imports(self):
        foreign = types.ModuleType("bundle_helper")
        foreign.VALUE = 99
        self.enterContext(patch.dict(sys.modules, {"bundle_helper": foreign}))
        functions = []
        for value in (1, 2):
            current = self.kernel(
                {
                    "binding.py": "def run():\n    import bundle_helper\n    return bundle_helper.VALUE\n",
                    "bundle_helper.py": f"VALUE = {value}\n",
                }
            )
            functions.append(load_entry(self.loader, current))

        self.assertEqual([function() for function in functions], [1, 2])
        self.assertEqual(functions[0](), 1)
        self.assertIs(sys.modules["bundle_helper"], foreign)
        self.assertEqual(list(self.loader.build_root.rglob("__pycache__")), [])

    def test_python_destination_entry(self):
        kernel = self.kernel(
            {"binding.py": "def run(x, output): output.append(x)\n"}, dps="true"
        )
        function = load_entry(self.loader, kernel)
        output = []
        self.assertIsNone(function(7, output))
        self.assertEqual(output, [7])


@unittest.skipUnless(_HAS_NATIVE, "PyTorch is unavailable")
class NativeBundleTests(BundleCase):
    def setUp(self):
        super().setUp()
        self.enterContext(patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "12.0"}))
        self.enterContext(patch("torch.cuda.get_device_properties", return_value=types.SimpleNamespace()))

    def test_native_sources_and_cache(self):
        first = self.kernel(
            {
                "kernel.cu": _CUDA.format(increment=1),
                "helper.cpp": "int helper() { return 0; }",
                "helper.cuh": "// header",
            },
            entry="kernel.cu::run",
            dps="true",
        )
        second = self.kernel(
            {"kernel.cu": _CUDA.format(increment=2)}, entry="kernel.cu::run"
        )
        with patch(
            "torch.utils.cpp_extension.load",
            return_value=types.SimpleNamespace(run=lambda x: x),
        ) as compile:
            function = load_entry(self.loader, first)
            self.assertEqual(function(2), 2)
            load_entry(self.loader, second)
            with patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "9.0"}):
                load_entry(self.loader, second)

        calls = [call.kwargs for call in compile.call_args_list]
        self.assertEqual(len({call["name"] for call in calls}), 3)
        self.assertEqual({Path(p).suffix for p in calls[0]["sources"]}, {".cu", ".cpp"})
        self.assertTrue(
            Path(calls[0]["build_directory"]).is_relative_to(self.root / "build")
        )
        self.assertNotIn(
            Path(calls[0]["build_directory"]),
            [Path(p).parent for p in calls[0]["sources"]],
        )
        self.assertEqual(len(calls[0]["extra_include_paths"]), 1)
        self.assertTrue(calls[0]["with_cuda"])
        self.assertNotIn("backend", calls[0])

    def test_parallel_arch_isolation(self):
        from klineage.backend.cuda import CudaBackend

        first_entered, second_info = threading.Event(), threading.Event()
        observed, failures = {}, []
        original_info = CudaBackend.info

        def info(backend):
            result = original_info(backend)
            if threading.current_thread().name == "second":
                second_info.set()
            return result

        def compile(**options):
            name = threading.current_thread().name
            observed[name] = os.environ.get("TORCH_CUDA_ARCH_LIST")
            if name == "first":
                first_entered.set()
                second_info.wait(THREAD_WAIT_SECONDS)
            return types.SimpleNamespace(run=lambda value: value)

        def build():
            try:
                compile_native({"solution/kernel.cu": "source"}, self.root / "parallel")
            except Exception as error:
                failures.append(error)

        first = threading.Thread(target=build, name="first")
        second = threading.Thread(target=build, name="second")
        with (
            patch.dict(os.environ),
            patch.object(CudaBackend, "info", new=info),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch(
                "torch.cuda.get_device_capability",
                side_effect=lambda: (
                    (9, 0) if threading.current_thread().name == "first" else (12, 0)
                ),
            ),
            patch("torch.utils.cpp_extension.load", side_effect=compile),
        ):
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            first.start()
            started = first_entered.wait(THREAD_WAIT_SECONDS)
            second.start()
            first.join(THREAD_WAIT_SECONDS * 3)
            second.join(THREAD_WAIT_SECONDS * 3)
            self.assertNotIn("TORCH_CUDA_ARCH_LIST", os.environ)
        self.assertTrue(started)
        self.assertFalse(first.is_alive() or second.is_alive())
        if failures:
            raise failures[0]
        self.assertEqual(observed, {"first": "9.0", "second": "12.0"})

    @unittest.skipUnless(RUN_CUDA, "CUDA toolchain tests are opt-in")
    def test_real_native_load_on_cpu(self):
        kernel = self.kernel(
            {"kernel.cu": _CUDA.format(increment=1)}, entry="kernel.cu::run"
        )
        function = load_entry(self.loader, kernel)
        self.assertEqual(function(8), 9)


if __name__ == "__main__":
    unittest.main()

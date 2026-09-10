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

from problem_fixtures import problem_spec

from klineage.artifact.kernel import Kernel
from klineage.contract import OutputStyle
from klineage.harness.artifacts import BundleLoader

_CONFIG = """[solution]
name = "example"
definition = "example"
author = "klineage"
[build]
language = "{language}"
entry_point = "{entry}"
destination_passing_style = {dps}
"""
_CUDA = """#include <pybind11/pybind11.h>
__global__ void unused() {{}}
int run(int value) {{ return value + {increment}; }}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {{ module.def("run", &run); }}
"""
_HAS_NATIVE = importlib.util.find_spec("torch") is not None


class BundleCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.loader = BundleLoader(self.root / "build")
        self.counter = 0

    def kernel(self, sources, entry="binding.py::run", dps="false", language=None):
        self.counter += 1
        root = self.root / str(self.counter)
        language = language or ("python" if entry.endswith(".py::run") else "cuda")
        files = {
            "config.toml": _CONFIG.format(entry=entry, dps=dps, language=language),
            **{"solution/" + name: value for name, value in sources.items()},
        }
        for name, source in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        return Kernel(
            name="example",
            source_files=files,
            problem=problem_spec("example", "cuda", "sm120"),
        )


class BundleTests(BundleCase):
    def test_python_entry_skips_loader(self):
        kernel = self.kernel(
            {
                "binding.py": "def load(): raise AssertionError()\ndef run(x): return x + 1\n"
            }
        )
        function = self.loader.load(kernel)

        self.assertEqual(function(3), 4)
        self.assertIs(kernel.output_style, OutputStyle.RETURN)
        self.assertEqual(list(self.loader.build_root.rglob("__pycache__")), [])
        self.assertEqual(self.loader.load(kernel)(4), 5)

    def test_loads_from_saved_text(self):
        kernel = self.kernel({"binding.py": "def run(): return 7"})
        import shutil

        shutil.rmtree(self.root / str(self.counter))
        restored = Kernel.from_dict(kernel.to_dict())
        self.assertEqual(self.loader.load(restored)(), 7)

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

        left = self.loader.load(first)
        right = self.loader.load(second)

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
            functions.append(self.loader.load(current))

        self.assertEqual([function() for function in functions], [1, 2])
        self.assertEqual(functions[0](), 1)
        self.assertIs(sys.modules["bundle_helper"], foreign)
        self.assertEqual(list(self.loader.build_root.rglob("__pycache__")), [])

    def test_namespace_isolation(self):
        foreign = types.ModuleType("bundle_helper")
        foreign.__path__ = [str(self.root / "foreign")]
        value = types.ModuleType("bundle_helper.values")
        value.VALUE = 99
        foreign.values = value
        self.enterContext(
            patch.dict(
                sys.modules,
                {"bundle_helper": foreign, "bundle_helper.values": value},
            )
        )
        functions = [
            self.loader.load(
                self.kernel(
                    {
                        "binding.py": (
                            "def run():\n"
                            "    from bundle_helper import values\n"
                            "    return values.VALUE\n"
                        ),
                        "bundle_helper/values.py": f"VALUE = {increment}\n",
                    }
                )
            )
            for increment in (1, 2)
        ]
        self.assertEqual([function() for function in functions], [1, 2])
        self.assertEqual(functions[0](), 1)
        self.assertIs(sys.modules["bundle_helper"], foreign)
        self.assertIs(sys.modules["bundle_helper.values"], value)
        sys.modules.pop("bundle_helper")
        sys.modules.pop("bundle_helper.values")
        self.assertEqual(functions[0](), 1)
        self.assertNotIn("bundle_helper", sys.modules)
        self.assertNotIn("bundle_helper.values", sys.modules)

    def test_python_destination_entry(self):
        kernel = self.kernel(
            {"binding.py": "def run(x, output): output.append(x)\n"}, dps="true"
        )
        function = self.loader.load(kernel)
        output = []
        self.assertIsNone(function(7, output))
        self.assertEqual(output, [7])
        self.assertIs(kernel.output_style, OutputStyle.DESTINATION)

    def test_bad_symbol(self):
        for source in ("other = 1", "run = 1"):
            with (
                self.subTest(source=source),
                self.assertRaises((AttributeError, TypeError)),
            ):
                self.loader.load(self.kernel({"binding.py": source}))


@unittest.skipUnless(_HAS_NATIVE, "PyTorch is unavailable")
class NativeBundleTests(BundleCase):
    def setUp(self):
        super().setUp()
        self.enterContext(patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "12.0"}))

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
            function = self.loader.load(first)
            self.assertEqual(function(2), 2)
            self.assertIs(first.output_style, OutputStyle.DESTINATION)
            self.loader.load(second)
            with patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "9.0"}):
                self.loader.load(second)

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

    def test_reuses_native_copies(self):
        sources = {
            "kernel.cu": _CUDA.format(increment=1),
            "value.cuh": "#define VALUE 1",
        }
        first = self.kernel(sources, entry="kernel.cu::run")
        second = replace(
            first, problem=problem_spec("other", description="Other contract.")
        )
        header_change = self.kernel(
            {**sources, "value.cuh": "#define VALUE 2"}, entry="kernel.cu::run"
        )
        with patch(
            "torch.utils.cpp_extension.load",
            return_value=types.SimpleNamespace(run=lambda x: x + 1),
        ) as compile:
            left = self.loader.load(first)
            right = self.loader.load(second)
            self.assertEqual((left(3), right(3)), (4, 4))
            self.assertEqual(compile.call_count, 1)
            self.loader.load(header_change)
            self.assertEqual(compile.call_count, 2)

    def test_build_uses_hashed_arch(self):
        kernel = self.kernel(
            {"kernel.cu": _CUDA.format(increment=1)}, entry="kernel.cu::run"
        )
        observed = []

        def compile(**kwargs):
            observed.append((kwargs["name"], os.environ.get("TORCH_CUDA_ARCH_LIST")))
            return types.SimpleNamespace(run=lambda value: value)

        with (
            patch.dict(os.environ),
            patch("torch.utils.cpp_extension.load", side_effect=compile),
        ):
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            for architecture in ((9, 0), (12, 0)):
                with patch(
                    "torch.cuda.get_device_capability", return_value=architecture
                ):
                    self.loader.load(kernel)
                self.assertNotIn("TORCH_CUDA_ARCH_LIST", os.environ)

        self.assertEqual([item[1] for item in observed], ["9.0", "12.0"])
        self.assertNotEqual(observed[0][0], observed[1][0])

    def test_arch_restored_on_error(self):
        kernel = self.kernel(
            {"kernel.cu": _CUDA.format(increment=1)}, entry="kernel.cu::run"
        )

        def compile(**kwargs):
            self.assertEqual(os.environ.get("TORCH_CUDA_ARCH_LIST"), "9.0")
            raise RuntimeError("compiler failed")

        with (
            patch.dict(os.environ),
            patch("torch.utils.cpp_extension.load", side_effect=compile),
        ):
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            with (
                patch("torch.cuda.get_device_capability", return_value=(9, 0)),
                self.assertRaisesRegex(RuntimeError, "compiler failed"),
            ):
                self.loader.load(kernel)
            self.assertNotIn("TORCH_CUDA_ARCH_LIST", os.environ)

    def test_native_missing_symbol(self):
        kernel = self.kernel(
            {"kernel.cu": _CUDA.format(increment=1)}, entry="kernel.cu::run"
        )
        with (
            patch(
                "torch.utils.cpp_extension.load", return_value=types.SimpleNamespace()
            ),
            self.assertRaises(AttributeError),
        ):
            self.loader.load(kernel)

    def test_real_native_load_on_cpu(self):
        kernel = self.kernel(
            {"kernel.cu": _CUDA.format(increment=1)}, entry="kernel.cu::run"
        )
        function = self.loader.load(kernel)
        self.assertEqual(function(8), 9)

    def test_real_raw_binding_loads(self):
        source = """#include <cuda_runtime_api.h>
extern "C" cudaError_t klineage_launch(
    const void* const*, void* const*, cudaStream_t) { return cudaSuccess; }
"""
        current = Kernel(
            "raw-binding",
            problem_spec("example", "cuda", "sm120"),
            {"kernel.cu": source},
        )
        self.assertTrue(callable(self.loader.load(current)))


if __name__ == "__main__":
    unittest.main()

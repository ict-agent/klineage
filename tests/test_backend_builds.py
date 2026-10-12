from __future__ import annotations

import importlib
import os
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kernel_fixtures import edit_kernel, kernel, load_entry, make_kernel

from klineage.artifact.problem import load_problem
from klineage.backend import (
    BackendKind,
    backends,
    detect_backend,
    get_backend,
    register_backend,
)
from klineage.backend.ascend import ASCEND_ARCH_ENV
from klineage.backend.ascend.kernel import check_storage, compile_ascend
from klineage.backend.ascend.kernel import compile_native as compile_npu
from klineage.backend.hip import HIP_ARCH_ENV
from klineage.backend.hip.kernel import CFLAGS, compile_extension
from klineage.backend.runtime import BundleLoader, compile_sources
from klineage.serialization import from_json, to_json

HIP_ARCH = "gfx928"
ASCEND_ARCH = "dav-2201"
ASCEND_BASE_FORMATS = {"NCHW": 0, "NHWC": 1, "ND": 2, "NCDHW": 30}
ASCEND_OTHER_FORMATS = {
    "UNDEFINED": -1,
    "NC1HWC0": 3,
    "FRACTAL_Z": 4,
    "NDHWC": 27,
    "FRACTAL_NZ": 29,
    "NDC1HWC0": 32,
    "FRACTAL_Z_3D": 33,
}


class BackendBuildTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.properties = SimpleNamespace(gcnArchName=f"{HIP_ARCH}:xnack-", name="DCU")
        self.runtime = SimpleNamespace(
            is_available=Mock(return_value=True),
            current_device=Mock(return_value=0),
            get_device_properties=Mock(return_value=self.properties),
        )
        self.torch = SimpleNamespace(
            __version__="2.6.0",
            version=SimpleNamespace(hip="6.2", cuda=None),
            _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=False),
            cuda=self.runtime,
        )
        self.extension = SimpleNamespace(
            IS_HIP_EXTENSION=True,
            ROCM_HOME=str(self.root / "dtk"),
            COMMON_HIP_FLAGS=["-D__HIP_PLATFORM_AMD__=1", "-DUSE_ROCM=1"],
            include_paths=lambda: [str(self.root / "torch/include")],
            library_paths=lambda: [str(self.root / "torch/lib")],
            load=Mock(return_value=object()),
        )
        self.npu = SimpleNamespace(
            __file__=str(self.root / "torch_npu/__init__.py"),
            __version__="2.6.0",
        )
        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "torch": self.torch,
                    "torch_npu": self.npu,
                    "torch.utils.cpp_extension": self.extension,
                },
            )
        )
        self.hip = get_backend("hip")
        self.ascend = get_backend("ascendc")

    def ascend_options(self):
        os.environ[ASCEND_ARCH_ENV] = ASCEND_ARCH
        os.environ["ASCEND_HOME_PATH"] = str(self.root / "cann")
        with patch(
            "klineage.backend.ascend.kernel.shutil.which", return_value=str(self.root / "bisheng")
        ):
            return self.ascend.info() | {"cflags": CFLAGS}

    def test_cuda_info(self):
        import json

        self.torch.version.hip = None
        self.torch.version.cuda = "13.0"
        self.runtime.get_device_capability = Mock(return_value=(9, 0))
        self.extension.CUDA_HOME = str(self.root / "cuda")
        with patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "8.0+PTX"}):
            info = get_backend("cuda").info()
        self.assertEqual(info["device_architecture"], "sm90")
        self.assertEqual(info["architecture"], "8.0+PTX")
        self.assertEqual(info["platform"], "nvidia-sm90-cuda13")
        self.assertEqual(info["torch_version"], "2.6.0")
        self.assertEqual(info["device"], "cuda:0")
        self.assertTrue(info["available"])
        self.extension.load.assert_not_called()
        json.dumps(info)

    def test_cuda_device_limits(self):
        self.torch.version.hip = None
        self.torch.version.cuda = "13.0"
        self.runtime.get_device_capability = Mock(return_value=(12, 0))
        self.extension.CUDA_HOME = str(self.root / "cuda")
        limits = {
            "multi_processor_count": 188,
            "warp_size": 32,
            "max_threads_per_block": 1024,
            "max_threads_per_multi_processor": 1536,
            "regs_per_multiprocessor": 65536,
            "shared_memory_per_block": 49152,
            "shared_memory_per_block_optin": 101376,
            "shared_memory_per_multiprocessor": 102400,
            "L2_cache_size": 134217728,
            "total_memory": 101973819392,
        }
        self.runtime.get_device_properties.return_value = SimpleNamespace(
            name="Blackwell", **limits
        )
        self.assertEqual(get_backend("cuda").info()["device_limits"], limits)

    def test_ascend_info_without_arch(self):
        import json

        info = self.ascend.info()
        self.assertIsNone(info["architecture"])
        self.assertFalse(info["available"])
        self.assertEqual(info["runtime_version"], self.npu.__version__)
        self.extension.load.assert_not_called()
        json.dumps(info)

    def test_ascend_base_formats(self):
        tensor = object()
        self.npu.get_npu_format = Mock()
        for name, value in ASCEND_BASE_FORMATS.items():
            with self.subTest(format=name):
                self.npu.get_npu_format.return_value = value
                check_storage(tensor)
                self.npu.get_npu_format.assert_called_with(tensor)

    def test_ascend_other_formats(self):
        tensor = object()
        self.npu.get_npu_format = Mock()
        for name, value in ASCEND_OTHER_FORMATS.items():
            with self.subTest(format=name):
                self.npu.get_npu_format.return_value = value
                with self.assertRaisesRegex(
                    ValueError, "base storage format.*cast to ND"
                ):
                    check_storage(tensor)

    def test_hip_detects_device_arch(self):
        options = self.hip.info() | {"extra_cflags": (*CFLAGS, *self.extension.COMMON_HIP_FLAGS), "extra_cuda_cflags": CFLAGS}
        self.assertEqual(options["architecture"], HIP_ARCH)
        self.assertEqual(options["compiler"], str(self.root / "dtk/bin/hipcc"))
        self.assertEqual(options["runtime_version"], self.torch.version.hip)
        self.assertNotIn(HIP_ARCH_ENV, os.environ)

    def test_hip_rejects_cuda_torch(self):
        self.torch.version.hip = None
        with self.assertRaisesRegex(RuntimeError, "HIP-enabled"):
            self.hip.info()
        self.extension.load.assert_not_called()

    def test_hip_compile_restores_env(self):
        options = self.hip.info() | {"extra_cflags": (*CFLAGS, *self.extension.COMMON_HIP_FLAGS), "extra_cuda_cflags": CFLAGS}
        sources = [str(self.root / "kernel.hip"), str(self.root / "binding.cpp")]
        includes = [str(self.root / "include")]
        seen = []

        def compile(**kwargs):
            seen.append(os.environ[HIP_ARCH_ENV])
            self.assertEqual(kwargs["sources"], sources)
            self.assertEqual(kwargs["extra_include_paths"], includes)
            self.assertTrue(kwargs["with_cuda"])
            self.assertFalse(
                any(flag.startswith("--expt") for flag in kwargs["extra_cuda_cflags"])
            )
            if kwargs["name"] == "failure":
                raise RuntimeError("hipcc failed")
            return self.extension

        self.extension.load.side_effect = compile
        for previous in (None, "gfx936"):
            with self.subTest(previous=previous):
                if previous is None:
                    os.environ.pop(HIP_ARCH_ENV, None)
                else:
                    os.environ[HIP_ARCH_ENV] = previous
                result = compile_extension(
                    "success", sources, self.root, includes, options
                )
                self.assertIs(result, self.extension)
                self.assertEqual(os.environ.get(HIP_ARCH_ENV), previous)
                with self.assertRaisesRegex(RuntimeError, "hipcc failed"):
                    compile_extension("failure", sources, self.root, includes, options)
                self.assertEqual(os.environ.get(HIP_ARCH_ENV), previous)

        self.assertEqual(seen, [HIP_ARCH] * len(seen))

    def test_ascend_requires_arch(self):
        with self.assertRaisesRegex(RuntimeError, ASCEND_ARCH_ENV):
            compile_npu({"kernel.asc": "source"}, self.root)

    def test_ascend_compile_dispatch(self):
        self.torch._C._PYBIND11_COMPILER_TYPE = "_gcc"
        self.torch._C._PYBIND11_STDLIB = "_libstdcpp"
        self.torch._C._PYBIND11_BUILD_ABI = "_cxxabi1011"
        options = self.ascend_options()
        os.environ[ASCEND_ARCH_ENV] = "dav-3510"
        sources = [str(self.root / "kernel.asc"), str(self.root / "binding.cpp")]
        include = str(self.root / "source include")
        module = SimpleNamespace(run=lambda: None)
        loader = Mock()
        spec = SimpleNamespace(loader=loader)

        def compile(command, **kwargs):
            Path(command[command.index("-o") + 1]).write_text("compiled module")
            return SimpleNamespace(returncode=0)

        with (
            patch("klineage.backend.ascend.kernel.subprocess.run", side_effect=compile) as run,
            patch(
                "klineage.backend.ascend.kernel.importlib.util.spec_from_file_location",
                return_value=spec,
            ) as load_spec,
            patch(
                "klineage.backend.ascend.kernel.importlib.util.module_from_spec", return_value=module
            ),
        ):
            result = compile_ascend(
                "example", sources, self.root, [include], options
            )

        self.assertIs(result, module)
        loader.exec_module.assert_called_once_with(module)
        command = run.call_args.args[0]
        self.assertEqual(command[0], options["compiler"])
        self.assertIn(f"--npu-arch={ASCEND_ARCH}", command)
        self.assertEqual(command[command.index("-x") + 1], "asc")
        self.assertIn(f"-I{include}", command)
        self.assertIn("-D_GLIBCXX_USE_CXX11_ABI=0", command)
        self.assertIn("-DTORCH_EXTENSION_NAME=example", command)
        for field in ("COMPILER_TYPE", "STDLIB", "BUILD_ABI"):
            value = getattr(self.torch._C, f"_PYBIND11_{field}")
            self.assertIn(f'-DPYBIND11_{field}="{value}"', command)
        for library in ("torch_python", "torch_cpu", "c10", "torch_npu", "ascendcl"):
            self.assertIn(f"-l{library}", command)
        for source in sources:
            self.assertIn(source, command)
        output = self.root / f"example{sysconfig.get_config_var('EXT_SUFFIX')}"
        self.assertEqual(load_spec.call_args.args, ("example", output))
        self.assertEqual(output.read_text(), "compiled module")
        self.assertTrue((self.root / "build.log").is_file())
        self.extension.load.assert_not_called()

    def test_ascend_compile_error(self):
        options = self.ascend_options()

        def compile(command, **kwargs):
            kwargs["stdout"].write("error: invalid AscendC source\n")
            return SimpleNamespace(returncode=1)

        with (
            patch("klineage.backend.ascend.kernel.subprocess.run", side_effect=compile),
            patch(
                "klineage.backend.ascend.kernel.importlib.util.spec_from_file_location"
            ) as load_spec,
            self.assertRaisesRegex(RuntimeError, "invalid AscendC source"),
        ):
            compile_ascend("example", ["invalid.asc"], self.root, [], options)
        self.assertIn("invalid AscendC source", (self.root / "build.log").read_text())
        load_spec.assert_not_called()


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


class KernelPortTests(unittest.TestCase):

    def test_new_kernel_registration(self):
        from dataclasses import replace

        from kernel_fixtures import kernel

        from klineage.artifact import Kernel, Node
        from klineage.backend.cuda import CudaBackend, CudaKernel
        from klineage.backend.registry import REGISTERED_BACKENDS

        class ExampleKernel(CudaKernel):
            artifact_kind = "kernel.example"

            @classmethod
            def kernel_spec(cls):
                spec = super().kernel_spec()
                return replace(
                    spec,
                    children=(
                        *spec.children,
                        Node.file("notes.txt", description="Backend notes."),
                    ),
                )

        class ExampleBackend(CudaBackend):
            kernel_type = ExampleKernel

            def __init__(self):
                super().__init__()
                object.__setattr__(self, "kind", "example")
                object.__setattr__(self, "language", "example")


        self.addCleanup(REGISTERED_BACKENDS.remove, backend := ExampleBackend())
        register_backend(backend)
        tree = Node.directory(".", kernel().tree.children)
        with self.assertRaisesRegex(ValueError, "notes.txt"):
            ExampleKernel(tree)
        value = ExampleKernel(tree.with_child(Node.file("notes.txt", "example")))
        restored = from_json(to_json(value), Kernel)
        self.assertIsInstance(restored, ExampleKernel)
        self.assertEqual(restored, value)
        self.assertEqual(restored.find("notes.txt").description, "")
        empty = tree.with_child(Node.directory("submission"))
        self.assertIsInstance(ExampleKernel(empty), ExampleKernel)
        python = tree.with_child(Node.directory("submission", (
            Node.file("config.toml", "configuration"),
            Node.directory("solution", (Node.file("kernel.py", "def run(): pass"),)),
        )))
        self.assertIsInstance(ExampleKernel(python.with_child(Node.file("notes.txt", "notes"))), ExampleKernel)


class BackendTests(unittest.TestCase):
    def test_native_selection(self):
        for language, platform, kind, device in (
            ("cuda", "nvidia-sm90-cuda13", BackendKind.CUDA, "cuda"),
            ("hip", "hygon-gfx928-hip6", BackendKind.HYGON, "cuda"),
            ("ascendc", "ascend-910b", BackendKind.ASCEND, "npu"),
        ):
            with self.subTest(language=language):
                backend = get_backend(language, platform)
                self.assertEqual(backend.kind, kind)
                self.assertEqual(backend.device_type, device)
                self.assertIs(get_backend("python", platform), backend)

    def test_rejects_target_mismatch(self):
        for language, platform in (
            ("hip", "nvidia-sm90"),
            ("cuda", "hygon-gfx928"),
            ("ascendc", "nvidia-sm90"),
            ("python", "unknown"),
        ):
            with (
                self.subTest(language=language, platform=platform),
                self.assertRaises(ValueError),
            ):
                get_backend(language, platform)

    def test_detection_uses_info(self):
        cuda, hip, ascend = backends()
        with (
            patch.object(type(cuda), "info", side_effect=RuntimeError("wrong torch")),
            patch.object(type(hip), "info", return_value={"available": True}),
            patch.object(type(ascend), "info", return_value={"available": False}),
        ):
            self.assertIs(detect_backend(), hip)
        with (
            patch.object(type(cuda), "info", return_value={"available": False}),
            patch.object(type(hip), "info", return_value={"available": False}),
            patch.object(type(ascend), "info", return_value={"available": False}),
            self.assertRaises(RuntimeError),
        ):
            detect_backend()

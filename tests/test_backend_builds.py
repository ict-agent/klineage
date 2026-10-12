import os
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from klineage.backend import get_backend
from klineage.backend.hip.kernel import CFLAGS
from klineage.backend.hip.kernel import compile_extension
from klineage.backend.ascend.kernel import compile_ascend
from klineage.backend.ascend.kernel import check_storage, compile_native as compile_npu
from klineage.backend.hip.kernel import compile_native as compile_hip
from klineage.backend.ascend import ASCEND_ARCH_ENV
from klineage.backend.hip import HIP_ARCH_ENV

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

    def test_ascend_unknown_format(self):
        self.npu.get_npu_format = Mock(
            return_value=max(ASCEND_OTHER_FORMATS.values()) + 1
        )
        with self.assertRaisesRegex(ValueError, "base storage format"):
            check_storage(object())

    def test_hip_detects_device_arch(self):
        options = self.hip.info() | {"extra_cflags": (*CFLAGS, *self.extension.COMMON_HIP_FLAGS), "extra_cuda_cflags": CFLAGS}
        self.assertEqual(options["architecture"], HIP_ARCH)
        self.assertEqual(options["compiler"], str(self.root / "dtk/bin/hipcc"))
        self.assertEqual(options["runtime_version"], self.torch.version.hip)
        self.assertNotIn(HIP_ARCH_ENV, os.environ)

    def test_hip_explicit_archs(self):
        architectures = f"{HIP_ARCH};gfx936"
        os.environ[HIP_ARCH_ENV] = architectures
        options = self.hip.info() | {"extra_cflags": (*CFLAGS, *self.extension.COMMON_HIP_FLAGS), "extra_cuda_cflags": CFLAGS}
        self.assertEqual(options["architecture"], architectures)
        self.assertEqual(options["device_architecture"], HIP_ARCH)

    def test_hip_configures_dtk(self):
        os.environ["DTK_HOME"] = str(self.root / "custom-dtk")
        self.hip.info()
        self.assertEqual(os.environ["ROCM_HOME"], os.environ["DTK_HOME"])

    def test_hip_rejects_cuda_torch(self):
        self.torch.version.hip = None
        with self.assertRaisesRegex(RuntimeError, "HIP-enabled"):
            self.hip.info()
        self.extension.load.assert_not_called()

    def test_cuda_rejects_hip_torch(self):
        with self.assertRaisesRegex(RuntimeError, "HIP-enabled"):
            get_backend("cuda").info()

    def test_hip_rejects_stale_import(self):
        self.extension.IS_HIP_EXTENSION = False
        with self.assertRaisesRegex(RuntimeError, "before importing"):
            self.hip.info()

    def test_hip_abi_changes_options(self):
        original = self.hip.info()
        self.torch._C._GLIBCXX_USE_CXX11_ABI = True
        changed = self.hip.info()
        self.assertNotEqual(original, changed)

    def test_hip_host_runtime_flags(self):
        compile_hip({"binding.cpp": "binding", "kernel.hip": "source"}, self.root)
        flags = self.extension.load.call_args.kwargs["extra_cflags"]
        self.assertTrue(set(self.extension.COMMON_HIP_FLAGS).issubset(flags))

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

    def test_ascend_compiler_in_cann(self):
        os.environ[ASCEND_ARCH_ENV] = ASCEND_ARCH
        os.environ["ASCEND_HOME_PATH"] = str(self.root)
        compiler = self.root / "tools/bisheng_compiler/bin/bisheng"
        compiler.parent.mkdir(parents=True)
        compiler.touch()
        compiler.chmod(0o755)
        with patch("klineage.backend.ascend.kernel.shutil.which", return_value=None):
            options = self.ascend.info()
        self.assertEqual(options["compiler"], str(compiler))

    def test_ascend_compiler_missing(self):
        os.environ[ASCEND_ARCH_ENV] = ASCEND_ARCH
        os.environ["ASCEND_HOME_PATH"] = str(self.root)
        with (
            patch("klineage.backend.ascend.kernel.shutil.which", return_value=None),
            self.assertRaisesRegex(RuntimeError, "bisheng"),
        ):
            compile_npu({"kernel.asc": "source"}, self.root)

    def test_ascend_cache_inputs(self):
        original = self.ascend_options()
        self.assertEqual(original["architecture"], ASCEND_ARCH)
        self.torch._C._GLIBCXX_USE_CXX11_ABI = True
        self.assertNotEqual(original, self.ascend_options())
        self.torch._C._GLIBCXX_USE_CXX11_ABI = False
        self.npu.__version__ = "2.6.1"
        self.assertNotEqual(original, self.ascend_options())

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

    def test_ascend_keeps_built_module(self):
        options = self.ascend_options()
        output = self.root / f"example{sysconfig.get_config_var('EXT_SUFFIX')}"
        output.write_text("loaded module")

        def compile(command, **kwargs):
            Path(command[command.index("-o") + 1]).write_text("partial output")
            return SimpleNamespace(returncode=1)

        with (
            patch("klineage.backend.ascend.kernel.subprocess.run", side_effect=compile),
            self.assertRaises(RuntimeError),
        ):
            compile_ascend("example", ["invalid.asc"], self.root, [], options)

        self.assertEqual(output.read_text(), "loaded module")


if __name__ == "__main__":
    unittest.main()

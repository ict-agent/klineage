import os
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from klineage.backend import ASCEND_ARCH_ENV, HIP_ARCH_ENV, get_backend

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
            "klineage.backend.shutil.which", return_value=str(self.root / "bisheng")
        ):
            return self.ascend.build_options()

    def test_ascend_base_formats(self):
        tensor = object()
        self.npu.get_npu_format = Mock()
        for name, value in ASCEND_BASE_FORMATS.items():
            with self.subTest(format=name):
                self.npu.get_npu_format.return_value = value
                self.ascend.validate_tensor(tensor)
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
                    self.ascend.validate_tensor(tensor)

    def test_ascend_unknown_format(self):
        self.npu.get_npu_format = Mock(
            return_value=max(ASCEND_OTHER_FORMATS.values()) + 1
        )
        with self.assertRaisesRegex(ValueError, "base storage format"):
            self.ascend.validate_tensor(object())

    def test_hip_detects_device_arch(self):
        options = self.hip.build_options()
        self.assertEqual(options["architecture"], HIP_ARCH)
        self.assertEqual(options["compiler"], str(self.root / "dtk/bin/hipcc"))
        self.assertEqual(options["runtime_version"], self.torch.version.hip)
        self.assertNotIn(HIP_ARCH_ENV, os.environ)

    def test_hip_explicit_archs(self):
        architectures = f"{HIP_ARCH};gfx936"
        os.environ[HIP_ARCH_ENV] = architectures
        options = self.hip.build_options()
        self.assertEqual(options["architecture"], architectures)
        self.runtime.get_device_properties.assert_not_called()

    def test_hip_configures_dtk(self):
        os.environ["DTK_HOME"] = str(self.root / "custom-dtk")
        self.hip.build_options()
        self.assertEqual(os.environ["ROCM_HOME"], os.environ["DTK_HOME"])

    def test_hip_rejects_cuda_torch(self):
        self.torch.version.hip = None
        with self.assertRaisesRegex(RuntimeError, "HIP-enabled"):
            self.hip.build_options()
        self.extension.load.assert_not_called()

    def test_cuda_rejects_hip_torch(self):
        with self.assertRaisesRegex(RuntimeError, "HIP-enabled"):
            get_backend("cuda").build_options()

    def test_hip_rejects_stale_import(self):
        self.extension.IS_HIP_EXTENSION = False
        with self.assertRaisesRegex(RuntimeError, "before importing"):
            self.hip.build_options()

    def test_hip_abi_changes_options(self):
        original = self.hip.build_options()
        self.torch._C._GLIBCXX_USE_CXX11_ABI = True
        changed = self.hip.build_options()
        self.assertNotEqual(original, changed)

    def test_hip_host_runtime_flags(self):
        options = self.hip.build_options()
        self.hip.compile(
            "example", ["binding.cpp", "kernel.hip"], self.root, [], options
        )
        flags = self.extension.load.call_args.kwargs["extra_cflags"]
        self.assertTrue(set(self.extension.COMMON_HIP_FLAGS).issubset(flags))

    def test_hip_compile_restores_env(self):
        options = self.hip.build_options()
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
                result = self.hip.compile(
                    "success", sources, self.root, includes, options
                )
                self.assertIs(result, self.extension)
                self.assertEqual(os.environ.get(HIP_ARCH_ENV), previous)
                with self.assertRaisesRegex(RuntimeError, "hipcc failed"):
                    self.hip.compile("failure", sources, self.root, includes, options)
                self.assertEqual(os.environ.get(HIP_ARCH_ENV), previous)

        self.assertEqual(seen, [HIP_ARCH] * len(seen))

    def test_ascend_requires_arch(self):
        with self.assertRaisesRegex(RuntimeError, ASCEND_ARCH_ENV):
            self.ascend.build_options()

    def test_ascend_compiler_in_cann(self):
        os.environ[ASCEND_ARCH_ENV] = ASCEND_ARCH
        os.environ["ASCEND_HOME_PATH"] = str(self.root)
        compiler = self.root / "tools/bisheng_compiler/bin/bisheng"
        compiler.parent.mkdir(parents=True)
        compiler.touch()
        compiler.chmod(0o755)
        with patch("klineage.backend.shutil.which", return_value=None):
            options = self.ascend.build_options()
        self.assertEqual(options["compiler"], str(compiler))

    def test_ascend_compiler_missing(self):
        os.environ[ASCEND_ARCH_ENV] = ASCEND_ARCH
        os.environ["ASCEND_HOME_PATH"] = str(self.root)
        with (
            patch("klineage.backend.shutil.which", return_value=None),
            self.assertRaisesRegex(RuntimeError, "bisheng"),
        ):
            self.ascend.build_options()

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
            patch("klineage.backend.subprocess.run", side_effect=compile) as run,
            patch(
                "klineage.backend.importlib.util.spec_from_file_location",
                return_value=spec,
            ) as load_spec,
            patch(
                "klineage.backend.importlib.util.module_from_spec", return_value=module
            ),
        ):
            result = self.ascend.compile(
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
            patch("klineage.backend.subprocess.run", side_effect=compile),
            patch(
                "klineage.backend.importlib.util.spec_from_file_location"
            ) as load_spec,
            self.assertRaisesRegex(RuntimeError, "build.log"),
        ):
            self.ascend.compile("example", ["invalid.asc"], self.root, [], options)
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
            patch("klineage.backend.subprocess.run", side_effect=compile),
            self.assertRaises(RuntimeError),
        ):
            self.ascend.compile("example", ["invalid.asc"], self.root, [], options)

        self.assertEqual(output.read_text(), "loaded module")


if __name__ == "__main__":
    unittest.main()

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from klineage.backend import get_backend
from klineage.harness import AgentRuntime


class LocalRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.devices = self.root / "dev"
        self.devices.mkdir()
        self.toolkit = self.root / "vendor"
        (self.toolkit / "bin").mkdir(parents=True)
        self.compiler = self.toolkit / "bin" / "compiler"
        self.compiler.touch()
        self.compiler.chmod(0o755)
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.enterContext(
            patch("klineage.backend.runtime.DEVICE_ROOT", self.devices, create=True)
        )

    def node(self, name):
        path = self.devices / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to("/dev/null")
        return path

    def local(self, language="cuda", **info):
        backend = get_backend(language)
        metadata = {"available": True, "compiler": str(self.compiler)} | info
        with patch.object(type(backend), "info", return_value=metadata):
            return AgentRuntime.local(backend)

    def test_device_families(self):
        names = {
            "cuda": (
                "nvidia0",
                "nvidia2",
                "nvidiactl",
                "nvidia-uvm",
                "nvidia-uvm-tools",
                "nvidia-caps/nvidia-cap2",
            ),
            "hip": ("kfd", "dri/renderD128", "dri/renderD130"),
            "ascendc": (
                "davinci0",
                "davinci3",
                "davinci_manager",
                "devmm_svm",
                "hisi_hdc",
            ),
        }
        for group in names.values():
            for name in group:
                self.node(name)
        (self.devices / "nvidia9").touch()
        (self.devices / "davinci7").mkdir()
        self.node("unrelated")

        for language, expected in names.items():
            with self.subTest(language=language):
                runtime = self.local(language)
                self.assertEqual(
                    set(runtime.device_nodes),
                    {self.devices / name for name in expected},
                )
                self.assertEqual(runtime.python, Path(sys.executable).absolute())

    def test_visibility_and_secrets(self):
        self.node("nvidia2")
        with patch.dict(
            os.environ,
            {
                "CUDA_VISIBLE_DEVICES": "2",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "KLINEAGE_BACKEND": "ascend",
                "AIPING_API_KEY": "secret",
                "LD_PRELOAD": "/host/hook.so",
                "PYTHONPATH": "/host/private",
                "UNRELATED_SETTING": "private",
                "HOME": str(self.root),
            },
        ):
            runtime = self.local()
        self.assertEqual(
            dict(runtime.environment),
            {
                "CUDA_VISIBLE_DEVICES": "2",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "KLINEAGE_BACKEND": "cuda",
                "PATH": f"{self.compiler.parent}:/usr/bin:/bin",
            },
        )

    def test_compiler_is_on_path(self):
        self.node("nvidia0")
        for path in ("/usr/bin:/bin", f"{self.compiler.parent}:/usr/bin:/bin"):
            with self.subTest(path=path), patch.dict(os.environ, {"PATH": path}):
                runtime = self.local()
                search = runtime.environment["PATH"]
                self.assertEqual(shutil.which(self.compiler.name, path=search), str(self.compiler))
                self.assertEqual(search.split(os.pathsep).count(str(self.compiler.parent)), 1)
                self.assertTrue(search.endswith("/usr/bin:/bin"))

    def test_ascend_architecture(self):
        self.node("davinci0")
        with patch.dict(os.environ, {"ASCEND_ARCH": "dav-2201"}):
            runtime = self.local("ascendc", architecture="dav-2201")
        self.assertEqual(runtime.environment["ASCEND_ARCH"], "dav-2201")

    def test_external_toolkit_paths(self):
        self.node("nvidia0")
        libraries = self.root / "vendor-libraries"
        libraries.mkdir()
        with patch.dict(
            os.environ,
            {
                "PATH": f"{self.toolkit / 'bin'}:/usr/bin:/bin",
                "LD_LIBRARY_PATH": str(libraries),
            },
        ):
            runtime = self.local()
        self.assertEqual(set(runtime.read_only), {self.toolkit, libraries})

    def test_toolkit_symlink(self):
        self.node("nvidia0")
        alias = self.root / "toolkit-alias"
        alias.symlink_to(self.toolkit, target_is_directory=True)
        with patch.dict(os.environ, {"CUDA_HOME": str(alias)}):
            runtime = self.local(compiler=str(alias / "bin/compiler"))
        self.assertEqual(set(runtime.read_only), {alias, self.toolkit})

    def test_system_link_to_external(self):
        self.node("nvidia0")
        system = self.root / "system"
        system.mkdir()
        alias = system / "toolkit"
        alias.symlink_to(self.toolkit, target_is_directory=True)
        with patch("klineage.harness.sandbox.SYSTEM_RUNTIME", (system,)):
            runtime = self.local(compiler=str(alias / "bin/compiler"))
        self.assertEqual(runtime.read_only, (self.toolkit,))

    def test_symlink_to_home_is_hidden(self):
        self.node("nvidia0")
        alias = self.root / "private-alias"
        alias.symlink_to(Path.home(), target_is_directory=True)
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": str(alias)}):
            runtime = self.local()
        self.assertEqual(runtime.read_only, (self.toolkit,))

    def test_skips_private_and_broad_paths(self):
        self.node("nvidia0")
        unsafe = (
            str(Path.home()),
            str(Path.cwd()),
            str(Path.cwd().parent),
            "/dev",
            "/proc",
            "/sys",
            "/",
            "/tmp",
            ".",
        )
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": ":".join(unsafe)}):
            runtime = self.local()
        self.assertEqual(runtime.read_only, (self.toolkit,))

    def test_system_paths_are_implicit(self):
        self.node("nvidia0")
        with patch.dict(
            os.environ, {"PATH": "/usr/bin:/bin", "LD_LIBRARY_PATH": "/usr/lib:/lib"}
        ):
            runtime = self.local(compiler="/usr/bin/c++", toolkit="/usr/local")
        self.assertEqual(runtime.read_only, ())

    def test_unavailable_backend(self):
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            self.local(available=False)

    def test_missing_device_nodes(self):
        with self.assertRaisesRegex(RuntimeError, "device nodes"):
            self.local()

    def test_unreadable_nodes_omitted(self):
        allowed = self.node("nvidia0")
        denied = self.node("nvidia1")
        access = os.access
        with patch(
            "klineage.backend.runtime.os.access",
            side_effect=lambda path, mode: path != denied and access(path, mode),
        ):
            runtime = self.local()
        self.assertEqual(runtime.device_nodes, (allowed,))


if __name__ == "__main__":
    unittest.main()

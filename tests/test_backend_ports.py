import importlib
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from klineage.backend import get_backend, register_backend
from klineage.serialization import from_json, to_json


class BackendPortTests(unittest.TestCase):
    def test_required_modules(self):
        from klineage.backend import Backend
        from klineage.backend.timing import TimingPolicy

        self.assertEqual(Backend.__abstractmethods__, {"bench", "info"})
        for package, language in (("cuda", "cuda"), ("ascend", "ascendc"), ("hip", "hip")):
            with self.subTest(backend=package):
                kernel = importlib.import_module(f"klineage.backend.{package}.kernel")
                timing = importlib.import_module(f"klineage.backend.{package}.timing")
                backend = get_backend(language)
                self.assertEqual(type(backend).__module__, kernel.__name__)
                self.assertIs(backend.timing, timing)
                self.assertIs(backend.bench, timing.bench)
                self.assertIsInstance(timing.DEFAULT_POLICY, TimingPolicy)
                for name in (
                    "compile",
                    "timer",
                    "build_options",
                    "build_config",
                    "compile_impl",
                    "platform_info",
                    "artifact_format",
                    "load_torch",
                ):
                    self.assertFalse(hasattr(backend, name))

    def test_missing_timing_rejected(self):
        from klineage.backend.cuda import CudaBackend

        class MissingTiming(CudaBackend):
            timing = object()

        with self.assertRaisesRegex(TypeError, "timing"):
            register_backend(MissingTiming())

    def test_timing_name_required(self):
        from klineage.backend import registry as drivers
        from klineage.backend.cuda import CudaBackend, CudaKernel

        class ExampleKernel(CudaKernel):
            artifact_kind = "kernel.invalid-timing"

        class ExampleBackend(CudaBackend):
            kind = "invalid-timing"
            language = "invalid-timing"
            kernel_type = ExampleKernel

        for name in (None, "", " ", 7):
            backend = ExampleBackend()
            backend.timing = SimpleNamespace(
                DEFAULT_POLICY=CudaBackend.timing.DEFAULT_POLICY,
                check_timing=CudaBackend.timing.check_timing,
            )
            if name is not None:
                backend.timing.NAME = name
            with (
                self.subTest(name=name),
                patch.object(drivers, "REGISTERED_BACKENDS", list(drivers.backends())),
                self.assertRaisesRegex(TypeError, "timing"),
            ):
                register_backend(backend)


class KernelPortTests(unittest.TestCase):
    def test_independent_imports(self):
        for module in (
            "klineage.backend",
            "klineage.artifact",
            "klineage.backend.cuda",
            "klineage.backend.ascend",
        ):
            with self.subTest(module=module):
                script = (
                    f"import importlib, sys; importlib.import_module({module!r}); "
                    "assert not {'torch', 'torch_npu', 'flashinfer', 'cupti'} & sys.modules.keys()"
                )
                result = subprocess.run(
                    [sys.executable, "-c", script], capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_kernel_rejected(self):
        from klineage.backend.cuda import CudaBackend

        class MissingKernel(CudaBackend):
            kernel_type = object

        with self.assertRaisesRegex(TypeError, "kernel"):
            register_backend(MissingKernel())

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
        self.assertEqual(restored.node("notes.txt").description, "")
        empty = tree.with_child(Node.directory("submission"))
        self.assertIsInstance(ExampleKernel(empty), ExampleKernel)
        python = tree.with_child(Node.directory("submission", (
            Node.file("config.toml", "configuration"),
            Node.directory("solution", (Node.file("kernel.py", "def run(): pass"),)),
        )))
        self.assertIsInstance(ExampleKernel(python.with_child(Node.file("notes.txt", "notes"))), ExampleKernel)

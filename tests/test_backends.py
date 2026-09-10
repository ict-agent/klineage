import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from kernel_fixtures import kernel

from klineage.artifact.kernel import Kernel
from klineage.backend import BACKENDS, BackendKind, detect_backend, get_backend


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

    def test_detects_hip_as_hygon(self):
        torch = SimpleNamespace(
            version=SimpleNamespace(hip="6.2", cuda=None),
            cuda=SimpleNamespace(is_available=lambda: True),
        )
        with patch.dict(sys.modules, {"torch": torch}):
            self.assertEqual(detect_backend().kind, BackendKind.HYGON)
            with self.assertRaisesRegex(RuntimeError, "HIP"):
                get_backend("cuda").runtime()

    def test_no_hardware_fallback(self):
        torch = SimpleNamespace(
            version=SimpleNamespace(hip=None, cuda="13.0"),
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        with patch.dict(sys.modules, {"torch": torch, "torch_npu": None}):
            with self.assertRaises(RuntimeError):
                detect_backend()
            with self.assertRaisesRegex(RuntimeError, "HIP"):
                get_backend("hip").runtime()

    def test_raw_backend_sources(self):
        original = kernel()
        for backend in BACKENDS:
            with self.subTest(backend=backend.kind):
                problem = replace(
                    original.problem, language=backend.language, platform=backend.kind
                )
                current = Kernel("example", problem, {backend.raw_source: "// source"})
                self.assertEqual(current.source_path, backend.raw_source)
                self.assertEqual(current.symbol, "klineage_launch")
                self.assertEqual(Kernel.from_dict(current.to_dict()), current)


if __name__ == "__main__":
    unittest.main()

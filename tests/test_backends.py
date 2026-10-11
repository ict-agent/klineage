from klineage.serialization import from_json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from kernel_fixtures import kernel, make_kernel

from klineage.artifact.problem import load_problem
from klineage.backend import BackendKind, backends, detect_backend, get_backend
from klineage.backend.runtime import load_torch
from klineage.serialization import to_json


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

    def test_runtime_matches_torch(self):
        for language, hip in (("cuda", "6.2"), ("hip", None)):
            torch = SimpleNamespace(version=SimpleNamespace(hip=hip))
            with (
                self.subTest(language=language),
                patch.dict(sys.modules, {"torch": torch}),
                self.assertRaisesRegex(RuntimeError, "HIP"),
            ):
                load_torch(get_backend(language))

    def test_raw_backend_sources(self):
        original = kernel()
        for backend in backends():
            with self.subTest(backend=backend.kind):
                problem = load_problem(original).replace(
                    language=backend.language, platform=backend.kind
                )
                current = make_kernel(
                    "example", problem, {("kernel" + backend.kernel_type.entry_suffixes[0]): "// source"}
                )
                self.assertEqual(current.find("submission").find("solution").find("kernel" + backend.kernel_type.entry_suffixes[0]).content, "// source")
                self.assertEqual(from_json(to_json(current), type(current)), current)


if __name__ == "__main__":
    unittest.main()

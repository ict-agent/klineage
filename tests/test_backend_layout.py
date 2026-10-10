import unittest
from unittest.mock import patch

from klineage.artifact import Node
from klineage.artifact import kernel as kernel_module
from klineage.backend import get_backend
from klineage.backend.ascend import AscendBackend
from klineage.backend.cuda import CudaBackend
from klineage.backend.hip import HipBackend


class BackendLayoutTests(unittest.TestCase):
    def test_layout_ignores_base_order(self):
        base = kernel_module.KERNEL
        reordered = Node.directory(
            base.name, reversed(base.children), description=base.description
        )
        for backend_type in (CudaBackend, AscendBackend, HipBackend):
            kernel_type = backend_type.kernel_type
            expected = kernel_type.kernel_spec()
            with self.subTest(backend=backend_type.__name__), patch(
                f"{kernel_type.__module__}.KERNEL", reordered
            ):
                self.assertEqual(kernel_type.kernel_spec(), expected)

    def test_backend_owns_resources(self):
        for language, cls in (
            ("cuda", CudaBackend),
            ("ascendc", AscendBackend),
            ("hip", HipBackend),
        ):
            with self.subTest(language=language):
                backend = get_backend(language)
                self.assertIsInstance(backend, cls)
                self.assertTrue((backend.skill_root / "SKILL.md").is_file())
                self.assertEqual(backend.kernel_type.artifact_kind, "kernel." + language)
                self.assertIn("backend", backend.skill_root.parts)

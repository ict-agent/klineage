from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from klineage.artifact.repository import stage_repository


class RepositoryTests(unittest.TestCase):

    def test_local_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "kernel.cu").write_bytes(b"// kernel\r\n")
            (source / ".git").mkdir()
            result = stage_repository(source, root / "copy")
            self.assertEqual((result / "kernel.cu").read_bytes(), b"// kernel\r\n")
            self.assertFalse((result / ".git").exists())

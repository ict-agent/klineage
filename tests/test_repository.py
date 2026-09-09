import tempfile
import unittest
from contextlib import chdir
from pathlib import Path

from klineage.repository import repository_ignore, stage_repository


class RepositoryTests(unittest.TestCase):
    def test_relative_destination(self):
        with tempfile.TemporaryDirectory() as temporary, chdir(temporary):
            root = Path(temporary)
            (root / "nested").mkdir()
            ignore = repository_ignore(Path("nested/repository"))
            self.assertIn("nested", ignore(str(root), ["nested"]))

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

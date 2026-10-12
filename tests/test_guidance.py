from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from klineage.constants import AGENT_DOCS_START, AGENT_FILES
from klineage.harness import guidance


class AgentDocsTests(unittest.TestCase):
    def setUp(self):
        self.work = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.docs = self.work / "workspace"
        self.docs.mkdir()

    def test_preserves_user_text(self):
        path = self.docs / "AGENTS.md"
        before = "# User instructions\r\n\r\nPreserve this text.\r\n"
        path.write_bytes(before.encode())
        guidance.write_agent_docs(self.work)
        first = path.read_bytes()
        self.assertTrue(first.startswith(before.encode()))
        guidance.write_agent_docs(self.work)
        self.assertEqual(path.read_bytes(), first)

        after = "\nUser notes added later.\n"
        path.write_bytes(first + after.encode())
        guidance.write_agent_docs(self.work)
        contents = path.read_bytes().decode()
        self.assertTrue(contents.startswith(before))
        self.assertTrue(contents.endswith(after))
        self.assertEqual(contents.count(AGENT_DOCS_START), 1)

    def test_updates_active_override(self):
        path = self.docs / "AGENTS.override.md"
        path.write_text("# Override\nKeep these rules.\n")
        guidance.write_agent_docs(self.work)
        self.assertTrue(path.read_text().startswith("# Override\nKeep these rules.\n"))
        for filename in AGENT_FILES:
            self.assertIn("klineage.evaluate", (self.work / filename).read_text())

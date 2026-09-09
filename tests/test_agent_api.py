import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from klineage import agent_api, agent_tools


def extra_tool(value: int = 1) -> int:
    """Increment a value for the caller."""
    return value + 1


class ToolFactory:
    @classmethod
    def make(cls, value: int) -> int:
        """Build a value using the class factory."""
        return value


class AgentApiTests(unittest.TestCase):
    def setUp(self):
        self.work = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.dict(agent_api._FUNCTIONS))

    def test_register_refreshes_docs(self):
        self.assertIs(agent_api.agent_function(extra_tool), extra_tool)
        self.assertEqual(extra_tool(3), 4)
        docs = agent_api.function_docs()
        self.assertIn("from test_agent_api import extra_tool", docs)
        self.assertIn("extra_tool(value: int = 1) -> int", docs)
        self.assertIn("Increment a value", docs)
        self.assertIn(agent_tools.profile.__doc__.strip().splitlines()[0], docs)

    def test_factory_signature(self):
        agent_api.agent_function(ToolFactory.make.__func__)
        docs = agent_api.function_docs()
        self.assertIn("from test_agent_api import ToolFactory", docs)
        self.assertIn("ToolFactory.make(value: int) -> int", docs)
        self.assertNotIn("ToolFactory.make(cls", docs)
        self.assertEqual(ToolFactory.make(3), 3)

    def test_catalog_covers_task_apis(self):
        docs = agent_api.function_docs()
        paths = (
            "agent_tools.profile",
            "agent_tools.retrieve",
            "harness.eval.inspect_problem",
            "harness.eval.evaluate",
            "harness.artifacts.load_kernel",
            "harness.artifacts.save_kernel",
            "harness.artifacts.read_source_tree",
            "repository.stage_repository",
            "backend.get_backend",
            "backend.detect_backend",
            "kernel.Kernel.from_sources",
            "kernel.Kernel.from_dict",
            "kernel.Kernel.build",
            "kernel.Kernel.to_dict",
            "memory.storage.load_skill",
            "memory.storage.save_skill",
            "memory.skillcard.SkillCard.from_dict",
            "memory.skillcard.SkillCard.to_dict",
        )
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual(docs.count(f"### klineage.{path}\n"), 1)

    def test_preserves_user_text(self):
        path = self.work / "AGENTS.md"
        before = "# User instructions\r\n\r\nPreserve this text.\r\n"
        path.write_bytes(before.encode())
        agent_api.write_agent_docs(self.work)
        first = path.read_bytes()
        self.assertTrue(first.startswith(before.encode()))
        agent_api.write_agent_docs(self.work)
        self.assertEqual(path.read_bytes(), first)

        after = "\nUser notes added later.\n"
        path.write_bytes(first + after.encode())
        agent_api.agent_function(extra_tool)
        agent_api.write_agent_docs(self.work)
        contents = path.read_bytes().decode()
        self.assertTrue(contents.startswith(before))
        self.assertTrue(contents.endswith(after))
        self.assertEqual(contents.count(agent_api.FUNCTION_START), 1)
        self.assertIn("extra_tool", contents)

    def test_updates_active_override(self):
        path = self.work / "AGENTS.override.md"
        path.write_text("# Override\nKeep these rules.\n")
        agent_api.write_agent_docs(self.work)
        self.assertTrue(path.read_text().startswith("# Override\nKeep these rules.\n"))
        for filename in agent_api.AGENT_FILES:
            self.assertIn(
                "klineage.agent_tools.profile", (self.work / filename).read_text()
            )

    def test_keeps_empty_override(self):
        override = self.work / "AGENTS.override.md"
        override.write_text("\n")
        agent_api.write_agent_docs(self.work)
        self.assertEqual(override.read_text(), "\n")
        self.assertIn(
            "klineage.agent_tools.retrieve", (self.work / "AGENTS.md").read_text()
        )

    def test_rejects_damaged_markers(self):
        path = self.work / "AGENTS.md"
        start, end = agent_api.FUNCTION_START, agent_api.FUNCTION_END
        for original in (start, end, end + start, start + start + end):
            with self.subTest(original=original):
                path.write_text(original)
                with self.assertRaisesRegex(ValueError, "malformed"):
                    agent_api.write_agent_docs(self.work)
                self.assertEqual(path.read_text(), original)

    def test_rejects_symlink_output(self):
        original = self.work / "shared.md"
        original.write_text("Shared instructions")
        (self.work / "AGENTS.md").symlink_to(original)
        with self.assertRaisesRegex(ValueError, "symlink"):
            agent_api.write_agent_docs(self.work)
        self.assertEqual(original.read_text(), "Shared instructions")

    def test_failed_write_keeps_file(self):
        path = self.work / "AGENTS.md"
        path.write_text("Keep existing instructions")
        with (
            patch("klineage.agent_api.os.replace", side_effect=OSError("failed")),
            self.assertRaises(OSError),
        ):
            agent_api.write_agent_docs(self.work)
        self.assertEqual(path.read_text(), "Keep existing instructions")
        self.assertEqual(list(self.work.iterdir()), [path])

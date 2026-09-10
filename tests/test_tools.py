import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from kernel_fixtures import kernel, skill

from klineage import tools
from klineage.constants import AGENT_FILES, FUNCTION_END, FUNCTION_START
from klineage.errors import ActionError
from klineage.memory import save_skill


def extra_tool(value: int = 1) -> int:
    """Increment a value for the caller."""
    return value + 1


class FunctionDocsTests(unittest.TestCase):
    def setUp(self):
        self.work = Path(self.enterContext(tempfile.TemporaryDirectory()))
        tools.function_docs()
        self.enterContext(patch.dict(tools._FUNCTIONS))

    def test_register_refreshes_docs(self):
        self.assertIs(tools.agent_function(extra_tool), extra_tool)
        docs = tools.function_docs()
        self.assertIn("from test_tools import extra_tool", docs)
        self.assertIn("extra_tool(value: int = 1) -> int", docs)
        self.assertIn("Increment a value", docs)
        self.assertIn(tools.profile.__doc__.strip().splitlines()[0], docs)

    def test_factory_signature(self):
        docs = tools.function_docs()
        self.assertIn("from klineage.artifact.kernel import Kernel", docs)
        self.assertIn("Signature: `Kernel.from_sources(source_files:", docs)
        self.assertNotIn("Kernel.from_sources(cls", docs)

    def test_catalog_covers_task_apis(self):
        docs = tools.function_docs()
        paths = (
            "tools.profile",
            "harness.eval.inspect_problem",
            "harness.eval.evaluate",
            "artifact.kernel.load_kernel",
            "artifact.kernel.save_kernel",
            "artifact.source.read_source_tree",
            "artifact.repository.stage_repository",
            "backend.get_backend",
            "backend.detect_backend",
            "artifact.kernel.Kernel.from_sources",
            "artifact.kernel.Kernel.build",
        )
        documented = {
            line.removeprefix("### klineage.")
            for line in docs.splitlines()
            if line.startswith("### klineage.")
        }
        self.assertEqual(documented, set(paths))

    def test_preserves_user_text(self):
        path = self.work / "AGENTS.md"
        before = "# User instructions\r\n\r\nPreserve this text.\r\n"
        path.write_bytes(before.encode())
        tools.write_agent_docs(self.work)
        first = path.read_bytes()
        self.assertTrue(first.startswith(before.encode()))
        tools.write_agent_docs(self.work)
        self.assertEqual(path.read_bytes(), first)

        after = "\nUser notes added later.\n"
        path.write_bytes(first + after.encode())
        tools.agent_function(extra_tool)
        tools.write_agent_docs(self.work)
        contents = path.read_bytes().decode()
        self.assertTrue(contents.startswith(before))
        self.assertTrue(contents.endswith(after))
        self.assertEqual(contents.count(FUNCTION_START), 1)
        self.assertIn("extra_tool", contents)

    def test_updates_active_override(self):
        path = self.work / "AGENTS.override.md"
        path.write_text("# Override\nKeep these rules.\n")
        tools.write_agent_docs(self.work)
        self.assertTrue(path.read_text().startswith("# Override\nKeep these rules.\n"))
        for filename in AGENT_FILES:
            self.assertIn("klineage.tools.profile", (self.work / filename).read_text())

    def test_keeps_empty_override(self):
        override = self.work / "AGENTS.override.md"
        override.write_text("\n")
        tools.write_agent_docs(self.work)
        self.assertEqual(override.read_text(), "\n")
        self.assertIn("klineage.tools.profile", (self.work / "AGENTS.md").read_text())

    def test_rejects_damaged_markers(self):
        path = self.work / "AGENTS.md"
        start, end = FUNCTION_START, FUNCTION_END
        for original in (start, end, end + start, start + start + end):
            with self.subTest(original=original):
                path.write_text(original)
                with self.assertRaisesRegex(ValueError, "malformed"):
                    tools.write_agent_docs(self.work)
                self.assertEqual(path.read_text(), original)

    def test_rejects_symlink_output(self):
        original = self.work / "shared.md"
        original.write_text("Shared instructions")
        (self.work / "AGENTS.md").symlink_to(original)
        with self.assertRaisesRegex(ValueError, "symlink"):
            tools.write_agent_docs(self.work)
        self.assertEqual(original.read_text(), "Shared instructions")

    def test_failed_write_keeps_file(self):
        path = self.work / "AGENTS.md"
        path.write_text("Keep existing instructions")
        with (
            patch("klineage.tools.os.replace", side_effect=OSError("failed")),
            self.assertRaises(OSError),
        ):
            tools.write_agent_docs(self.work)
        self.assertEqual(path.read_text(), "Keep existing instructions")
        self.assertEqual(list(self.work.iterdir()), [path])


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.kernel = kernel()

    def test_profile_rejects_backend(self):
        for language, platform in (("hip", "hygon"), ("ascendc", "ascend")):
            problem = replace(self.kernel.problem, language=language, platform=platform)
            current = replace(self.kernel, problem=problem)
            with (
                self.subTest(language=language),
                self.assertRaisesRegex(
                    ActionError, "hardware-counter profiling is unsupported"
                ),
            ):
                tools.profile(current, self.root)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_retrieve_preserves_order(self):
        first, last = skill("first"), skill("last")
        path = save_skill(last, self.root / "saved.md")
        actual = tools.retrieve(self.kernel, (first, path))
        self.assertEqual(actual, (first, last))

    def test_retrieve_directory(self):
        first, last = skill("first"), skill("last")
        save_skill(last, self.root / "z/SKILL.md")
        save_skill(first, self.root / "a/nested/SKILL.md")
        save_skill(skill("unlisted"), self.root / "notes.md")
        self.assertEqual(tools.retrieve(self.kernel, self.root), (first, last))

    def test_retrieve_empty_and_file(self):
        self.assertEqual(tools.retrieve(self.kernel, ()), ())
        self.assertEqual(tools.retrieve(self.kernel, self.root), ())
        card = skill("single")
        path = save_skill(card, self.root / "SKILL.md")
        self.assertEqual(tools.retrieve(self.kernel, path), (card,))
        with self.assertRaises(FileNotFoundError):
            tools.retrieve(self.kernel, self.root / "missing")

    def test_requires_kernel(self):
        with self.assertRaisesRegex(TypeError, "kernel must be a Kernel"):
            tools.profile(self.kernel.problem)
        with self.assertRaisesRegex(TypeError, "kernel must be a Kernel"):
            tools.retrieve(self.kernel.problem, ())

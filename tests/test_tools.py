from klineage.artifact.problem import load_problem
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kernel_fixtures import kernel, skill

from klineage import tools
from klineage.constants import AGENT_FILES, FUNCTION_END, FUNCTION_START
from klineage.memory import save_skill


def extra_tool(value: int = 1) -> int:
    """Increment a value for the caller."""
    return value + 1


class FunctionDocsTests(unittest.TestCase):
    def setUp(self):
        self.work = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.docs = self.work / "workspace"
        self.docs.mkdir()
        tools.function_docs()
        self.enterContext(patch.dict(tools._FUNCTIONS))

    def test_register_refreshes_docs(self):
        self.assertIs(tools.agent_function(extra_tool), extra_tool)
        docs = tools.function_docs()
        self.assertIn("from test_tools import extra_tool", docs)
        self.assertIn("extra_tool(value: int = 1) -> int", docs)
        self.assertIn("Increment a value", docs)
        self.assertIn("klineage.harness.evaluate", docs)

    def test_factory_signature(self):
        docs = tools.function_docs()
        self.assertIn("from klineage.artifact import Kernel", docs)
        self.assertIn("Signature: `Kernel.load_kernel(build_root:", docs)
        self.assertNotIn("Kernel.load_kernel(cls", docs)

    def test_catalog_covers_task_apis(self):
        docs = tools.function_docs()
        paths = (
            "artifact.load_problem",
            "artifact.Problem.problem_spec",
            "harness.inspect_problem",
            "artifact.stage_repository",
            "harness.evaluate",
            "artifact.Kernel.load_kernel",
            "artifact.read_source_tree",
            "artifact.load_artifact",
            "artifact.save_artifact",
            "backend.get_backend",
            "backend.detect_backend",
            "artifact.Kernel.kernel_spec",
            "server.Client.evaluate",
            "server.Client.info",
            "server.Client.problem",
            "server.Client.result",
        )
        documented = {
            line.removeprefix("### klineage.")
            for line in docs.splitlines()
            if line.startswith("### klineage.")
        }
        self.assertEqual(documented, set(paths))

    def test_preserves_user_text(self):
        path = self.docs / "AGENTS.md"
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
        path = self.docs / "AGENTS.override.md"
        path.write_text("# Override\nKeep these rules.\n")
        tools.write_agent_docs(self.work)
        self.assertTrue(path.read_text().startswith("# Override\nKeep these rules.\n"))
        for filename in AGENT_FILES:
            self.assertIn("klineage.harness.evaluate", (self.work / filename).read_text())

    def test_keeps_empty_override(self):
        override = self.docs / "AGENTS.override.md"
        override.write_text("\n")
        tools.write_agent_docs(self.work)
        self.assertEqual(override.read_text(), "\n")
        self.assertIn("klineage.harness.evaluate", (self.docs / "AGENTS.md").read_text())

    def test_rejects_damaged_markers(self):
        path = self.docs / "AGENTS.md"
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
        (self.docs / "AGENTS.md").symlink_to(original)
        with self.assertRaisesRegex(ValueError, "symlink"):
            tools.write_agent_docs(self.work)
        self.assertEqual(original.read_text(), "Shared instructions")

    def test_docs_symlink_race(self):
        path = self.docs / "AGENTS.md"
        path.write_text("workspace guidance")
        with tempfile.TemporaryDirectory() as outside:
            secret = Path(outside) / "secret"
            secret.write_text("host-only guidance")
            original_check = Path.is_symlink
            changed = False

            def swap_after_check(candidate):
                nonlocal changed
                if candidate == path and not changed:
                    changed = True
                    path.unlink()
                    path.symlink_to(secret)
                    return False
                return original_check(candidate)

            with patch("klineage.tools.Path.is_symlink", swap_after_check):
                with self.assertRaises(OSError):
                    tools.write_agent_docs(self.work)
            self.assertEqual(secret.read_text(), "host-only guidance")

    def test_failed_write_keeps_file(self):
        path = self.docs / "AGENTS.md"
        path.write_text("Keep existing instructions")
        with (
            patch("klineage.tools.os.replace", side_effect=OSError("failed")),
            self.assertRaises(OSError),
        ):
            tools.write_agent_docs(self.work)
        self.assertEqual(path.read_text(), "Keep existing instructions")
        self.assertEqual(list(self.docs.iterdir()), [path])


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.kernel = kernel()

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
            tools.retrieve(load_problem(self.kernel), ())

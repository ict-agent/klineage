import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from kernel_fixtures import kernel, skill

from klineage import agent_tools
from klineage.errors import ActionError
from klineage.memory import save_skill


class AgentToolsTests(unittest.TestCase):
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
                agent_tools.profile(current, self.root)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_retrieve_preserves_order(self):
        first, last = skill("first"), skill("last")
        path = save_skill(last, self.root / "saved.md")
        actual = agent_tools.retrieve(self.kernel, (first, path))
        self.assertEqual(actual, (first, last))

    def test_retrieve_directory(self):
        first, last = skill("first"), skill("last")
        save_skill(last, self.root / "z/SKILL.md")
        save_skill(first, self.root / "a/nested/SKILL.md")
        save_skill(skill("unlisted"), self.root / "notes.md")
        self.assertEqual(agent_tools.retrieve(self.kernel, self.root), (first, last))

    def test_retrieve_empty_and_file(self):
        self.assertEqual(agent_tools.retrieve(self.kernel, ()), ())
        self.assertEqual(agent_tools.retrieve(self.kernel, self.root), ())
        card = skill("single")
        path = save_skill(card, self.root / "SKILL.md")
        self.assertEqual(agent_tools.retrieve(self.kernel, path), (card,))
        with self.assertRaises(FileNotFoundError):
            agent_tools.retrieve(self.kernel, self.root / "missing")

    def test_requires_kernel(self):
        with self.assertRaisesRegex(TypeError, "kernel must be a Kernel"):
            agent_tools.profile(self.kernel.problem)
        with self.assertRaisesRegex(TypeError, "kernel must be a Kernel"):
            agent_tools.retrieve(self.kernel.problem, ())

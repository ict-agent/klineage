import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from kernel_fixtures import kernel, skill

from klineage import agent_tools
from klineage.errors import ActionError
from klineage.memory import Scope, save_skill
from klineage.profiling import ProfileOptions


class AgentToolsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.kernel = kernel()

    def test_profile_delegates(self):
        options = ProfileOptions(sections=("LaunchStats",))
        report = {"metrics": [{"metric": "duration", "value": "1"}]}
        with patch.object(agent_tools, "capture", return_value=report) as capture:
            actual = agent_tools.profile(self.kernel, self.root, options=options)
        self.assertIs(actual, report)
        capture.assert_called_once_with(self.kernel, self.root.resolve(), options)

    def test_profile_defaults(self):
        with patch.object(agent_tools, "capture") as capture:
            agent_tools.profile(self.kernel)
        capture.assert_called_once_with(self.kernel, Path.cwd(), ProfileOptions())

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
        excluded = skill("excluded")
        wrong_scope = skill("hip-only", declared=Scope(languages=("hip",)))
        path = save_skill(last, self.root / "saved.md")
        actual = agent_tools.retrieve(
            self.kernel,
            (first, wrong_scope, excluded, path, first),
            exclude_skills=(excluded.skill_id,),
        )
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

    def test_conflicting_cards_fail(self):
        original = skill("conflict")
        changed = replace(original, intent="Different technique")
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            agent_tools.retrieve(self.kernel, (original, changed))

    def test_rejects_bad_exclusions(self):
        card = skill("warp.store")
        for excluded in (card.skill_id, card.skill_id.encode(), (None,), (1,)):
            with (
                self.subTest(excluded=excluded),
                self.assertRaisesRegex((TypeError, ValueError), "exclu"),
            ):
                agent_tools.retrieve(self.kernel, (card,), exclude_skills=excluded)

    def test_requires_kernel(self):
        with self.assertRaisesRegex(TypeError, "kernel must be a Kernel"):
            agent_tools.profile(self.kernel.problem)
        with self.assertRaisesRegex(TypeError, "kernel must be a Kernel"):
            agent_tools.retrieve(self.kernel.problem, ())

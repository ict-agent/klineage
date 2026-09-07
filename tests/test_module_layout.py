from __future__ import annotations

import unittest

from klineage import action
from klineage.harness.eval import ValidationResult
from klineage.kernel import Kernel, TargetContext
from klineage.memory.lineage import Lineage
from klineage.memory.skillcard import SkillCard


class ModuleLayoutTests(unittest.TestCase):
    def test_action_module_contains_only_public_action_functions(self) -> None:
        self.assertEqual(
            action.__all__,
            ["apply", "code_gen", "decompose", "init", "profile", "retrieve"],
        )

    def test_actions_are_defined_by_their_action_modules(self) -> None:
        expected_modules = {
            action.apply: "klineage.action.apply",
            action.code_gen: "klineage.action.code_gen",
            action.decompose: "klineage.action.decompose",
            action.init: "klineage.action.init",
            action.profile: "klineage.action.profile",
            action.retrieve: "klineage.action.retrieve",
        }
        for function, module in expected_modules.items():
            with self.subTest(function=function.__name__):
                self.assertEqual(function.__module__, module)

    def test_classes_are_defined_by_their_domain_modules(self) -> None:
        expected_modules = {
            Kernel: "klineage.kernel",
            Lineage: "klineage.memory.lineage",
            SkillCard: "klineage.memory.skillcard",
            TargetContext: "klineage.kernel",
            ValidationResult: "klineage.harness.eval",
        }
        for class_, module in expected_modules.items():
            with self.subTest(class_=class_.__name__):
                self.assertEqual(class_.__module__, module)


if __name__ == "__main__":
    unittest.main()

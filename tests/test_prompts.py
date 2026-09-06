from __future__ import annotations

import json
import re
import unittest

from jinja2 import UndefinedError

from klineage.prompts import render_prompt


class PromptTemplateTests(unittest.TestCase):
    def test_every_codex_operation_has_a_renderable_template(self) -> None:
        prompts = {
            "init": {},
            "decompose": {"step_number": 1},
            "decompose_rederive": {},
            "decompose_lift": {"transition_count": 2},
            "code_gen": {"skill_count": 1},
        }
        for name, context in prompts.items():
            with self.subTest(name=name):
                self.assertTrue(render_prompt(name, **context))

    def test_templates_use_strict_variables(self) -> None:
        with self.assertRaises(UndefinedError):
            render_prompt("code_gen")

    def test_lift_prompt_names_scope_key(self) -> None:
        prompt = render_prompt("decompose_lift", transition_count=1)

        self.assertIn('"scope" as', prompt)

    def test_mma_example_contract(self) -> None:
        for name, context in (
            ("decompose", {"step_number": 1}),
            ("decompose_lift", {"transition_count": 1}),
        ):
            with self.subTest(name=name):
                prompt = render_prompt(name, **context)
                example = re.search(r"Example SkillCard:\n```json\n(.*?)\n```", prompt, re.S)
                self.assertIsNotNone(example)
                card = json.loads(example.group(1))

                self.assertEqual(card["intent"], "Use tensor-core MMA for the tiled contraction.")
                self.assertIn("wmma::mma_sync", card["carrier"])
                self.assertTrue(card["preconditions"])
                self.assertTrue(card["effects"])
                self.assertIn({"locus": "gemm.main", "name": "mma"}, card["provides"])
                self.assertIn({"locus": "gemm.main", "name": "mma"}, card["conflicts"])
                self.assertIn("Before:", prompt)
                self.assertIn("After:", prompt)
                self.assertIn("Anti-pattern:", prompt)
                self.assertIn("tile 128 -> 64 -> 32", prompt)
                self.assertIn("vector width 4 -> 2 -> 1", prompt)

    def test_decompose_stop_evidence(self) -> None:
        prompt = render_prompt("decompose", step_number=1)

        self.assertIn('"reason"', prompt)
        self.assertIn("budget exhaustion", prompt)
        self.assertIn("actually present", prompt)
        self.assertIn("one semantic transformation", prompt)

    def test_codegen_state_contract(self) -> None:
        prompt = render_prompt("code_gen", skill_count=1)

        self.assertIn("requires/provides/conflicts", prompt)
        self.assertIn("same locus", prompt)
        self.assertIn("prior_actions", prompt)

    def test_template_names_cannot_escape_the_prompt_package(self) -> None:
        with self.assertRaises(ValueError):
            render_prompt("../code_gen")


if __name__ == "__main__":
    unittest.main()

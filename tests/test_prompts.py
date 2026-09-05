from __future__ import annotations

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

    def test_template_names_cannot_escape_the_prompt_package(self) -> None:
        with self.assertRaises(ValueError):
            render_prompt("../code_gen")


if __name__ == "__main__":
    unittest.main()

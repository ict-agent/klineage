from __future__ import annotations

import re
import unittest

from jinja2 import UndefinedError
from prompt_fixtures import prompt_inputs, task_contexts

from klineage.prompts import render_prompt

PRECONDITION_CATEGORIES = ("Data types", "Layout", "Storage", "Pipeline", "Hardware")


class PromptTemplateTests(unittest.TestCase):
    def setUp(self):
        self.contexts = task_contexts()

    def test_task_values_stay_data(self):
        values = {
            "problem": '/tmp/问题 "quoted"\n{{ repository }}.json',
            "repository": "https://example.com/repo?a=1&b=2",
            "expert_kernel": "{% include 'missing' %}",
        }
        self.assertEqual(prompt_inputs(render_prompt("init", **values)), values)

    def test_tensor_order_is_preserved(self):
        from kernel_fixtures import kernel

        current = kernel().to_dict()
        inputs = current["problem"]["definition"]["inputs"]
        current["problem"]["definition"]["inputs"] = dict(reversed(inputs.items()))
        prompt = render_prompt("profile", kernel=current, options={})
        restored = prompt_inputs(prompt)["kernel"]
        self.assertEqual(
            list(restored["problem"]["definition"]["inputs"]),
            list(reversed(inputs)),
        )

    def test_every_codex_operation_has_a_renderable_template(self):
        prompts = {
            **self.contexts,
            "decompose_lift": {},
            "decompose_gemm": {},
            "verify": {"criteria": "Inspect artifacts.", "run_id": "Init-test"},
            **{
                f"verify_{name}": {}
                for name in (
                    "init",
                    "code_gen",
                    "decompose",
                    "apply",
                    "profile",
                    "retrieve",
                )
            },
        }
        for name, context in prompts.items():
            with self.subTest(name=name):
                self.assertTrue(render_prompt(name, **context))

    def test_templates_use_strict_variables(self):
        for name, context in self.contexts.items():
            for field in context:
                missing = {key: value for key, value in context.items() if key != field}
                with (
                    self.subTest(template=name, field=field),
                    self.assertRaises((UndefinedError, TypeError)),
                ):
                    render_prompt(name, **missing)

    def test_decompose_sections(self):
        prompt = render_prompt("decompose", **self.contexts["decompose"])
        instructions = re.sub(
            r"^````markdown\n.*?\n````", "", prompt, flags=re.MULTILINE | re.DOTALL
        )
        self.assertEqual(
            re.findall(r"^# (.+)$", instructions, re.MULTILINE),
            ["Overview", "Procedure", "Self-Verification"],
        )
        self.assertEqual(prompt_inputs(prompt), self.contexts["decompose"])
        for text in (prompt, render_prompt("verify_decompose")):
            self.assertNotIn("step_index", text)
            self.assertNotIn("Step index:", text)

    def test_decompose_stop_evidence(self):
        prompt = render_prompt("decompose", **self.contexts["decompose"])

        self.assertIn("If the input is already naive, preserve it and emit no", prompt)
        self.assertIn("actually present", prompt)
        self.assertIn("one semantic transformation", prompt)

    def test_skill_section_contract(self):
        prompt = render_prompt("decompose", **self.contexts["decompose"])
        self.assertNotIn("Before & And", prompt)
        self.assertIn("End body with a Code Change Snippet section", prompt)
        self.assertIn("minimum prerequisites", prompt)
        self.assertIn("example configuration", prompt)
        self.assertIn("klineage.harness.evaluate", prompt)
        self.assertNotIn("check_streams", prompt)
        self.assertNotIn("graph replay", prompt.lower())
        self.assertIn("Once these checks pass", prompt)

    def test_eval_docs_use_one_entry(self):
        from pathlib import Path

        package = Path(__file__).parents[1] / "src/klineage"
        for path in (
            package / "skills/cuda/cuda.md",
            package / "skills/bench/SKILL.md",
        ):
            with self.subTest(path=path):
                text = path.read_text()
                self.assertIn("klineage.harness", text)
                self.assertNotIn("check_streams", text)
                self.assertNotIn("graph replay", text.lower())
                self.assertNotIn("replay a captured", text.lower())

    def test_prerequisites_are_dependencies(self):
        import yaml

        prompt = render_prompt("decompose_lift")
        metadata = yaml.safe_load(prompt.split("---", 2)[1])
        for condition in metadata["preconditions"]:
            with self.subTest(condition=condition):
                self.assertIn("required", condition)
        self.assertNotIn("describes the existing stage count", prompt)
        self.assertIn("## Example configuration", prompt)

    def test_code_section_is_consistent(self):
        from pathlib import Path

        for name in ("decompose_lift", "decompose_gemm"):
            prompt = render_prompt(name)
            self.assertEqual(
                re.findall(r"^# (.+)$", prompt, re.MULTILINE),
                ["Overview", "Precondition", "Scope", "Code Change Snippet"],
            )
        contract = (
            Path(__file__).parents[1] / "src/klineage/message/decompose.md"
        ).read_text()
        self.assertIn("# Code Change Snippet", contract)
        self.assertNotIn("Before & And", contract)

    def test_skill_dependency_audit(self):
        prompt = render_prompt("decompose", **self.contexts["decompose"])
        checks = prompt.split("# Self-Verification", 1)[1]
        self.assertIn("**Skill prerequisites.**", checks)
        self.assertIn(
            "YAML preconditions and the entire # Precondition section", checks
        )
        self.assertIn("fails without it", checks)
        self.assertIn(
            "Serialization roundtrip does not validate these semantics", checks
        )

    def test_precondition_categories(self):
        import yaml

        for name in ("decompose_lift", "decompose_gemm"):
            with self.subTest(template=name):
                prompt = render_prompt(name)
                metadata = yaml.safe_load(prompt.split("---", 2)[1])
                self.assertEqual(
                    tuple(item.split(":", 1)[0] for item in metadata["preconditions"]),
                    PRECONDITION_CATEGORIES,
                )
                body = prompt.split("# Precondition\n", 1)[1].split("# Scope\n", 1)[0]
                self.assertEqual(
                    tuple(re.findall(r"^- ([^:]+):", body, re.MULTILINE)),
                    PRECONDITION_CATEGORIES,
                )

    def test_category_review_contract(self):
        from pathlib import Path

        prompt = render_prompt("decompose", **self.contexts["decompose"])
        contract = (
            Path(__file__).parents[1] / "src/klineage/message/decompose.md"
        ).read_text()
        texts = {
            "format": render_prompt("decompose_lift"),
            "self-review": prompt.split("# Self-Verification", 1)[1],
            "verifier": render_prompt("verify_decompose"),
            "contract": contract,
        }
        for name, text in texts.items():
            with self.subTest(name=name):
                self.assertIn("Keep all five categories explicit", text)
                self.assertIn("no additional requirement", text)
                self.assertNotIn("omit categories", text.lower())

    def test_example_uses_dependencies(self):
        import yaml

        prompt = render_prompt("decompose_gemm")
        metadata = yaml.safe_load(prompt.split("---", 2)[1])
        conditions = "\n".join(metadata["preconditions"])
        conditions += prompt.split("# Precondition\n", 1)[1].split("# Scope\n", 1)[0]
        self.assertNotIn("are FP32", conditions)
        self.assertNotIn("FP32 inputs", conditions)
        self.assertNotIn("signed 32-bit", conditions)
        self.assertIn("FP32", prompt.split("## Example configuration\n", 1)[1])

    def test_codegen_state_contract(self):
        prompt = render_prompt("code_gen", **self.contexts["code_gen"])

        self.assertIn("preconditions", prompt)
        self.assertIn("actual code", prompt)
        self.assertIn("source_files", prompt)

    def test_native_backend_routing(self):
        from copy import deepcopy

        for language, platform in (("hip", "hygon"), ("ascendc", "ascend")):
            for action, field in (
                ("code_gen", "current_kernel"),
                ("apply", "current_kernel"),
                ("decompose", "input_kernel"),
            ):
                context = deepcopy(self.contexts[action])
                context[field]["problem"].update(language=language, platform=platform)
                with self.subTest(action=action, language=language):
                    prompt = render_prompt(action, **context)
                    self.assertIn(f"Use .agents/skills/{language}/SKILL.md", prompt)
                    self.assertNotIn('language="python"', prompt)
                    self.assertNotIn("You are given a cuda kernel", prompt)

    def test_example_respects_backend(self):
        from copy import deepcopy

        for language in ("cuda", "hip", "ascendc"):
            context = deepcopy(self.contexts["decompose"])
            context["input_kernel"]["problem"]["language"] = language
            prompt = render_prompt("decompose", **context)
            with self.subTest(language=language):
                self.assertEqual(
                    "skill_id: gemm.shared-memory-staging" in prompt,
                    language == "cuda",
                )

    def test_template_names_cannot_escape_the_prompt_package(self):
        with self.assertRaises(ValueError):
            render_prompt("../code_gen")


if __name__ == "__main__":
    unittest.main()

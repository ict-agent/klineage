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
        context = {**self.contexts["apply"], "current_kernel": current}
        restored = prompt_inputs(render_prompt("apply", **context))["current_kernel"]
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
                    "decompose",
                )
            },
            "verify_apply": {"memory": self.contexts["apply"]["memory"]},
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

    def test_eval_docs_use_one_entry(self):
        from pathlib import Path

        package = Path(__file__).parents[1] / "src/klineage"
        texts = {
            "decompose": render_prompt("decompose", **self.contexts["decompose"]),
            **{
                name: (package / name).read_text()
                for name in ("skills/cuda/cuda.md", "skills/bench/SKILL.md")
            },
        }
        for name, text in texts.items():
            with self.subTest(name=name):
                self.assertIn("klineage.harness", text)
                self.assertNotIn("check_streams", text)
                self.assertNotIn("graph replay", text.lower())
                self.assertNotIn("replay a captured", text.lower())

    def test_naive_decompose_stops(self):
        prompt = render_prompt("decompose", **self.contexts["decompose"])
        first_step = prompt.split("# Procedure\n", 1)[1].split("\n2.", 1)[0]
        self.assertIn("skip steps 2-5", first_step)
        self.assertIn("source audit", first_step)
        atomicity = prompt.split("**Atomicity and replayability.**", 1)[1]
        self.assertTrue(atomicity.lstrip().startswith("For a removal,"))

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

    def test_apply_memory_directory(self):
        context = self.contexts["apply"]
        prompt = render_prompt("apply", **context)
        self.assertEqual(prompt_inputs(prompt), context)
        self.assertNotIn("Skill:", prompt)
        self.assertIn("retrieve(current_kernel, Path(memory)", prompt)
        self.assertIn("profile(current_kernel, Path.cwd())", prompt)
        self.assertIn("enumerate and read", prompt)
        self.assertNotIn("SkillCard.from_dict", prompt)
        self.assertIn("no SKILL.md or submission/", prompt)

    def test_apply_baseline_modes(self):
        for memory in (None, "", " \t\n"):
            with self.subTest(memory=memory):
                context = {**self.contexts["apply"], "memory": memory}
                prompt = render_prompt("apply", **context)
                self.assertEqual(
                    prompt_inputs(prompt),
                    {"current_kernel": context["current_kernel"]},
                )
                self.assertIn("Independently choose one optimization", prompt)
                self.assertIn("Do not emit SKILL.md", prompt)
                for text in (prompt, render_prompt("verify_apply", memory=memory)):
                    self.assertNotIn("retrieve", text)
                    self.assertNotIn(".agents/skills/memory", text)
                    self.assertNotIn("no candidate", text)

    def test_apply_self_verification(self):
        for memory in (None, self.contexts["apply"]["memory"]):
            with self.subTest(memory=memory):
                prompt = render_prompt(
                    "apply", **{**self.contexts["apply"], "memory": memory}
                )
                self.assertEqual(
                    re.findall(r"^# (.+)$", prompt, re.MULTILINE),
                    ["Overview", "Procedure", "Self-Verification"],
                )
                checks = prompt.split("# Self-Verification", 1)[1]
                self.assertIn(
                    "evaluate(candidate, Path.cwd(), reference=current_kernel)", checks
                )
                self.assertIn("paired performance gate", checks)

    def test_decompose_backends(self):
        from copy import deepcopy

        for language, platform in (("hip", "hygon"), ("ascendc", "ascend")):
            context = deepcopy(self.contexts["decompose"])
            context["input_kernel"]["problem"].update(
                language=language, platform=platform
            )
            with self.subTest(language=language):
                prompt = render_prompt("decompose", **context)
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
            render_prompt("../apply")


if __name__ == "__main__":
    unittest.main()

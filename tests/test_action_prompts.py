import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from problem_fixtures import problem_spec
from prompt_fixtures import prompt_inputs

from klineage.action import Apply, CodeGen, Decompose, Init, Profile, Retrieve
from klineage.kernel import Kernel
from klineage.memory import (
    Scope,
    SkillCard,
)
from klineage.profiling import ProfileOptions


class ActionPromptTests(unittest.TestCase):
    def setUp(self):
        self.work = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.current = Kernel(
            name="gemm",
            source_files={"kernel.cu": "__global__ void gemm() {}"},
            problem=problem_spec(),
        )
        self.expert = replace(
            self.current,
            source_files={
                "kernel.cu": self.current.source_files["kernel.cu"] + " // tiled"
            },
        )
        self.card = SkillCard(
            skill_id="tile",
            intent="Reuse tiled operands.",
            preconditions=("Shared memory is available.",),
            scope=Scope(),
            body="# Overview\n\nCooperatively load operands, synchronize, then reuse.",
        )
        self.runner = Mock(return_value=SimpleNamespace(final_message="generated"))
        self.enterContext(
            patch("klineage.action.action.CodexRunner", return_value=self.runner)
        )

    def payload(self, action):
        return prompt_inputs(action.prompt)

    def test_apply_inputs(self):
        for current, card in (
            (self.current, self.card),
            (self.work / "current", self.card),
            (self.current, self.work / "SKILL.md"),
        ):
            with self.subTest(current=current, card=card):
                action = Apply(current, card, workdir=self.work)
                data = self.payload(action)
                expected = (
                    current.to_dict() if isinstance(current, Kernel) else str(current)
                )
                self.assertEqual(data["current_kernel"], expected)
                expected_card = (
                    card.to_dict() if isinstance(card, SkillCard) else str(card)
                )
                self.assertEqual(data["skill"], expected_card)
        self.runner.assert_not_called()
        self.assertEqual(list(self.work.iterdir()), [])

    def test_decompose_inputs(self):
        action = Decompose(self.expert, workdir=self.work)
        data = self.payload(action)
        self.assertEqual(data, {"input_kernel": self.expert.to_dict()})
        self.runner.assert_not_called()

    def test_decompose_path(self):
        source = self.work / 'previous "step"' / "decompose" / "2"
        action = Decompose(source, workdir=self.work)
        self.assertEqual(self.payload(action), {"input_kernel": str(source)})
        self.runner.assert_not_called()

    def test_decompose_verifier_mode(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                action = Decompose(
                    self.current, workdir=self.work, enable_verifier=enabled
                )
                self.assertIs(action.enable_verifier, enabled)
                self.assertIn("Perform these checks before finishing", action.prompt)
                self.assertEqual(
                    "The external verifier independently" in action.prompt, enabled
                )

    def test_retry_keeps_input(self):
        action = Decompose(self.expert, workdir=self.work, max_retries=1)
        with patch.object(action, "verify", side_effect=(False, True)):
            action.run()
        self.assertEqual(action.attempt, 2)
        self.assertEqual(self.runner.call_count, 2)
        for call in self.runner.call_args_list:
            self.assertEqual(
                prompt_inputs(call.args[0]), {"input_kernel": self.expert.to_dict()}
            )

    def test_codegen_inputs(self):
        action = CodeGen(self.current, self.card, workdir=self.work)
        data = self.payload(action)
        self.assertEqual(data["current_kernel"], self.current.to_dict())
        self.assertEqual(data["skill"], self.card.to_dict())
        self.runner.assert_not_called()

    def test_codegen_language(self):
        for language in ("triton", "notcuda"):
            with self.subTest(language=language), self.assertRaises(ValueError):
                current = replace(
                    self.current,
                    problem=replace(self.current.problem, language=language),
                )
                CodeGen(current, self.card, workdir=self.work)

    def test_python_keeps_entrypoint(self):
        from klineage.prompts import render_prompt

        current = replace(
            self.current, problem=replace(self.current.problem, language="python")
        )
        prompts = (
            render_prompt("backend", target_language="python"),
            CodeGen(current, self.card, workdir=self.work).prompt,
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt[:40]):
                self.assertIn("configured callable", prompt)
                self.assertNotIn('entry_point="kernel.py::run"', prompt)
                self.assertNotIn("destination_passing_style=false", prompt)

    def test_codegen_native_backends(self):
        from klineage.backend import BACKENDS

        for backend in BACKENDS:
            current = replace(
                self.current,
                problem=replace(
                    self.current.problem,
                    language=backend.language,
                    platform=backend.kind.value,
                ),
                source_files={backend.raw_source: "native source"},
            )
            with self.subTest(backend=backend.kind):
                action = CodeGen(current, self.card, workdir=self.work)
                self.assertEqual(
                    self.payload(action)["current_kernel"], current.to_dict()
                )
                self.assertIn(
                    f"Use .agents/skills/{backend.skill_name}/SKILL.md", action.prompt
                )

    def test_apply_is_one_step(self):
        with self.assertRaises(TypeError):
            Apply(self.current, (self.card, self.card), workdir=self.work)

    def test_init_bundle_input(self):
        problem = self.work / "reference.py"
        problem.touch()
        action = Init(problem, self.work, "expert.cu", workdir=self.work)
        self.assertEqual(
            set(self.payload(action)), {"problem", "repository", "expert_kernel"}
        )
        self.assertIn("submission/", action.prompt)
        self.assertFalse((self.work / "submission").exists())

    def test_profile_inputs(self):
        options = ProfileOptions(set="full", sections=("WarpStateStats",))
        action = Profile(self.current, options=options, workdir=self.work)
        data = self.payload(action)
        self.assertEqual(data["kernel"], self.current.to_dict())
        self.assertEqual(data["options"], json.loads(json.dumps(asdict(options))))
        self.runner.assert_not_called()

    def test_retrieve_inputs(self):
        for skills in ((), (self.card,)):
            with self.subTest(skills=skills):
                action = Retrieve(
                    self.current,
                    skills,
                    workdir=self.work,
                    exclude_skills=("tile",),
                )
                data = self.payload(action)
                self.assertEqual(data["current_kernel"], self.current.to_dict())
                self.assertEqual(data["skills"], [item.to_dict() for item in skills])
                self.assertEqual(data["exclude_skills"], ["tile"])
        self.runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()

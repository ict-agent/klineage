import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from problem_fixtures import problem_spec
from prompt_fixtures import prompt_inputs

from klineage.action import Apply, Decompose, Init
from klineage.kernel import Kernel
from klineage.memory import (
    Scope,
    SkillCard,
)


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

    def test_apply_language(self):
        for language in ("triton", "notcuda"):
            with self.subTest(language=language), self.assertRaises(ValueError):
                current = replace(
                    self.current,
                    problem=replace(self.current.problem, language=language),
                )
                Apply(current, self.card, workdir=self.work)

    def test_python_keeps_entrypoint(self):
        from klineage.prompts import render_prompt

        current = replace(
            self.current, problem=replace(self.current.problem, language="python")
        )
        prompts = (
            render_prompt("backend", target_language="python"),
            Apply(current, self.card, workdir=self.work).prompt,
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt[:40]):
                self.assertIn("configured callable", prompt)
                self.assertNotIn('entry_point="kernel.py::run"', prompt)
                self.assertNotIn("destination_passing_style=false", prompt)

    def test_apply_native_backends(self):
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
                action = Apply(current, self.card, workdir=self.work)
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

    def test_apply_memory_inputs(self):
        for memory in ((), (self.card,), self.work / "memory"):
            with self.subTest(memory=memory):
                action = Apply(
                    self.current,
                    memory=memory,
                    workdir=self.work,
                    exclude_skills=("tile",),
                )
                data = self.payload(action)
                self.assertEqual(data["current_kernel"], self.current.to_dict())
                self.assertIsNone(data["skill"])
                expected = (
                    str(memory)
                    if isinstance(memory, Path)
                    else [item.to_dict() for item in memory]
                )
                self.assertEqual(data["memory"], expected)
                self.assertEqual(data["exclude_skills"], ["tile"])
        self.runner.assert_not_called()

    def test_apply_bad_exclusions(self):
        for excluded in ("warp.store", b"warp.store", (None,), (1,)):
            with (
                self.subTest(excluded=excluded),
                self.assertRaises((TypeError, ValueError)),
            ):
                Apply(self.current, memory=(self.card,), exclude_skills=excluded)
        self.runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()

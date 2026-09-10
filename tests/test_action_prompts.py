import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from problem_fixtures import problem_spec
from prompt_fixtures import prompt_inputs

from klineage.action import Apply, Decompose, Init
from klineage.artifact.kernel import Kernel
from klineage.memory import (
    Scope,
    SkillCard,
    save_skill,
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
        self.memory = self.work / "memory"
        save_skill(self.card, self.memory / "tile" / "SKILL.md")
        self.runner = Mock(return_value=SimpleNamespace(final_message="generated"))
        self.enterContext(
            patch("klineage.action.action.CodexRunner", return_value=self.runner)
        )

    def payload(self, action):
        return prompt_inputs(action.prompt)

    def test_apply_inputs(self):
        cases = (
            (self.current, self.memory, ()),
            (self.work / "current", str(self.memory), ("tile",)),
        )
        for current, memory, excluded in cases:
            with self.subTest(current=current):
                action = Apply(
                    current,
                    memory=memory,
                    exclude_skills=excluded,
                    workdir=self.work,
                )
                data = self.payload(action)
                expected = (
                    current.to_dict() if isinstance(current, Kernel) else str(current)
                )
                self.assertEqual(
                    data,
                    {
                        "current_kernel": expected,
                        "memory": str(self.work / ".agents/skills/memory"),
                        "exclude_skills": list(excluded),
                    },
                )
                self.assertNotIn("\nSkill:", action.prompt)
                self.assertIn("retrieve", action.verify_prompt)
        self.runner.assert_not_called()
        self.assertEqual(list(self.work.iterdir()), [self.memory])

    def test_decompose_inputs(self):
        for source in (self.expert, self.work / 'previous "step"' / "decompose" / "2"):
            with self.subTest(source=source):
                action = Decompose(source, workdir=self.work)
                expected = (
                    source.to_dict() if isinstance(source, Kernel) else str(source)
                )
                self.assertEqual(self.payload(action), {"input_kernel": expected})
        self.runner.assert_not_called()

    def test_decompose_verifier_mode(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                action = Decompose(
                    self.current, workdir=self.work, enable_verifier=enabled
                )
                self.assertIs(action.enable_verifier, enabled)
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

    def test_python_keeps_entrypoint(self):
        current = replace(
            self.current, problem=replace(self.current.problem, language="python")
        )
        prompt = Apply(current, memory=self.memory, workdir=self.work).prompt
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
                action = Apply(current, memory=self.memory, workdir=self.work)
                self.assertEqual(
                    self.payload(action)["current_kernel"], current.to_dict()
                )
                self.assertIn(
                    f"Use .agents/skills/{backend.skill_name}/SKILL.md", action.prompt
                )

    def test_init_bundle_input(self):
        problem = self.work / "reference.py"
        problem.touch()
        action = Init(problem, self.work, "expert.cu", workdir=self.work)
        self.assertEqual(
            set(self.payload(action)), {"problem", "repository", "expert_kernel"}
        )
        self.assertIn("submission/", action.prompt)
        self.assertFalse((self.work / "submission").exists())

    def test_apply_mounts_before_run(self):
        events = []
        self.runner.mount_memory.side_effect = lambda memory: events.append("mount")
        self.runner.side_effect = lambda *args, **kwargs: (
            events.append("run") or SimpleNamespace(final_message="generated")
        )
        action = Apply(
            self.current, memory=self.memory, workdir=self.work, enable_verifier=False
        )
        self.runner.mount_memory.assert_not_called()
        action.run()
        self.assertEqual(events, ["mount", "run"])
        self.runner.mount_memory.assert_called_once_with(self.memory)

    def test_apply_without_memory(self):
        for memory in (None, "", " \t "):
            with self.subTest(memory=memory):
                action = Apply(
                    self.current,
                    memory=memory,
                    workdir=self.work,
                    enable_verifier=False,
                )
                self.assertIsNone(action.memory)
                self.assertEqual(
                    self.payload(action), {"current_kernel": self.current.to_dict()}
                )
                self.assertNotIn("retrieve", action.verify_prompt)
                action.run()
                self.runner.mount_memory.assert_called_with(None)

        self.assertIsNone(Apply(self.current, workdir=self.work).memory)

    def test_requires_memory_directory(self):
        for memory, error in (
            (self.work / "missing", FileNotFoundError),
            (self.memory / "tile/SKILL.md", NotADirectoryError),
            ((self.card,), TypeError),
        ):
            with self.subTest(memory=memory), self.assertRaises(error):
                Apply(self.current, memory=memory, workdir=self.work)
        self.runner.assert_not_called()

    def test_apply_bad_exclusions(self):
        for excluded in ("warp.store", b"warp.store", (None,), (1,)):
            with (
                self.subTest(excluded=excluded),
                self.assertRaises((TypeError, ValueError)),
            ):
                Apply(self.current, memory=self.memory, exclude_skills=excluded)
        self.runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kernel_fixtures import accepted, kernel, skill
from prompt_fixtures import prompt_inputs

from klineage.action import Init, init_memory, workflow
from klineage.errors import StructuredOutputError, ValidationGateError
from klineage.harness.artifacts import load_kernel, save_kernel
from klineage.memory import load_skill, save_skill


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.work = self.root / "workflow"
        self.problem = self.root / "reference.py"
        self.problem.write_text("def run(a, b): return a @ b\n")
        self.repo = self.root / "repository"
        self.repo.mkdir()
        self.expert_path = self.repo / "expert.cu"
        mechanisms = ("tile", "mma")
        self.states = tuple(
            replace(
                kernel(source=f"__global__ void state{index}() {{}}"),
            )
            for index in range(3)
        )
        self.expert_path.write_text(self.states[-1].source_files["kernel.cu"])
        self.cards = [skill(mechanism) for mechanism in mechanisms]
        self.events = []
        self.empty_at = None
        self.failed_stage = None
        self.missing_stage = None
        self.omit_skill = False
        self.enterContext(
            patch(
                "klineage.harness.codex_runner.resolve_executable",
                return_value="/bin/true",
            )
        )
        self.calls = self.enterContext(
            patch(
                "klineage.harness.codex_runner.CodexRunner.__call__",
                autospec=True,
                side_effect=self.run_codex,
            )
        )

    def run_codex(self, runner, prompt, *, run_id):
        run_id = run_id.split("-", 1)[0]
        stage = runner.work_dir.relative_to(self.work).as_posix()
        data = {} if run_id == "Verify" else prompt_inputs(prompt)
        self.events.append((stage, run_id, data, runner.timeout))
        if run_id == "Verify":
            if stage == self.failed_stage:
                return SimpleNamespace(final_message="false")
            if stage.startswith("decompose/"):
                baseline = replace(load_kernel(runner.work_dir), validation=accepted())
                save_kernel(baseline, runner.work_dir)
            return SimpleNamespace(final_message="true")

        if stage == self.missing_stage:
            return SimpleNamespace(final_message="missing output")

        if run_id == "Init":
            self.assertEqual(data["problem"], str(self.problem))
            save_kernel(self.states[-1], runner.work_dir)
        elif run_id == "Decompose":
            current = load_kernel(Path(data["input_kernel"]))
            self.assertEqual(set(data), {"input_kernel"})
            index = next(
                i
                for i, state in enumerate(self.states)
                if state.fingerprint == current.fingerprint
            )
            if index:
                if not self.omit_skill:
                    save_skill(self.cards[index - 1], runner.work_dir / "SKILL.md")
                current = self.states[index - 1]
            save_kernel(current, runner.work_dir)
        elif run_id == "Profile":
            current = load_kernel(Path(data["kernel"]))
            raw = runner.work_dir / "metrics.csv"
            report = runner.work_dir / "capture.ncu-rep"
            raw.write_text("captured metrics")
            report.write_text("captured report")
            save_kernel(current, runner.work_dir)
        elif run_id == "Retrieve":
            current = load_kernel(Path(data["current_kernel"]))
            self.assertTrue((Path(data["current_kernel"]) / "metrics.csv").is_file())
            index = int(runner.work_dir.name)
            choices = []
            for path in data["skills"]:
                card = load_skill(path)
                if card.skill_id in data.get("exclude_skills", ()):
                    continue
                index_in_memory = next(
                    i
                    for i, item in enumerate(self.cards)
                    if item.skill_id == card.skill_id
                )
                if (
                    index_in_memory
                    and self.cards[index_in_memory - 1].skill_id
                    not in current.source_files["kernel.cu"]
                ):
                    continue
                choices.append(card)
            if choices and index != self.empty_at:
                save_skill(choices[0], runner.work_dir / "SKILL.md")
        elif run_id == "Apply":
            current = load_kernel(Path(data["current_kernel"]))
            card = load_skill(data["skill"])
            candidate = replace(
                current,
                source_files={
                    "kernel.cu": current.source_files["kernel.cu"]
                    + f" // {card.skill_id}"
                },
                validation=None,
            )
            save_kernel(candidate, runner.work_dir)
        else:
            self.fail(f"unexpected stage: {stage}")
        return SimpleNamespace(final_message="generated")

    def run_workflow(self, **options):
        return workflow(
            self.problem,
            self.repo,
            self.expert_path,
            max_decompose_step=options.pop("max_decompose_step", 2),
            workdir=self.work,
            enable_verifier=options.pop("enable_verifier", False),
            **options,
        )

    def run_init_memory(self, **options):
        return init_memory(
            self.problem,
            self.repo,
            self.expert_path,
            max_decompose_step=options.pop("max_decompose_step", 2),
            enable_verifier=options.pop("enable_verifier", False),
            workdir=self.work,
            memory_dir=options.pop("memory_dir", self.root / "memory"),
            **options,
        )

    def test_memory_collects_cards(self):
        paths = self.run_init_memory(max_decompose_step=5, timeout=42)
        self.assertEqual(tuple(map(load_skill, paths)), tuple(reversed(self.cards)))
        self.assertEqual(
            [event[0] for event in self.events],
            ["init", "decompose/0", "decompose/1", "decompose/2"],
        )
        self.assertTrue(all(event[3] == 42 for event in self.events))
        self.assertEqual(set(paths), set((self.root / "memory").rglob("SKILL.md")))
        for step, path in enumerate(paths):
            self.assertTrue(path.is_relative_to(self.root / "memory"))
            source = self.work / f"decompose/{step}/SKILL.md"
            self.assertEqual(load_skill(path), load_skill(source))

    def test_memory_step_limit(self):
        paths = self.run_init_memory(max_decompose_step=1)
        self.assertEqual(tuple(map(load_skill, paths)), (self.cards[-1],))
        self.assertEqual([event[0] for event in self.events], ["init", "decompose/0"])

    def test_memory_naive_input(self):
        self.states = self.states[:1]
        self.assertEqual(self.run_init_memory(), ())
        self.assertEqual(list((self.root / "memory").iterdir()), [])
        self.assertEqual([event[0] for event in self.events], ["init", "decompose/0"])

    def test_memory_duplicate_ids(self):
        self.cards[0] = replace(self.cards[0], skill_id=self.cards[1].skill_id)
        memory = self.root / "memory"
        existing = save_skill(self.cards[0], memory / "existing" / "SKILL.md")
        contents = existing.read_bytes()
        paths = self.run_init_memory()
        self.assertEqual(tuple(map(load_skill, paths)), tuple(reversed(self.cards)))
        self.assertEqual(len(set(paths)), 2)
        self.assertNotIn(existing, paths)
        self.assertEqual(existing.read_bytes(), contents)

    def test_memory_keeps_successes(self):
        self.failed_stage = "decompose/1"
        with self.assertRaises(ValidationGateError):
            self.run_init_memory(enable_verifier=True, max_retries=1)
        paths = tuple((self.root / "memory").rglob("SKILL.md"))
        self.assertEqual(tuple(map(load_skill, paths)), (self.cards[-1],))
        self.assertEqual(
            [event[1] for event in self.events if event[0] == self.failed_stage],
            ["Decompose", "Verify", "Decompose", "Verify"],
        )

    def test_memory_requires_card(self):
        self.omit_skill = True
        with self.assertRaises(StructuredOutputError):
            self.run_init_memory()
        self.assertEqual(list((self.root / "memory").iterdir()), [])

    def test_memory_requires_kernel(self):
        def generate(runner, prompt, *, run_id):
            result = self.run_codex(runner, prompt, run_id=run_id)
            if run_id.startswith("Decompose-"):
                (runner.work_dir / "kernel.json").unlink()
            return result

        self.calls.side_effect = generate
        with self.assertRaises(FileNotFoundError):
            self.run_init_memory(max_decompose_step=1)
        self.assertEqual(list((self.root / "memory").iterdir()), [])

    def test_memory_preserves_problem(self):
        current = self.states[1]
        self.states = (
            self.states[0],
            replace(current, problem=replace(current.problem, name="different")),
            self.states[2],
        )
        with self.assertRaisesRegex(StructuredOutputError, "problem"):
            self.run_init_memory(max_decompose_step=1)
        self.assertEqual(list((self.root / "memory").iterdir()), [])

    def test_memory_requires_removal(self):
        self.states = self.states[:1]

        def generate(runner, prompt, *, run_id):
            result = self.run_codex(runner, prompt, run_id=run_id)
            if run_id.startswith("Decompose-"):
                save_skill(self.cards[0], runner.work_dir / "SKILL.md")
            return result

        self.calls.side_effect = generate
        with self.assertRaisesRegex(StructuredOutputError, "unchanged"):
            self.run_init_memory(max_decompose_step=1)
        self.assertEqual(list((self.root / "memory").iterdir()), [])

    def test_memory_nested_dir(self):
        memory = self.work / "memory"
        paths = self.run_init_memory(memory_dir=memory)
        self.assertTrue(all(path.is_relative_to(memory) for path in paths))

    def test_memory_invalid_steps(self):
        for steps in (0, -1, True, 1.5):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                self.run_init_memory(max_decompose_step=steps)
        self.assertFalse(self.work.exists())
        self.calls.assert_not_called()

    def test_memory_existing_workdir(self):
        self.work.mkdir()
        marker = self.work / "keep.txt"
        marker.write_text("existing artifacts")
        with self.assertRaises(FileExistsError):
            self.run_init_memory()
        self.assertEqual(marker.read_text(), "existing artifacts")
        self.calls.assert_not_called()

    def test_memory_defaults(self):
        with patch("klineage.action.workflow.new_workdir", return_value=self.work):
            paths = init_memory(
                self.problem,
                self.repo,
                self.expert_path,
                max_decompose_step=1,
                memory_dir=self.root / "memory",
            )
        self.assertEqual(len(paths), 1)
        self.assertEqual(
            [(event[0], event[1]) for event in self.events],
            [("init", "Init"), ("decompose/0", "Decompose")],
        )

    def test_memory_safe_skill_paths(self):
        self.cards[-1] = replace(self.cards[-1], skill_id="../../outside")
        paths = self.run_init_memory(max_decompose_step=1)
        self.assertTrue(paths[0].is_relative_to(self.root / "memory"))
        self.assertEqual(load_skill(paths[0]), self.cards[-1])

    def test_memory_file_destination(self):
        destination = self.root / "memory"
        destination.write_text("existing file")
        with self.assertRaises(FileExistsError):
            self.run_init_memory()
        self.assertEqual(destination.read_text(), "existing file")
        self.calls.assert_not_called()

    def test_init_path_inputs(self):
        action = Init(
            self.problem,
            self.repo,
            self.expert_path,
            workdir=self.work / "init",
            enable_verifier=False,
        )
        data = prompt_inputs(action.prompt)
        self.assertEqual(data["problem"], str(self.problem))
        self.assertEqual(data["expert_kernel"], str(self.expert_path))
        self.calls.assert_not_called()

    def test_default_runs_verifier(self):
        workflow(
            self.problem,
            self.repo,
            self.expert_path,
            max_decompose_step=1,
            max_apply_step=0,
            workdir=self.work,
        )
        self.assertTrue(any(event[1] == "Verify" for event in self.events))

    def test_artifact_handoff(self):
        result = self.run_workflow(max_apply_step=2, timeout=42)
        self.assertEqual(
            [event[0] for event in self.events],
            [
                "init",
                "decompose/0",
                "decompose/1",
                "profile/0",
                "retrieve/0",
                "apply/0",
                "profile/1",
                "retrieve/1",
                "apply/1",
                "profile/2",
            ],
        )
        self.assertTrue(result.source_files["kernel.cu"].endswith(" // tile // mma"))
        self.assertEqual(result, load_kernel(self.work / "profile/2"))
        self.assertTrue((self.work / "profile/2/metrics.csv").is_file())
        self.assertTrue(all(event[3] == 42 for event in self.events))
        data = {stage: value for stage, _, value, _ in self.events}
        self.assertEqual(data["decompose/0"]["input_kernel"], str(self.work / "init"))
        self.assertEqual(
            data["decompose/1"]["input_kernel"], str(self.work / "decompose/0")
        )
        self.assertEqual(data["profile/0"]["kernel"], str(self.work / "decompose/1"))
        self.assertEqual(
            data["retrieve/1"]["current_kernel"], str(self.work / "profile/1")
        )
        self.assertEqual(
            data["apply/1"]["current_kernel"], str(self.work / "profile/1")
        )
        self.assertEqual(data["profile/2"]["kernel"], str(self.work / "apply/1"))
        self.assertEqual(data["retrieve/1"]["exclude_skills"], ["tile"])
        self.assertEqual(
            data["retrieve/1"]["skills"],
            [
                str(self.work / "decompose/0/SKILL.md"),
                str(self.work / "decompose/1/SKILL.md"),
            ],
        )
        for step in range(2):
            card = load_skill(self.work / f"decompose/{step}/SKILL.md")
            self.assertEqual(card, self.cards[1 - step])

    def test_complete_stops_decompose(self):
        self.run_workflow(max_decompose_step=5, max_apply_step=0)
        self.assertEqual(
            [event[0] for event in self.events],
            ["init", "decompose/0", "decompose/1", "decompose/2", "profile/0"],
        )
        self.assertEqual(
            load_kernel(self.work / "profile/0").source_files["kernel.cu"],
            self.states[0].source_files["kernel.cu"],
        )

    def test_decompose_budget(self):
        self.run_workflow(max_decompose_step=1, max_apply_step=0)
        self.assertEqual(
            [event[0] for event in self.events], ["init", "decompose/0", "profile/0"]
        )
        self.assertEqual(
            load_kernel(self.work / "profile/0").source_files["kernel.cu"],
            self.states[1].source_files["kernel.cu"],
        )

    def test_changed_kernel_needs_card(self):
        self.omit_skill = True
        with self.assertRaises(StructuredOutputError):
            self.run_workflow(max_apply_step=0)
        self.assertFalse((self.work / "profile").exists())

    def test_naive_input_stops(self):
        self.states = self.states[:1]
        result = self.run_workflow(max_decompose_step=5, max_apply_step=0)
        self.assertEqual(result, self.states[0])
        self.assertEqual(
            [event[0] for event in self.events], ["init", "decompose/0", "profile/0"]
        )
        self.assertFalse((self.work / "decompose/0/SKILL.md").exists())

    def test_apply_budget(self):
        result = self.run_workflow(max_apply_step=1)
        self.assertTrue(result.source_files["kernel.cu"].endswith(" // tile"))
        self.assertEqual(self.events[-1][0], "profile/1")
        self.assertFalse((self.work / "retrieve/1").exists())

    def test_verification_handoff(self):
        self.run_workflow(max_apply_step=2, enable_verifier=True)
        self.assertEqual(len(self.events), 20)
        for generation, verification in zip(self.events[::2], self.events[1::2]):
            self.assertEqual(generation[0], verification[0])
            self.assertEqual(verification[1], "Verify")
        self.assertTrue(load_kernel(self.work / "decompose/1").validation.accepted)
        self.assertTrue(load_kernel(self.work / "profile/0").validation.accepted)

    def test_failure_stops_workflow(self):
        self.failed_stage = "decompose/1"
        with self.assertRaises(ValidationGateError):
            self.run_workflow(max_apply_step=2, enable_verifier=True, max_retries=0)
        self.assertEqual(
            [event[0] for event in self.events],
            [
                "init",
                "init",
                "decompose/0",
                "decompose/0",
                "decompose/1",
                "decompose/1",
            ],
        )

    def test_empty_selection(self):
        self.empty_at = 0
        result = self.run_workflow(max_apply_step=3)
        self.assertEqual(result, load_kernel(self.work / "profile/0"))
        self.assertEqual(self.events[-1][0], "retrieve/0")
        self.assertFalse((self.work / "apply").exists())

    def test_stops_after_last_skill(self):
        result = self.run_workflow(max_apply_step=5)
        self.assertTrue(result.source_files["kernel.cu"].endswith(" // tile // mma"))
        self.assertEqual(self.events[-1][0], "retrieve/2")
        self.assertEqual(self.events[-1][2]["exclude_skills"], ["tile", "mma"])
        self.assertFalse((self.work / "apply/2").exists())

    def test_missing_output_fails(self):
        self.missing_stage = "apply/0"
        with self.assertRaises(FileNotFoundError):
            self.run_workflow(max_apply_step=2)
        self.assertFalse((self.work / "retrieve/1").exists())

    def test_existing_dir_is_kept(self):
        self.work.mkdir()
        marker = self.work / "keep.txt"
        marker.write_text("existing artifacts")
        with self.assertRaises(FileExistsError):
            self.run_workflow()
        self.assertEqual(marker.read_text(), "existing artifacts")
        self.calls.assert_not_called()

    def test_invalid_steps_fail(self):
        for key, values in (
            ("max_decompose_step", (0, -1, True)),
            ("max_apply_step", (-1, True, 1.5)),
        ):
            for steps in values:
                with self.subTest(key=key, steps=steps), self.assertRaises(ValueError):
                    self.run_workflow(**{key: steps})
        self.calls.assert_not_called()


if __name__ == "__main__":
    unittest.main()

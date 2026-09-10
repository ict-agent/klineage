import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kernel_fixtures import accepted, kernel, skill
from prompt_fixtures import prompt_inputs

from klineage.cli.common import read_step
from klineage.cli.init_memory import init_memory
from klineage.cli.optimize import optimize
from klineage.cli.workflow import workflow
from klineage.constants import RunKind
from klineage.contract import ValueRole
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
        elif run_id == "Apply":
            current = load_kernel(Path(data["current_kernel"]))
            self.assertNotIn("skill", data)
            memory = runner.work_dir / ".agents/skills/memory"
            index = int(runner.work_dir.name)
            if "memory" not in data:
                self.assertFalse(memory.exists())
            else:
                self.assertEqual(data["memory"], str(memory))
                self.assertTrue(memory.is_dir())
            if index != self.empty_at:
                current = replace(
                    current,
                    source_files={
                        "kernel.cu": current.source_files["kernel.cu"]
                        + f" // round-{index}"
                    },
                    validation=None,
                )
            save_kernel(current, runner.work_dir)
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
        with patch("klineage.cli.init_memory.new_workdir", return_value=self.work):
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

    def test_artifact_handoff(self):
        result = self.run_workflow(max_apply_step=2, timeout=42)
        self.assertEqual(
            [event[0] for event in self.events],
            [
                "init",
                "decompose/0",
                "decompose/1",
                "apply/0",
                "apply/1",
            ],
        )
        self.assertTrue(
            result.source_files["kernel.cu"].endswith(" // round-0 // round-1")
        )
        self.assertEqual(result, load_kernel(self.work / "apply/1"))
        self.assertTrue(all(event[3] == 42 for event in self.events))
        data = {stage: value for stage, _, value, _ in self.events}
        self.assertEqual(data["decompose/0"]["input_kernel"], str(self.work / "init"))
        self.assertEqual(
            data["decompose/1"]["input_kernel"], str(self.work / "decompose/0")
        )
        self.assertEqual(
            data["apply/0"]["current_kernel"], str(self.work / "decompose/1")
        )
        self.assertEqual(data["apply/1"]["current_kernel"], str(self.work / "apply/0"))
        self.assertEqual(data["apply/1"]["exclude_skills"], [])
        self.assertEqual(
            data["apply/1"]["memory"],
            str(self.work / "apply/1/.agents/skills/memory"),
        )
        self.assertEqual(
            Path(data["apply/1"]["memory"]).resolve(), self.work / "memory"
        )
        for step in range(2):
            card = load_skill(self.work / f"decompose/{step}/SKILL.md")
            self.assertEqual(card, self.cards[1 - step])
            self.assertEqual(load_skill(self.work / f"memory/{step}/SKILL.md"), card)
            self.assertFalse((self.work / f"apply/{step}/SKILL.md").exists())

    def test_complete_stops_decompose(self):
        result = self.run_workflow(max_decompose_step=5, max_apply_step=0)
        self.assertEqual(
            [event[0] for event in self.events],
            ["init", "decompose/0", "decompose/1", "decompose/2"],
        )
        self.assertEqual(
            result.source_files["kernel.cu"],
            self.states[0].source_files["kernel.cu"],
        )

    def test_decompose_budget(self):
        result = self.run_workflow(max_decompose_step=1, max_apply_step=0)
        self.assertEqual([event[0] for event in self.events], ["init", "decompose/0"])
        self.assertEqual(
            result.source_files["kernel.cu"],
            self.states[1].source_files["kernel.cu"],
        )

    def test_apply_budget(self):
        result = self.run_workflow(max_apply_step=1)
        self.assertTrue(result.source_files["kernel.cu"].endswith(" // round-0"))
        self.assertEqual(self.events[-1][0], "apply/0")
        self.assertFalse((self.work / "apply/1").exists())

    def test_verification_handoff(self):
        workflow(
            self.problem,
            self.repo,
            self.expert_path,
            max_decompose_step=2,
            max_apply_step=2,
            workdir=self.work,
        )
        self.assertEqual(len(self.events), 10)
        for generation, verification in zip(self.events[::2], self.events[1::2]):
            self.assertEqual(generation[0], verification[0])
            self.assertEqual(verification[1], "Verify")
        self.assertTrue(load_kernel(self.work / "decompose/1").validation.accepted)

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

    def test_unchanged_round_stops(self):
        self.empty_at = 0
        result = self.run_workflow(max_apply_step=3)
        self.assertEqual(result, load_kernel(self.work / "apply/0"))
        self.assertEqual(result, load_kernel(self.work / "decompose/1"))
        self.assertEqual(self.events[-1][0], "apply/0")
        self.assertFalse((self.work / "apply/0/SKILL.md").exists())
        self.assertFalse((self.work / "apply/1").exists())

    def test_missing_output_fails(self):
        self.missing_stage = "apply/0"
        with self.assertRaises(FileNotFoundError):
            self.run_workflow(max_apply_step=2)
        self.assertFalse((self.work / "apply/1").exists())

    def test_apply_checks_artifacts(self):
        start = self.root / "start"
        start.mkdir()
        save_kernel(self.states[0], start)
        memory = self.root / "memory"
        save_skill(self.cards[0], memory / "tile" / "SKILL.md")
        for directory in (None, memory):
            for change in ("problem", "card", "name"):
                self.work = self.root / f"{directory is None}-{change}"
                self.empty_at = 0 if change == "name" else None

                def generate(runner, prompt, *, run_id, change=change):
                    result = self.run_codex(runner, prompt, run_id=run_id)
                    candidate = load_kernel(runner.work_dir)
                    if change == "card":
                        save_skill(self.cards[0], runner.work_dir / "SKILL.md")
                    elif change == "problem":
                        candidate = replace(
                            candidate,
                            problem=replace(candidate.problem, name="changed"),
                        )
                    else:
                        candidate = replace(candidate, name="changed")
                    save_kernel(candidate, runner.work_dir)
                    return result

                self.calls.side_effect = generate
                with (
                    self.subTest(memory=directory, change=change),
                    self.assertRaises(StructuredOutputError),
                ):
                    optimize(
                        start,
                        directory,
                        self.work,
                        max_apply_step=1,
                        enable_verifier=False,
                    )

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

    def test_optimize_from_kernel_file(self):
        start = self.root / "start"
        start.mkdir()
        save_kernel(self.states[0], start)
        memory = self.root / "memory"
        for index, card in enumerate(self.cards):
            save_skill(card, memory / str(index) / "SKILL.md")

        self.empty_at = 2
        result = optimize(
            start / "kernel.json",
            memory,
            self.work,
            max_apply_step=5,
            enable_verifier=False,
            timeout=42,
        )
        self.assertTrue(
            result.source_files["kernel.cu"].endswith(" // round-0 // round-1")
        )
        self.assertEqual(
            [event[0] for event in self.events], ["apply/0", "apply/1", "apply/2"]
        )
        self.assertEqual(Path(self.events[0][2]["memory"]).resolve(), memory)
        self.assertEqual(load_kernel(start), self.states[0])
        self.assertTrue(all(event[3] == 42 for event in self.events))

    def test_optimize_zero_budget(self):
        start = self.root / "start"
        start.mkdir()
        save_kernel(self.states[0], start)
        memory = self.root / "memory"
        memory.mkdir()
        self.assertEqual(
            optimize(start, memory, self.work, max_apply_step=0), self.states[0]
        )
        self.calls.assert_not_called()

    def test_optimize_validates_paths(self):
        start = self.root / "start"
        start.mkdir()
        save_kernel(self.states[0], start)
        memory = self.root / "memory"
        memory.mkdir()
        cases = (
            (self.root / "missing", memory, self.work),
            (start, self.root / "missing", self.work),
            (start, memory, memory),
        )
        for source, cards, work in cases:
            with (
                self.subTest(source=source, memory=cards, work=work),
                self.assertRaises((FileNotFoundError, FileExistsError, ValueError)),
            ):
                optimize(source, cards, work)
        self.calls.assert_not_called()

    def test_baseline_without_memory(self):
        start = self.root / "start"
        start.mkdir()
        save_kernel(self.states[0], start)
        self.empty_at = 2
        for index, memory in enumerate((None, "", " \t ")):
            self.work = self.root / f"baseline-{index}"
            self.events.clear()
            with self.subTest(memory=memory):
                result = optimize(
                    start,
                    memory,
                    self.work,
                    max_apply_step=5,
                    enable_verifier=False,
                )
                self.assertTrue(
                    result.source_files["kernel.cu"].endswith(" // round-0 // round-1")
                )
                self.assertEqual(
                    [event[0] for event in self.events],
                    ["apply/0", "apply/1", "apply/2"],
                )
                self.assertFalse(list(self.work.rglob("SKILL.md")))

    def test_empty_memory_round(self):
        start = self.root / "start"
        start.mkdir()
        save_kernel(self.states[0], start)
        memory = self.root / "empty-memory"
        memory.mkdir()
        self.empty_at = 1
        result = optimize(
            start, memory, self.work, max_apply_step=3, enable_verifier=False
        )
        self.assertTrue(result.source_files["kernel.cu"].endswith(" // round-0"))
        self.assertEqual([event[0] for event in self.events], ["apply/0", "apply/1"])
        self.assertEqual(Path(self.events[0][2]["memory"]).resolve(), memory)

    def test_rejects_reordered_abi(self):
        for role in ValueRole:
            definition = dict(self.states[0].problem.definition)
            members = dict(definition[role])
            if role is ValueRole.OUTPUTS:
                members["auxiliary"] = members["output"]
            definition[role] = members
            before = replace(
                self.states[0],
                problem=replace(self.states[0].problem, definition=definition),
            )
            definition = {**definition, role: dict(reversed(tuple(members.items())))}
            after = replace(
                self.states[1], problem=replace(before.problem, definition=definition)
            )
            self.assertEqual(before.problem, after.problem)
            work = self.root / role
            work.mkdir()
            save_kernel(after, work)
            with (
                self.subTest(role=role),
                self.assertRaisesRegex(StructuredOutputError, "ABI"),
            ):
                read_step(before, work, RunKind.APPLY)

    def test_terminal_contract(self):
        for stage in (RunKind.DECOMPOSE, RunKind.APPLY):
            for artifact in ("card_dir", "card_link", "submission", "validation"):
                work = self.root / f"{stage}-{artifact}"
                work.mkdir()
                before = self.states[0]
                after = (
                    replace(before, validation=accepted())
                    if artifact == "validation"
                    else before
                )
                save_kernel(after, work)
                if artifact == "card_dir":
                    (work / "SKILL.md").mkdir()
                elif artifact == "card_link":
                    (work / "SKILL.md").symlink_to(work / "missing")
                elif artifact == "submission":
                    (work / "submission").mkdir()
                with self.subTest(stage=stage, artifact=artifact):
                    if stage is RunKind.DECOMPOSE and artifact == "validation":
                        self.assertEqual(read_step(before, work, stage), (after, None))
                        continue
                    with self.assertRaises(StructuredOutputError):
                        read_step(before, work, stage)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib
import inspect
import json
import tempfile
import unittest
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from klineage.action import apply, code_gen, decompose, init
from klineage.action.apply import _apply
from klineage.action.code_gen import _materialize
from klineage.action.decompose import _admit_skills, _decompose
from klineage.contract import EvaluatorInterface, KernelABI, ProblemSpec
from klineage.errors import ActionError, ValidationGateError
from klineage.harness.artifacts import (
    read_generated_source,
    read_generated_source_bundle,
)
from klineage.harness.eval import ValidationResult
from klineage.harness.structured import run_json
from klineage.kernel import Feature, Kernel, TargetContext
from klineage.memory import (
    Scope,
    SkillAdmission,
    SkillCard,
    TransitionEvidence,
    VerificationTrial,
    load_lineage,
    load_memory,
    save_lineage,
    save_memory,
)

CUDA_SOURCE = 'extern "C" __global__ void kernel(float *x) { x[0] += 1; }'
LOADER_SOURCE = "def load():\n    return lambda x: x\n"


def accepted(latency: float = 1.0, reference: float = 2.0) -> ValidationResult:
    def timing(value: float) -> dict[str, object]:
        return {
            "samples_ms": [[value], [value]], "median_ms": value,
            "backend_used": "cupti",
            "details": {
                "cupti_version": "13.0",
                "policy": {"warmup": 1, "repeat": 1, "trials": 2, "cold_l2": True},
            },
        }

    return ValidationResult(
        compile_passed=True,
        correctness_passed=True,
        profile_passed=True,
        latency_ms=latency,
        reference_latency_ms=reference,
        details={"timing": {"candidate": timing(latency), "reference": timing(reference)}},
    )


def rejected() -> ValidationResult:
    return ValidationResult(
        compile_passed=True,
        correctness_passed=False,
        profile_passed=False,
    )


def kernel(
    name: str = "kernel",
    *,
    case: str = "gemm",
    language: str = "cuda",
    platform: str = "sm120",
    actions: tuple[str, ...] = (),
    source: str = CUDA_SOURCE,
    validation: ValidationResult | None = None,
) -> Kernel:
    return Kernel(
        name=name,
        source=source,
        context=TargetContext(case, language, platform, actions),
        validation=validation,
    )


def materializable(value: Kernel, marker: str = "") -> Kernel:
    source = LOADER_SOURCE + marker
    return replace(
        value,
        source=source,
        problem=ProblemSpec("gemm", "Compute GEMM."),
        abi=KernelABI(),
        source_files={"submission.py": source, "kernel.cu": CUDA_SOURCE},
    )


def bundle(marker: str = "") -> dict[str, str]:
    return {
        "config.toml": '''[solution]
name = "gemm"
definition = "gemm"
author = "klineage"
[build]
language = "cuda"
entry_point = "binding.py::kernel"
destination_passing_style = false
''',
        "solution/binding.py": "def kernel(x):\n    return x\n" + marker,
        "solution/kernel.cu": CUDA_SOURCE,
    }


def skill(
    skill_id: str,
    verified: TargetContext | None,
    *,
    declared: Scope | None = None,
) -> SkillCard:
    before = kernel(name=f"{skill_id}-before", source=CUDA_SOURCE + " // before")
    after = kernel(name=f"{skill_id}-after", source=CUDA_SOURCE + " // after")
    evidence = TransitionEvidence(
        action_category=skill_id,
        locus="the output loop",
        before_fingerprint=before.fingerprint,
        after_fingerprint=after.fingerprint,
        backward_edit="remove optimization",
        forward_edit="restore optimization",
        predecessor_validation=accepted(2.0, 1.0),
        roundtrip_validation=accepted(1.0, 2.0),
    )
    trials = ()
    if verified is not None:
        trials = (
            VerificationTrial(
                target=verified,
                validation=accepted(),
                expected_effect_observed=True,
                held_out=True,
            ),
        )
    return SkillCard(
        skill_id=skill_id,
        intent=f"intent {skill_id}",
        anchor="output loop",
        carrier=f"apply {skill_id} here",
        preconditions=("precondition",),
        effects=("lower latency",),
        evidence=(evidence,),
        risks=("tail handling",),
        scope=declared or Scope(),
        verification_log=trials,
    )


class FakeRunner:
    def __init__(self, sandbox: Path, responses: Sequence[dict[str, object]]) -> None:
        self.sandbox_dir = sandbox.resolve()
        self.state_dir = self.sandbox_dir / ".klineage"
        self.state_dir.mkdir(parents=True)
        self.responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str, **_: object) -> SimpleNamespace:
        self.prompts.append(prompt)
        response = dict(self.responses.pop(0))
        source_to_write = response.pop("_write_source", None)
        if source_to_write is not None:
            payload = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
            output_path = self.sandbox_dir / payload["output_path"]
            output_path.write_text(str(source_to_write), encoding="utf-8")
        files_to_write = response.pop("_write_bundle", None)
        if files_to_write is not None:
            payload = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
            submission = self.sandbox_dir / payload["submission_dir"]
            for relative, source in files_to_write.items():
                path = submission / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source, encoding="utf-8")
        trace_path = self.state_dir / f"trace-{len(self.prompts)}.jsonl"
        trace_path.touch()
        return SimpleNamespace(
            final_message=json.dumps(response),
            trace_path=trace_path,
        )


class SourceEvaluator:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[Kernel, Kernel | None]] = []

    def evaluate(
        self,
        candidate: Kernel,
        *,
        reference: Kernel | None = None,
    ) -> ValidationResult:
        self.calls.append((candidate, reference))
        if self.fail:
            return rejected()
        if "naive" in candidate.source:
            return accepted(2.0, 1.0)
        return accepted(1.0, 2.0)


class FakeSandbox:
    def __init__(
        self,
        runner: FakeRunner,
        evaluator: SourceEvaluator | None = None,
        *,
        repo: Path | None = None,
    ) -> None:
        self._runner = runner
        self._evaluator = evaluator or SourceEvaluator()
        self._repo = (repo or runner.sandbox_dir / "repository").resolve()
        self._repo.mkdir(parents=True, exist_ok=True)
        self._logs = runner.state_dir / "evaluations"
        self._logs.mkdir(parents=True, exist_ok=True)
        self._artifacts = runner.state_dir / "artifacts"
        self._artifacts.mkdir(parents=True, exist_ok=True)

    def _file(self, name: str) -> str:
        path = self._runner.sandbox_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return name

    def _dir(self, name: str) -> str:
        (self._runner.sandbox_dir / name).mkdir(parents=True)
        return name

    def _snapshot(self, sources: dict[str, str]) -> str:
        root = Path(tempfile.mkdtemp(prefix="audit-", dir=self._runner.sandbox_dir))
        for name, source in sources.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        return str(root)

    def _ask(
        self,
        purpose: str,
        instructions: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        return run_json(self._runner, purpose, instructions, payload)  # type: ignore[arg-type]

    def _take_file(self, name: str, label: str) -> tuple[str, Path]:
        source = read_generated_source(
            self._runner.sandbox_dir / name,
            label,
            allowed_root=self._runner.sandbox_dir,
        )
        artifact = self._artifacts / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(source, encoding="utf-8")
        return source, artifact.resolve()

    def _take_bundle(
        self,
        name: str,
        label: str,
        interface: EvaluatorInterface,
    ) -> tuple[dict[str, str], Path]:
        files = read_generated_source_bundle(
            self._runner.sandbox_dir / name,
            label,
            allowed_root=self._runner.sandbox_dir,
            interface=interface,
        )
        artifact = self._artifacts / name
        artifact.mkdir(parents=True)
        for relative, source in files.items():
            path = artifact / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        return files, artifact.resolve()

    def _evaluate(
        self,
        candidate: Kernel,
        *,
        reference: Kernel | None = None,
    ) -> ValidationResult:
        return self._evaluator.evaluate(candidate, reference=reference)


class ActionTests(unittest.TestCase):
    def test_init_public_api_has_only_three_inputs(self) -> None:
        self.assertEqual(
            tuple(inspect.signature(init).parameters),
            ("problem", "repo", "expert_kernel"),
        )

    def test_actions_hide_infrastructure(self) -> None:
        self.assertEqual(
            tuple(inspect.signature(code_gen).parameters),
            ("current_kernel", "skills"),
        )
        self.assertEqual(
            tuple(inspect.signature(apply).parameters),
            ("current_kernel", "skills", "skill_admission"),
        )
        self.assertEqual(
            tuple(inspect.signature(decompose).parameters),
            (
                "expert_kernel",
                "max_steps",
                "max_rejections",
                "roundtrip_cases",
                "effect_verifier",
                "skill_admission",
            ),
        )
        self.assertIs(
            inspect.signature(apply).parameters["skill_admission"].default,
            SkillAdmission.OFF,
        )
        self.assertIs(
            inspect.signature(decompose).parameters["skill_admission"].default,
            SkillAdmission.OFF,
        )

    def test_init_snapshots_an_external_problem_inside_the_action(self) -> None:
        module = importlib.import_module("klineage.action._sandbox")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "outside" / "reference.py"
            source.parent.mkdir()
            source.write_text("def torch_ref():\n    pass\n", encoding="utf-8")
            inputs = root / "input"
            inputs.mkdir()

            snapshot = module._copy_problem(source, inputs)

            self.assertEqual(
                snapshot,
                (inputs / "problem" / "outside" / "reference.py").resolve(),
            )
            self.assertEqual(snapshot.parent.name, source.parent.name)
            self.assertEqual(snapshot.name, source.name)
            self.assertEqual(snapshot.read_text(encoding="utf-8"), source.read_text())
            self.assertEqual(tuple(inspect.signature(module._project_root).parameters), ())

    def test_init_snapshots_an_external_expert_inside_the_repo(self) -> None:
        module = importlib.import_module("klineage.action.init")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repository"
            repository.mkdir()
            source = root / "expert.cu"
            source.write_text(CUDA_SOURCE, encoding="utf-8")

            snapshot = module._expert_path(repository, source)

            self.assertEqual(snapshot.parent, repository / ".klineage-expert")
            self.assertEqual(snapshot.name, source.name)
            self.assertEqual(snapshot.read_text(encoding="utf-8"), CUDA_SOURCE)

    def test_code_gen_materializes_selected_skills(self) -> None:
        current = materializable(kernel())
        card = skill("tile", TargetContext("gemm", "cuda", "sm120"))
        with tempfile.TemporaryDirectory() as temporary:
            runner = FakeRunner(
                Path(temporary),
                [{"done": True, "_write_bundle": bundle("# generated\n")}],
            )
            sandbox = FakeSandbox(runner)

            generated = _materialize(sandbox, current, (card,))  # type: ignore[arg-type]

            self.assertIn("generated", generated.source)
            payload = json.loads(runner.prompts[0].split("INPUT_JSON:\n", 1)[1])
            self.assertEqual(
                payload["current_kernel"],
                {
                    "context": current.context.to_dict(),
                    "problem": current.problem.to_dict(),
                    "abi": current.abi.to_dict(),
                    "source_files": current.source_files,
                },
            )
            self.assertEqual(
                payload["skill_cards"],
                [{
                    "intent": card.intent,
                    "anchor": card.anchor,
                    "locus": card.evidence[0].locus,
                    "forward_edit": card.evidence[0].forward_edit,
                    "carrier": card.carrier,
                    "precondition": list(card.preconditions),
                    "effect": list(card.effects),
                    "risk": list(card.risks),
                    "scope": card.scope.to_dict(),
                }],
            )

            with self.assertRaisesRegex(ValueError, "at least one"):
                code_gen(current, ())

    def test_skill_admission_requires_a_successful_heldout_trial(self) -> None:
        context = TargetContext("gemm", "cuda", "sm120", ("tile",))
        admitted = skill("tile", context)
        non_heldout_trial = replace(
            admitted.verification_log[0],
            held_out=False,
        )
        hypothesis = replace(admitted, verification_log=(non_heldout_trial,))

        self.assertTrue(admitted.admitted)
        self.assertFalse(hypothesis.admitted)

        serialized = admitted.to_dict()
        for paper_field in (
            "intent",
            "anchor",
            "carrier",
            "precondition",
            "effect",
            "evidence",
            "risk",
            "scope",
            "verification_log",
        ):
            self.assertIn(paper_field, serialized)
        self.assertEqual(SkillCard.from_dict(serialized), admitted)

        malformed = admitted.to_dict()
        malformed["verification_log"][0]["held_out"] = "false"
        with self.assertRaises(TypeError):
            SkillCard.from_dict(malformed)

    def test_memory_storage_accepts_only_flat_unique_cards(self) -> None:
        card = skill("tile", TargetContext("gemm", "cuda", "sm120"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "memory.json"
            save_memory((card, card), path)

            self.assertEqual(load_memory(path), (card,))
            with self.assertRaises(TypeError):
                save_memory([[card]], path)  # type: ignore[list-item]
            with self.assertRaises(ValueError):
                save_memory((card, replace(card, intent="conflict")), path)

    def test_decompose_admission_switch_requires_explicit_heldout_cases(self) -> None:
        expert = kernel(validation=accepted())
        with tempfile.TemporaryDirectory() as temporary:
            runner = FakeRunner(Path(temporary), [])
            sandbox = FakeSandbox(runner)
            with self.assertRaisesRegex(ValueError, "requires at least one"):
                _decompose(
                    sandbox,  # type: ignore[arg-type]
                    expert,
                    max_steps=1,
                    max_rejections=1,
                    roundtrip_cases=(),
                    effect_verifier=None,
                    skill_admission=SkillAdmission.ON,
                )
            with self.assertRaisesRegex(ValueError, "require skill admission"):
                _decompose(
                    sandbox,  # type: ignore[arg-type]
                    expert,
                    max_steps=1,
                    max_rejections=1,
                    roundtrip_cases=(kernel(name="heldout"),),
                    effect_verifier=None,
                    skill_admission=SkillAdmission.OFF,
                )
            with self.assertRaisesRegex(ValueError, "explicit effect_verifier"):
                _decompose(
                    sandbox,  # type: ignore[arg-type]
                    expert,
                    max_steps=1,
                    max_rejections=1,
                    roundtrip_cases=(kernel(name="heldout"),),
                    effect_verifier=None,
                    skill_admission=SkillAdmission.ON,
                )

    def test_decompose_stops_at_the_rejection_budget(self) -> None:
        class RejectPredecessors(SourceEvaluator):
            def evaluate(
                self,
                candidate: Kernel,
                *,
                reference: Kernel | None = None,
            ) -> ValidationResult:
                self.calls.append((candidate, reference))
                return accepted() if reference is None else rejected()

        responses = [
            {
                "done": False,
                "action_category": "tile",
                "locus": "output loop",
                "backward_edit": "remove tiling",
                "_write_source": CUDA_SOURCE + f" // rejected {index}",
            }
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            runner = FakeRunner(Path(temporary), responses)
            sandbox = FakeSandbox(runner, RejectPredecessors())
            lineage = _decompose(
                sandbox,  # type: ignore[arg-type]
                kernel(validation=accepted()),
                max_steps=4,
                max_rejections=2,
                roundtrip_cases=(),
                effect_verifier=None,
                skill_admission=SkillAdmission.OFF,
            )

        self.assertEqual(len(runner.prompts), 2)
        self.assertEqual(len(lineage.transitions), 0)

    def test_decompose_rechecks_serialized_expert_validation(self) -> None:
        expert = kernel(validation=accepted())
        with tempfile.TemporaryDirectory() as temporary:
            runner = FakeRunner(Path(temporary), [{"done": True}])
            evaluator = SourceEvaluator(fail=True)
            sandbox = FakeSandbox(runner, evaluator)

            with self.assertRaisesRegex(ValidationGateError, "expert kernel"):
                _decompose(
                    sandbox,  # type: ignore[arg-type]
                    expert,
                    max_steps=1,
                    max_rejections=1,
                    roundtrip_cases=(),
                    effect_verifier=None,
                    skill_admission=SkillAdmission.OFF,
                )

        self.assertEqual(len(evaluator.calls), 1)
        self.assertEqual(runner.prompts, [])

    def test_admission_rechecks_serialized_baseline_validation(self) -> None:
        baseline = materializable(
            kernel(case="heldout", validation=accepted())
        )
        card = skill("tile", TargetContext("gemm", "cuda", "sm120"))
        with tempfile.TemporaryDirectory() as temporary:
            runner = FakeRunner(
                Path(temporary),
                [{"done": True, "_write_bundle": bundle("# generated\n")}],
            )
            evaluator = SourceEvaluator(fail=True)
            sandbox = FakeSandbox(runner, evaluator)

            with (
                patch(
                    "klineage.action.decompose._next",
                    return_value=nullcontext(sandbox),
                ),
                self.assertRaisesRegex(ValidationGateError, "held-out baseline"),
            ):
                _admit_skills(
                    (card,),
                    states=(kernel(validation=accepted()),),
                    cases=(baseline,),
                    effect_verifier=lambda *_: True,
                )

        self.assertEqual(len(evaluator.calls), 1)
        self.assertEqual(runner.prompts, [])

    def test_apply_generates_once_then_enforces_the_external_gate(self) -> None:
        current = materializable(kernel(validation=accepted(2.0, 2.0)))
        card = skill("tile", TargetContext("gemm", "cuda", "sm120"))
        evaluator = SourceEvaluator()
        with tempfile.TemporaryDirectory() as temporary:
            runner = FakeRunner(
                Path(temporary),
                [{"done": True, "_write_bundle": bundle("# optimized\n")}],
            )
            sandbox = FakeSandbox(runner, evaluator)

            result = _apply(
                sandbox,  # type: ignore[arg-type]
                current,
                (card,),
                SkillAdmission.OFF,
            )

        self.assertEqual(len(runner.prompts), 1)
        self.assertEqual(len(evaluator.calls), 1)
        self.assertIs(evaluator.calls[0][1], current)
        self.assertTrue(result.validation.accepted)

    def test_apply_never_accepts_a_failed_gate(self) -> None:
        current = materializable(kernel())
        card = skill("tile", TargetContext("gemm", "cuda", "sm120"))
        with tempfile.TemporaryDirectory() as temporary:
            runner = FakeRunner(
                Path(temporary),
                [{"done": True, "_write_bundle": bundle("# broken\n")}],
            )
            sandbox = FakeSandbox(runner, SourceEvaluator(fail=True))

            with self.assertRaises(ValidationGateError):
                _apply(
                    sandbox,  # type: ignore[arg-type]
                    current,
                    (card,),
                    SkillAdmission.OFF,
                )

    def test_apply_rejects_unadmitted_cards_when_admission_is_enabled(self) -> None:
        current = kernel()
        admitted = skill("tile", TargetContext("gemm", "cuda", "sm120"))
        hypothesis = replace(
            admitted,
            verification_log=(replace(admitted.verification_log[0], held_out=False),),
        )

        with tempfile.TemporaryDirectory() as temporary:
            runner = FakeRunner(Path(temporary), [])
            sandbox = FakeSandbox(runner)

            with self.assertRaises(ActionError):
                _apply(
                    sandbox,  # type: ignore[arg-type]
                    current,
                    (hypothesis,),
                    SkillAdmission.ON,
                )
            self.assertEqual(runner.prompts, [])

    def test_skill_admission_is_disabled_by_default(self) -> None:
        current = kernel()
        admitted = skill("tile", TargetContext("gemm", "cuda", "sm120"))
        hypothesis = replace(
            admitted,
            verification_log=(replace(admitted.verification_log[0], held_out=False),),
        )

        current = materializable(current)
        with tempfile.TemporaryDirectory() as temporary:
            runner = FakeRunner(
                Path(temporary),
                [
                    {
                        "done": True,
                        "_write_bundle": bundle("# hypothesis applied\n"),
                    }
                ],
            )
            sandbox = FakeSandbox(runner)
            result = _apply(
                sandbox,  # type: ignore[arg-type]
                current,
                (hypothesis,),
                SkillAdmission.OFF,
            )

        self.assertTrue(result.validation.accepted)

    def test_init_verifies_against_the_explicit_expert(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = FakeRunner(
                root / "sandbox",
                [
                    {
                        "_write_source": CUDA_SOURCE,
                        "done": True,
                    }
                ],
            )
            source_repo = runner.sandbox_dir / "repository"
            source_repo.mkdir()
            (source_repo / "provided.cu").write_text(CUDA_SOURCE)
            evaluator = SourceEvaluator()
            sandbox = FakeSandbox(runner, evaluator, repo=source_repo)
            problem = ProblemSpec("gemm", "Compute GEMM.")
            abi = KernelABI()
            expert = replace(
                kernel(name="expert", source=CUDA_SOURCE + " // expert"),
                problem=problem,
                abi=abi,
            )

            module = importlib.import_module("klineage.action.init")
            with patch.object(module, "verify_mechanism", return_value={"passed": True}):
                result = module._generate(sandbox, expert)

            self.assertTrue(result.validation.accepted)
            self.assertGreaterEqual(result.validation.relative_performance, 0.9)
            self.assertTrue(result.artifact_path.is_file())
            payload = json.loads(runner.prompts[0].split("INPUT_JSON:\n", 1)[1])
            staged = runner.sandbox_dir / payload["repository"]
            self.assertEqual((staged / "provided.cu").read_text(), CUDA_SOURCE)
            self.assertEqual(payload["problem"], problem.to_dict())
            self.assertEqual(payload["kernel_abi"], abi.to_dict())
            self.assertEqual(payload["minimum_expert_ratio"], 0.99)
            self.assertIsNone(evaluator.calls[0][1])
            self.assertEqual(evaluator.calls[1][1].fingerprint, expert.fingerprint)
            self.assertEqual(len(evaluator.calls), 3)
            self.assertTrue(result.validation.details["confirmation"]["independent"])
            self.assertTrue((sandbox._logs / "fidelity-01.json").is_file())

    def test_init_retries_after_a_non_raw_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = FakeRunner(
                root / "sandbox",
                [
                    {"_write_source": "#include <cutlass/cutlass.h>\n", "done": True},
                    {"_write_source": CUDA_SOURCE, "done": True},
                ],
            )
            source_repo = runner.sandbox_dir / "repository"
            source_repo.mkdir()
            problem = ProblemSpec("gemm", "Compute GEMM.")
            abi = KernelABI()
            expert = replace(
                kernel(name="expert", source=CUDA_SOURCE + " // expert"),
                problem=problem,
                abi=abi,
            )
            sandbox = FakeSandbox(runner, repo=source_repo)

            module = importlib.import_module("klineage.action.init")
            with patch.object(module, "verify_mechanism", return_value={"passed": True}):
                result = module._generate(sandbox, expert)

            self.assertTrue(result.validation.accepted)
            payload = json.loads(runner.prompts[1].split("INPUT_JSON:\n", 1)[1])
            self.assertIn("artifact_gate", payload["previous_gate_feedback"])

    def test_init_rejects_a_candidate_slower_than_the_expert(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            module = importlib.import_module("klineage.action.init")
            runner = FakeRunner(
                root / "sandbox",
                [
                    {
                        "_write_source": CUDA_SOURCE + f" // attempt {attempt}",
                        "done": True,
                    }
                    for attempt in range(1, module._MAX_ATTEMPTS + 1)
                ],
            )
            source_repo = runner.sandbox_dir / "repository"
            source_repo.mkdir()
            (source_repo / "provided.cu").write_text(CUDA_SOURCE)
            problem = ProblemSpec("gemm", "Compute GEMM.")
            abi = KernelABI()
            expert = replace(
                kernel(name="expert", source=CUDA_SOURCE + " // expert"),
                problem=problem,
                abi=abi,
            )

            class SlowCandidateEvaluator(SourceEvaluator):
                def evaluate(
                    self,
                    candidate: Kernel,
                    *,
                    reference: Kernel | None = None,
                ) -> ValidationResult:
                    self.calls.append((candidate, reference))
                    return accepted(2.0, 1.0) if reference else accepted(1.0, 1.0)

            evaluator = SlowCandidateEvaluator()
            sandbox = FakeSandbox(runner, evaluator, repo=source_repo)
            with self.assertRaises(ValidationGateError) as raised:
                module._generate(sandbox, expert)

            validation = raised.exception.kernel.validation
            self.assertFalse(validation.profile_passed)
            self.assertEqual(validation.relative_performance, 0.5)
            self.assertEqual(
                validation.details["init_verifier"]["minimum_expert_ratio"],
                0.99,
            )

    def test_init_rechecks_performance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = FakeRunner(Path(temporary), [{"_write_source": CUDA_SOURCE, "done": True}])
            sandbox = FakeSandbox(runner)
            expert = replace(kernel(), problem=ProblemSpec("gemm", "GEMM"), abi=KernelABI())
            module = importlib.import_module("klineage.action.init")
            with (
                patch.object(module, "_MAX_ATTEMPTS", 1),
                patch.object(module, "verify_mechanism", return_value={"passed": True}),
                patch.object(sandbox, "_evaluate", side_effect=[
                    accepted(1.0, 1.0), accepted(1.0, 1.0), accepted(1.0, 0.98),
                ]),
                self.assertRaises(ValidationGateError) as raised,
            ):
                module._generate(sandbox, expert)
            validation = raised.exception.kernel.validation
            self.assertFalse(validation.accepted)
            self.assertTrue(validation.details["confirmation"]["independent"])
            self.assertTrue(validation.details["confirmation"]["selection"]["profile_passed"])

    def test_init_requires_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = FakeRunner(Path(temporary), [{"_write_source": CUDA_SOURCE, "done": True}])
            sandbox = FakeSandbox(runner)
            expert = replace(kernel(), problem=ProblemSpec("gemm", "GEMM"), abi=KernelABI())
            module = importlib.import_module("klineage.action.init")
            with (
                patch.object(module, "_MAX_ATTEMPTS", 1),
                patch.object(module, "verify_mechanism", return_value={"passed": False, "error": "unknown"}),
                self.assertRaises(ValidationGateError) as raised,
            ):
                module._generate(sandbox, expert)
            validation = raised.exception.kernel.validation
            self.assertFalse(validation.accepted)
            self.assertEqual(validation.details["mechanism_verifier"]["error"], "unknown")
            self.assertEqual(len(sandbox._evaluator.calls), 2)

    def test_decompose_rederives_forward_transition_and_admits_on_heldout(self) -> None:
        feature = Feature("global load", "vectorize_global")
        responses = [
            {
                "done": False,
                "action_category": "vectorize_global",
                "locus": "global load",
                "_write_source": CUDA_SOURCE + " // naive",
                "backward_edit": "replace vector load with scalar load",
                "before_features": [],
                "after_features": [feature.to_dict()],
            },
            {
                "_write_source": CUDA_SOURCE + " // roundtrip optimized",
                "forward_edit": "restore aligned vector load",
                "carrier": "reinterpret as aligned float4 and keep a scalar tail",
            },
            {"done": True, "reason": "Only scalar loads and stores remain."},
            {
                "intent": "vectorize aligned contiguous global loads",
                "anchor": "the global-load expression",
                "carrier": "reinterpret as aligned float4 and keep a scalar tail",
                "preconditions": ["16-byte alignment", "safe tail handling"],
                "effects": ["fewer global load instructions"],
                "risks": ["misaligned vector access"],
                "requires": [],
                "provides": [feature.to_dict()],
                "conflicts": [feature.to_dict()],
                "scope": {
                    "cases": ["gemm"],
                    "languages": ["cuda"],
                    "platforms": ["sm120"],
                    "prior_actions": ["tile"],
                },
            },
            {
                "done": True,
                "_write_bundle": bundle("# heldout vectorized\n"),
            },
        ]

        def effect_observed(
            _: SkillCard,
            baseline: Kernel,
            candidate: Kernel,
        ) -> bool:
            return candidate.source != baseline.source

        def observe(_, candidate, features):
            present = () if "naive" in candidate.source else tuple(features)
            return replace(candidate, features=present)

        with tempfile.TemporaryDirectory() as temporary:
            runner = FakeRunner(Path(temporary) / "sandbox", responses[:-1])
            heldout_runner = FakeRunner(Path(temporary) / "heldout", responses[-1:])
            evaluator = SourceEvaluator()
            expert = kernel(
                name="expert",
                source=CUDA_SOURCE + " // optimized",
                validation=accepted(1.0, 2.0),
            )
            heldout = kernel(
                name="heldout",
                case="heldout-gemm",
                source=CUDA_SOURCE + " // heldout baseline",
                validation=accepted(2.0, 2.0),
            )
            heldout = materializable(heldout, "# heldout baseline\n")
            heldout = replace(heldout, problem=ProblemSpec("heldout", "Held-out GEMM."))
            sandbox = FakeSandbox(runner, evaluator)
            heldout_sandbox = FakeSandbox(heldout_runner)

            with (
                patch(
                    "klineage.action.decompose._next",
                    return_value=nullcontext(heldout_sandbox),
                ) as resume,
                patch("klineage.action.decompose._observe", side_effect=observe),
            ):
                lineage = _decompose(
                    sandbox,  # type: ignore[arg-type]
                    expert,
                    max_steps=2,
                    max_rejections=4,
                    roundtrip_cases=(heldout,),
                    effect_verifier=effect_observed,
                    skill_admission=SkillAdmission.ON,
                )

            self.assertEqual(resume.call_args.args[1], heldout)
            self.assertEqual(len(evaluator.calls), 3)
            self.assertEqual(len(heldout_sandbox._evaluator.calls), 2)
            heldout_payload = json.loads(
                heldout_runner.prompts[0].split("INPUT_JSON:\n", 1)[1]
            )
            self.assertEqual(
                heldout_payload["current_kernel"]["problem"], heldout.problem.to_dict()
            )
            payloads = [
                json.loads(prompt.split("INPUT_JSON:\n", 1)[1])
                for prompt in runner.prompts
            ]
            self.assertEqual(payloads[0]["current"]["source"], expert.source)
            self.assertNotIn("validation", payloads[0]["current"])
            self.assertEqual(payloads[1]["target_latency_ms"], 1.0)
            self.assertNotIn("target_validation", payloads[1])
            lifted = payloads[3]["transitions"][0]
            self.assertEqual(lifted["before"], lineage.naive_kernel.source)
            self.assertEqual(lifted["after"], expert.source)
            self.assertEqual(lifted["forward_edit"], "restore aligned vector load")
            self.assertNotIn("evidence", lifted)

            self.assertEqual(len(lineage.states), 2)
            self.assertIn("naive", lineage.naive_kernel.source)
            self.assertEqual(lineage.states[-1].fingerprint, expert.fingerprint)
            self.assertEqual(len(lineage.transitions), 1)
            evidence = lineage.transitions[0]
            self.assertEqual(evidence.before_fingerprint, lineage.states[0].fingerprint)
            self.assertEqual(evidence.after_fingerprint, lineage.states[1].fingerprint)
            self.assertEqual(len(lineage.skills), 1)
            self.assertTrue(lineage.skills[0].admitted)

            restored = type(lineage).from_dict(lineage.to_dict())
            self.assertEqual(restored, lineage)
            lineage_path = save_lineage(lineage, Path(temporary) / "lineage.json")
            memory_path = save_memory(lineage.skills, Path(temporary) / "memory.json")
            self.assertEqual(load_lineage(lineage_path), lineage)
            self.assertEqual(load_memory(memory_path), lineage.skills)


if __name__ == "__main__":
    unittest.main()

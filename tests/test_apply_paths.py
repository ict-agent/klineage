import tempfile
import unittest
import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from test_action import (
    CUDA_SOURCE, FakeRunner, FakeSandbox, SourceEvaluator, accepted,
    bundle, kernel, materializable, skill,
)

from klineage.action.apply import _apply
from klineage.action.code_gen import _materialize
from klineage.action.decompose import _decompose
from klineage.errors import ValidationGateError
from klineage.memory import SkillAdmission


class ApplyPathTests(unittest.TestCase):
    def test_seed_requires_entry(self):
        from test_retrieval import TILE, _lineage
        from klineage.action.apply import _run_paths

        lineage = _lineage()
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = FakeSandbox(FakeRunner(Path(temporary), []))
            with (
                patch("klineage.action.apply._observe", side_effect=lambda _, k, fs: k),
                self.assertRaises(ValidationGateError),
            ):
                _run_paths(sandbox, lineage.states[0], (lineage,), SkillAdmission.OFF,
                           required=(TILE,))

    def test_step_tracks_extra(self):
        from test_retrieval import TILE, MMA, PIPELINE, _lineage
        from klineage.action.apply import _step

        lineage = _lineage()
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = FakeSandbox(FakeRunner(Path(temporary), [{
                "done": True, "_write_bundle": bundle(),
            }]))
            with patch("klineage.action.apply._observe", side_effect=lambda _, k, fs:
                       replace(k, features=tuple(item for item in fs if item != PIPELINE))):
                result = _step(sandbox, lineage.states[0], lineage.skills[:1], "step",
                               vocabulary=(TILE, MMA, PIPELINE))

        self.assertEqual(set(result.features), {TILE, MMA})

    def test_missing_effect_is_failed(self):
        from test_retrieval import _lineage
        from klineage.action.apply import _step

        lineage = _lineage()
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = FakeSandbox(FakeRunner(Path(temporary), [{
                "done": True, "_write_bundle": bundle(),
            }]))
            with (
                patch("klineage.action.apply._observe", side_effect=lambda _, k, fs: k),
                self.assertRaises(ValidationGateError) as caught,
            ):
                _step(sandbox, lineage.states[0], lineage.skills[:1], "step")
        self.assertFalse(caught.exception.kernel.validation.accepted)

    def test_new_case_retrieves_first(self):
        from test_retrieval import _lineage
        from klineage.action import apply

        lineage = _lineage()
        with tempfile.TemporaryDirectory() as temporary:
            runner = FakeRunner(Path(temporary), [{
                "done": True, "_write_source": CUDA_SOURCE,
            }])
            sandbox = FakeSandbox(runner)
            sandbox._problem = Path(temporary) / "new_shape.py"
            sandbox._problem.write_text("# authoritative new problem\n")
            sandbox._inspect = lambda: {
                "problem_name": "new_shape", "operator": "gemm",
                "platform": "sm120", "abi": lineage.states[0].abi.to_dict(),
            }
            sandbox._save = Mock()
            with (
                patch("klineage.action.apply._new", return_value=nullcontext(sandbox)),
                patch("klineage.action.apply._run_paths", side_effect=lambda _, k, *args, **kw: k),
            ):
                result = apply(sandbox._problem, (lineage,))

        payload = json.loads(runner.prompts[0].split("INPUT_JSON:\n", 1)[1])
        self.assertEqual(result.context.case, "gemm")
        self.assertEqual(result.problem.name, "new_shape")
        self.assertEqual(payload["planned_intents"], [card.intent for card in lineage.skills])
        self.assertEqual(payload["kernel_abi"], result.abi.to_dict())
        self.assertEqual(result.features, ())

    def test_path_keeps_best_kernel(self):
        from test_retrieval import _lineage
        from klineage.action.apply import _run_paths

        lineage = _lineage()
        baseline = replace(lineage.states[0], source=CUDA_SOURCE + " // baseline")

        class Timed(SourceEvaluator):
            def evaluate(self, candidate, *, reference=None):
                def latency(value):
                    if value is None:
                        return 10.0
                    source = "\n".join((value.source_files or {"source": value.source}).values())
                    return next((latency for name, latency in (
                        ("pipeline-step", 4.0), ("mma-step", 5.0), ("tile-step", 12.0),
                    ) if name in source), 10.0)
                return accepted(latency(candidate), latency(reference))

        responses = [{"done": True, "_write_bundle": bundle(f"# {name}-step\n")}
                     for name in ("tile", "mma", "pipeline")]
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = FakeSandbox(FakeRunner(Path(temporary), responses), Timed())
            def observed(_, value, features):
                count = next((count for name, count in (
                    ("pipeline-step", 3), ("mma-step", 2), ("tile-step", 1),
                ) if name in value.source), 0)
                return replace(value, features=tuple(
                    card.provides[0] for card in lineage.skills[:count]
                ))

            with patch("klineage.action.apply._observe", side_effect=observed):
                result = _run_paths(sandbox, baseline, (lineage,), SkillAdmission.OFF)

        self.assertIn("pipeline-step", result.source)
        self.assertEqual(result.validation.latency_ms, 4.0)
        self.assertEqual(len(result.features), 3)

    def test_failed_path_keeps_start(self):
        from test_retrieval import _lineage
        from klineage.action.apply import _run_paths

        lineage = _lineage()
        baseline = replace(lineage.states[0], source=CUDA_SOURCE)
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = FakeSandbox(FakeRunner(Path(temporary), [{
                "done": True, "_write_bundle": bundle("# no effect\n"),
            }]))
            with patch("klineage.action.apply._observe", side_effect=lambda _, k, fs:
                       replace(k, features=())):
                result = _run_paths(sandbox, baseline, (lineage,), SkillAdmission.OFF)

        self.assertEqual(result.fingerprint, baseline.fingerprint)
        self.assertEqual(result.context.prior_actions, ())
        self.assertIn("error", result.validation.details["apply_search"]["attempts"][0])

    def test_codegen_has_no_claims(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner = FakeRunner(Path(temporary), [{
                "done": True, "_write_bundle": bundle(),
            }])
            current = materializable(kernel())
            result = _materialize(FakeSandbox(runner), current, (skill("tile", None),))

        self.assertEqual(result.context.prior_actions, ())

    def test_apply_rejects_slow_code(self):
        class Slow(SourceEvaluator):
            def evaluate(self, candidate, *, reference=None):
                return accepted(2.0, 1.0)

        with tempfile.TemporaryDirectory() as temporary:
            runner = FakeRunner(Path(temporary), [{
                "done": True, "_write_bundle": bundle(),
            }])
            with self.assertRaises(ValidationGateError) as caught:
                _apply(
                    FakeSandbox(runner, Slow()), materializable(kernel()),
                    (skill("tile", None),), SkillAdmission.OFF,
                )
            self.assertFalse(caught.exception.kernel.validation.accepted)

    def test_roundtrip_restores_speed(self):
        from klineage.kernel import Feature

        tile = Feature("main", "tile")

        class SlowRoundtrip(SourceEvaluator):
            def evaluate(self, candidate, *, reference=None):
                self.calls.append((candidate, reference))
                if candidate.name.endswith(".roundtrip"):
                    return accepted(2.0, 1.0)
                return accepted()

        responses = [
            {
                "done": False, "action_category": "tile", "locus": "main",
                "backward_edit": "remove tiling", "_write_source": CUDA_SOURCE,
                "before_features": [], "after_features": [tile.to_dict()],
            },
            {"forward_edit": "restore tiling", "_write_source": CUDA_SOURCE},
            {
                "intent": "tile", "anchor": "main", "carrier": "tile the loop",
                "preconditions": ["loop"], "effects": ["reuse"], "risks": [],
                "scope": {"cases": ["*"], "languages": ["*"],
                          "platforms": ["*"], "prior_actions": []},
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = FakeSandbox(FakeRunner(Path(temporary), responses), SlowRoundtrip())
            with patch("klineage.action.decompose._observe", side_effect=lambda _, k, fs:
                       replace(k, features=() if k.name.endswith(".deopt1") else tuple(fs))):
                lineage = _decompose(
                    sandbox, kernel(), max_steps=1, max_rejections=1,
                    roundtrip_cases=(), effect_verifier=None,
                    skill_admission=SkillAdmission.OFF,
                )

        self.assertEqual(lineage.transitions, ())
        self.assertEqual(lineage.termination, "rejection_limit")
        self.assertEqual(len(sandbox._evaluator.calls), 3)

    def test_done_needs_a_reason(self):
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = FakeSandbox(FakeRunner(Path(temporary), [{"done": True}]))
            with self.assertRaisesRegex(ValueError, "reason"):
                _decompose(
                    sandbox, kernel(), max_steps=1, max_rejections=1,
                    roundtrip_cases=(), effect_verifier=None,
                    skill_admission=SkillAdmission.OFF,
                )

    def test_done_keeps_residual(self):
        from klineage.kernel import Feature

        expert = replace(kernel(), features=(Feature("main", "mma"),))
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = FakeSandbox(FakeRunner(Path(temporary), [{
                "done": True, "reason": "no further simplification",
            }]))
            lineage = _decompose(
                sandbox, expert, max_steps=1, max_rejections=1,
                roundtrip_cases=(), effect_verifier=None,
                skill_admission=SkillAdmission.OFF,
            )
        self.assertNotEqual(lineage.termination, "complete")
        self.assertIn("main/mma", lineage.reason)

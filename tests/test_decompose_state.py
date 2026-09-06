import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from test_action import (
    CUDA_SOURCE, FakeRunner, FakeSandbox, kernel, materializable, skill,
)

from klineage.action.decompose import _admit_skills, _decompose
from klineage.errors import StructuredOutputError
from klineage.kernel import Feature
from klineage.memory import SkillAdmission, retrieve_paths

_TILE = Feature("gemm.main", "shared_memory_tiling")
_MMA = Feature("gemm.main", "mma")
_PIPELINE = Feature("gemm.main", "pipeline")
_EXPERT = CUDA_SOURCE + " // expert"
_BEFORE = CUDA_SOURCE + " // naive"
_ROUNDTRIP = CUDA_SOURCE + " // roundtrip"


def _proposal():
    return {
        "done": False, "action_category": "mma", "locus": "gemm.main",
        "backward_edit": "Replace MMA with scalar accumulation.",
        "before_features": [_TILE.to_dict()],
        "after_features": [_TILE.to_dict(), _MMA.to_dict()],
        "_write_source": _BEFORE,
    }


def _lift():
    return {
        "intent": "Use tensor-core MMA.", "anchor": "gemm.main",
        "carrier": "wmma::mma_sync(acc, a, b, acc)",
        "preconditions": ["compatible fragments"], "effects": ["MMA computation"],
        "risks": [], "requires": [_TILE.to_dict()],
        "provides": [_MMA.to_dict()], "conflicts": [_MMA.to_dict()],
        "scope": {"cases": ["*"], "languages": ["*"], "platforms": ["*"],
                  "prior_actions": []},
    }


class _AuditSandbox(FakeSandbox):
    def __init__(self, runner, observed):
        super().__init__(runner)
        self._observed = observed
        self._audits = []

    def _ask(self, purpose, instructions, payload):
        if purpose != "observe-features":
            return super()._ask(purpose, instructions, payload)

        sources = payload["source_files"]
        name = "kernel.cu"
        source = sources[name]
        self._audits.append(source)
        present = self._observed[source]
        return {"checks": [{
            "feature": item,
            "status": "present" if Feature.from_dict(item) in present else "absent",
            "reason": "The fixture's recorded mechanism state determines this result.",
            "evidence": [{"path": name, "start": 1, "end": 1}],
        } for item in payload["features"]]}


class DecomposeStateTests(unittest.TestCase):
    def _run(self, *, proposal=None, lifted=None, observed=None, expert_features=()):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        responses = [
            proposal if proposal is not None else _proposal(),
            {"forward_edit": "Restore MMA.", "_write_source": _ROUNDTRIP},
            lifted if lifted is not None else _lift(),
        ]
        observed = observed or {
            _EXPERT: (_TILE, _MMA), _BEFORE: (_TILE,),
            _ROUNDTRIP: (_TILE, _MMA),
        }
        sandbox = _AuditSandbox(FakeRunner(Path(temporary.name), responses), observed)
        lineage = _decompose(
            sandbox, replace(kernel(source=_EXPERT), features=expert_features),
            max_steps=1, max_rejections=1,
            roundtrip_cases=(), effect_verifier=None,
            skill_admission=SkillAdmission.OFF,
        )
        return lineage, sandbox

    def test_states_enable_retrieval(self):
        lineage, sandbox = self._run()

        self.assertEqual(set(lineage.states[0].features), {_TILE})
        self.assertEqual(set(lineage.states[1].features), {_TILE, _MMA})
        self.assertEqual(set(sandbox._audits), {_EXPERT, _BEFORE, _ROUNDTRIP})
        card = lineage.skills[0]
        self.assertEqual(card.requires, (_TILE,))
        self.assertEqual(card.provides, (_MMA,))
        self.assertEqual(card.scope.platforms, ("sm120",))
        self.assertEqual(card.scope.cases, ("gemm",))
        self.assertEqual(len(retrieve_paths((lineage,), lineage.states[0])), 1)
        self.assertEqual(lineage.termination, "step_limit")
        self.assertIn("feature_verifier", lineage.transitions[0].roundtrip_validation.details)

    def test_claims_need_removed_code(self):
        for source, present in ((_BEFORE, (_TILE, _MMA)), (_EXPERT, (_TILE,))):
            with self.subTest(source=source):
                observed = {_EXPERT: (_TILE, _MMA), _BEFORE: (_TILE,),
                            _ROUNDTRIP: (_TILE, _MMA), source: present}
                lineage, _ = self._run(observed=observed)

                self.assertEqual(lineage.transitions, ())
                self.assertEqual(lineage.termination, "rejection_limit")

    def test_roundtrip_needs_effect(self):
        lineage, _ = self._run(observed={
            _EXPERT: (_TILE, _MMA), _BEFORE: (_TILE,), _ROUNDTRIP: (_TILE,),
        })

        self.assertEqual(lineage.transitions, ())
        self.assertEqual(lineage.termination, "rejection_limit")

    def test_preserves_known_features(self):
        lineage, _ = self._run(
            expert_features=(_PIPELINE,),
            observed={_EXPERT: (_TILE, _MMA, _PIPELINE), _BEFORE: (_TILE,),
                      _ROUNDTRIP: (_TILE, _MMA)},
        )

        self.assertEqual(lineage.transitions, ())
        self.assertEqual(lineage.termination, "rejection_limit")

    def test_rejects_parameter_only(self):
        proposal = _proposal()
        proposal["before_features"] = proposal["after_features"]
        lineage, _ = self._run(proposal=proposal)

        self.assertEqual(lineage.transitions, ())

    def test_lift_requires_evidence(self):
        for field, features in (
            ("requires", [_PIPELINE.to_dict()]),
            ("provides", [_PIPELINE.to_dict()]),
            ("provides", []),
            ("conflicts", [_TILE.to_dict()]),
        ):
            with self.subTest(field=field, features=features):
                lifted = {**_lift(), field: features}
                with self.assertRaises(StructuredOutputError):
                    self._run(lifted=lifted)

    def test_admission_advances_state(self):
        cards = (
            replace(skill("tile", None), provides=(_TILE,)),
            replace(skill("mma", None), requires=(_TILE,), provides=(_MMA,)),
        )
        baseline = materializable(kernel(case="heldout"))
        seen = []

        def materialize(_, current, selected, **kwargs):
            seen.append(current)
            return replace(current, source=current.source + "# generated\n",
                           source_files=None)

        def observe(_, current, features):
            return replace(current, features=tuple(features))

        with tempfile.TemporaryDirectory() as temporary:
            sandbox = FakeSandbox(FakeRunner(Path(temporary), []))
            with (
                patch("klineage.action.decompose._next", return_value=nullcontext(sandbox)),
                patch("klineage.action.decompose._materialize", side_effect=materialize),
                patch("klineage.action.decompose._observe", side_effect=observe),
            ):
                admitted = _admit_skills(
                    cards, states=(kernel(),), cases=(baseline,),
                    effect_verifier=lambda *_: True,
                )

        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[1].features, (_TILE,))
        self.assertTrue(all(card.admitted for card in admitted))

    def test_admission_skips_unmet(self):
        card = replace(skill("mma", None), requires=(_TILE,), provides=(_MMA,))
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = FakeSandbox(FakeRunner(Path(temporary), []))
            with (
                patch("klineage.action.decompose._next", return_value=nullcontext(sandbox)),
                patch("klineage.action.decompose._materialize") as generate,
            ):
                result = _admit_skills(
                    (card,), states=(kernel(),), cases=(materializable(kernel()),),
                    effect_verifier=lambda *_: True,
                )

        generate.assert_not_called()
        self.assertFalse(result[0].admitted)

    def test_admission_keeps_failures(self):
        cards = (
            replace(skill("tile", None), provides=(_TILE,)),
            replace(skill("mma", None), requires=(_TILE,), provides=(_MMA,)),
        )
        with tempfile.TemporaryDirectory() as temporary:
            sandbox = FakeSandbox(FakeRunner(Path(temporary), []))
            with (
                patch("klineage.action.decompose._next", return_value=nullcontext(sandbox)),
                patch("klineage.action.decompose._materialize",
                      return_value=materializable(kernel())) as generate,
                patch("klineage.action.decompose._observe",
                      side_effect=StructuredOutputError("unknown mechanism")),
            ):
                effect = Mock(return_value=True)
                result = _admit_skills(
                    cards, states=(kernel(),), cases=(materializable(kernel()),),
                    effect_verifier=effect,
                )

        self.assertEqual(generate.call_count, 1)
        effect.assert_not_called()
        self.assertFalse(result[0].verification_log[0].validation.accepted)
        self.assertEqual(result[1].verification_log, ())

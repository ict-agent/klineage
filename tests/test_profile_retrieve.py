import importlib
import json
import unittest
from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import Mock, patch

from test_retrieval import TILE, _abi, _lineage

from klineage.errors import ActionError
from klineage.kernel import Kernel, TargetContext
from klineage.memory import SkillAdmission, retrieve_paths

retrieval = importlib.import_module("klineage.action.retrieve")


def _response(*ids):
    return {"plan_ids": list(ids), "reasons": {key: "Addresses measured memory stalls." for key in ids}}


class ProfileRetrieveTests(unittest.TestCase):
    def setUp(self):
        self.lineage = _lineage()
        self.current = self.lineage.states[0]
        self.sandbox = Mock()
        self.sandbox._ask.return_value = _response("plan-00")
        self.profiled = Mock(spec=Kernel)
        self.profiled._prompt_input.return_value = {
            **self.current._prompt_input(), "profile": {"bottleneck": "memory"},
        }
        self.profile = self.enterContext(patch.object(
            retrieval, "_profile", return_value=self.profiled,
        ))

    def test_agent_reorders_only(self):
        lineages = (self.lineage, _lineage(4096))
        plans = retrieve_paths(lineages, self.current)
        self.sandbox._ask.return_value = _response("plan-01", "plan-00")
        with patch.object(retrieval, "retrieve_paths", return_value=plans):
            result = retrieval._retrieve(self.sandbox, self.current, lineages, SkillAdmission.OFF)

        self.assertIs(result[0], plans[1])
        self.assertIs(result[1], plans[0])
        self.profile.assert_called_once_with(self.sandbox, self.current)
        purpose, instructions, payload = self.sandbox._ask.call_args.args
        self.assertEqual(purpose, "retrieve")
        self.assertIn("profile", instructions)
        self.assertEqual(payload["current"], self.profiled._prompt_input.return_value)

    def test_memory_and_entry(self):
        lineage = replace(self.lineage, skills=tuple(reversed(self.lineage.skills)))
        current = self.lineage.states[2]
        result = retrieval._retrieve(self.sandbox, current, (lineage,), SkillAdmission.OFF)
        payload = self.sandbox._ask.call_args.args[2]["plans"][0]

        self.assertEqual(payload["skills"], [card.to_dict() for card in lineage.skills])
        self.assertEqual(payload["entry_index"], 2)
        self.assertEqual(payload["entry"]["features"], [item.to_dict() for item in current.features])
        self.assertEqual(payload["terminal"]["features"],
                         [item.to_dict() for item in lineage.states[-1].features])
        self.assertEqual(payload["entry"]["latency_ms"], current.validation.latency_ms)
        self.assertEqual([item["skill_id"] for item in payload["path"]],
                         [card.skill_id for card in result[0].skills])
        for state in lineage.states:
            self.assertNotIn(state.source, json.dumps(payload))

    def test_partial_path_end(self):
        blocked = replace(self.lineage.skills[1], conflicts=(TILE,))
        lineage = replace(self.lineage, skills=(self.lineage.skills[0], blocked, self.lineage.skills[2]))
        retrieval._retrieve(self.sandbox, self.current, (lineage,), SkillAdmission.OFF)
        payload = self.sandbox._ask.call_args.args[2]["plans"][0]

        self.assertEqual(len(payload["path"]), 1)
        self.assertEqual(len(payload["skills"]), 3)
        self.assertEqual(payload["path_end"]["fingerprint"], lineage.states[1].fingerprint)
        self.assertNotEqual(payload["path_end"], payload["terminal"])

    def test_binds_lineage_evidence(self):
        cards = tuple(replace(card, evidence=(
            replace(card.evidence[0], before_fingerprint="other"), *card.evidence,
        )) for card in self.lineage.skills)
        lineage = replace(self.lineage, skills=cards)
        result = retrieval._retrieve(self.sandbox, self.current, (lineage,), SkillAdmission.OFF)
        payload = self.sandbox._ask.call_args.args[2]["plans"][0]

        self.assertEqual(payload["skills"], [card.to_dict() for card in cards])
        for step, edge, card in zip(payload["path"], lineage.transitions, result[0].skills):
            self.assertEqual(card.evidence, (edge,))
            self.assertEqual((step["before"], step["after"]),
                             (edge.before_fingerprint, edge.after_fingerprint))

    def test_repeated_fingerprint(self):
        states = (*self.lineage.states[:-1], replace(self.lineage.states[-1], source=self.current.source))
        edges = tuple(replace(edge, before_fingerprint=states[index].fingerprint,
                              after_fingerprint=states[index + 1].fingerprint)
                      for index, edge in enumerate(self.lineage.transitions))
        cards = tuple(replace(card, evidence=(edge,)) for card, edge in zip(self.lineage.skills, edges))
        lineage = replace(self.lineage, states=states, transitions=edges, skills=cards)
        retrieval._retrieve(self.sandbox, self.current, (lineage,), SkillAdmission.OFF)
        payload = self.sandbox._ask.call_args.args[2]["plans"][0]

        self.assertEqual(payload["path_end"], payload["terminal"])

    def test_hard_filters_run_first(self):
        for current, admission, lineages in (
            (self.current, SkillAdmission.OFF, ()),
            (self.current, SkillAdmission.ON, (self.lineage,)),
            (replace(self.current, abi=_abi(dtype="float32")), SkillAdmission.OFF, (self.lineage,)),
            (replace(self.current, context=TargetContext("gemm", "cuda", "sm90")),
             SkillAdmission.OFF, (self.lineage,)),
        ):
            with self.subTest(current=current, admission=admission):
                self.assertEqual(retrieval._retrieve(self.sandbox, current, lineages, admission), ())
        self.profile.assert_not_called()
        self.sandbox._ask.assert_not_called()

    def test_rejects_invalid_rankings(self):
        valid = _response("plan-00")
        for response in (
            None, [], {}, {**valid, "path": []},
            {**valid, "plan_ids": []}, {**valid, "plan_ids": ["plan-00", "plan-00"]},
            {**valid, "plan_ids": ["other"]}, {**valid, "plan_ids": "plan-00"},
            {**valid, "plan_ids": [{}]}, {**valid, "reasons": []},
            {**valid, "reasons": {}}, {**valid, "reasons": {"other": "wrong plan"}},
            {**valid, "reasons": {"plan-00": " "}}, {**valid, "reasons": {"plan-00": 1}},
        ):
            with self.subTest(response=response), self.assertRaises(ActionError):
                self.sandbox._ask.return_value = response
                retrieval._retrieve(self.sandbox, self.current, (self.lineage,), SkillAdmission.OFF)

    def test_profile_failure(self):
        self.profile.side_effect = ActionError("profile failed")
        with self.assertRaisesRegex(ActionError, "profile failed"):
            retrieval._retrieve(self.sandbox, self.current, (self.lineage,), SkillAdmission.OFF)
        self.sandbox._ask.assert_not_called()

    def test_duplicate_omits_plan(self):
        self.sandbox._ask.return_value = {
            "plan_ids": ["plan-00", "plan-00"],
            "reasons": {"plan-00": "first", "plan-01": "second"},
        }
        with self.assertRaises(ActionError):
            retrieval._retrieve(self.sandbox, self.current,
                                (self.lineage, _lineage(4096)), SkillAdmission.OFF)

    def test_public_action_boundary(self):
        with patch.object(retrieval, "_next", return_value=nullcontext(self.sandbox)) as boundary:
            result = retrieval.retrieve(self.current, (self.lineage,))
        boundary.assert_called_once_with(retrieval._Kind.RETRIEVE, self.current)
        self.assertEqual(result[0].lineage, self.lineage)


if __name__ == "__main__":
    unittest.main()

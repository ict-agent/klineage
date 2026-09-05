import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from test_action import (
    FakeRunner,
    FakeSandbox,
    bundle,
    kernel,
    materializable,
    skill,
)

from klineage.action import apply
from klineage.kernel import TargetContext
from klineage.memory import Scope, SkillAdmission, retrieve


class MemoryTests(unittest.TestCase):
    def test_skillcard_module_compiles(self) -> None:
        source = Path(__file__).parents[1] / "src/klineage/memory/skillcard.py"
        compile(source.read_text(), str(source), "exec")

    def test_scope_and_prerequisites(self) -> None:
        target = TargetContext("gemm", "cuda", "sm120", ("tile",))
        eligible = skill("pipeline", None, declared=Scope(prior_actions=("tile",)))
        cards = (
            skill("wrong-case", None, declared=Scope(cases=("topk",))),
            skill("wrong-language", None, declared=Scope(languages=("triton",))),
            skill("wrong-platform", None, declared=Scope(platforms=("sm90",))),
            skill("tile", None),
            skill("needs-mma", None, declared=Scope(prior_actions=("mma",))),
            eligible,
            eligible,
        )
        self.assertEqual(retrieve(cards, target), (cards[3], eligible))

    def test_category_can_target_new_locus(self) -> None:
        card = skill("tile-second-loop", None)
        card = replace(
            card,
            evidence=(replace(
                card.evidence[0], action_category="tile", locus="second loop",
            ),),
        )
        target = TargetContext("gemm", "cuda", "sm120", ("tile",))

        self.assertEqual(retrieve((card,), target), (card,))

    def test_retrieve_apply_progress(self) -> None:
        tile = skill("tile", None)
        pipeline = skill("pipeline", None, declared=Scope(prior_actions=("tile",)))
        current = materializable(kernel())
        pending = (pipeline, tile)

        with tempfile.TemporaryDirectory() as temporary:
            sandboxes = []
            for name in ("tile", "pipeline"):
                directory = Path(temporary) / name
                directory.mkdir()
                runner = FakeRunner(directory, [{
                    "done": True, "_write_bundle": bundle(f"# {name}\n"),
                }])
                sandboxes.append(FakeSandbox(runner))

            with patch(
                "klineage.action.apply._next",
                side_effect=[nullcontext(sandbox) for sandbox in sandboxes],
            ):
                while cards := retrieve(pending, current.context):
                    current = apply(current, cards)
                    applied = {card.skill_id for card in cards}
                    pending = tuple(c for c in pending if c.skill_id not in applied)

        self.assertEqual(pending, ())
        self.assertEqual(current.context.prior_actions, ("tile", "pipeline"))
        self.assertTrue(current.validation.accepted)
        self.assertEqual([len(s._evaluator.calls) for s in sandboxes], [1, 1])

    def test_order_and_admission(self) -> None:
        target = TargetContext("gemm", "cuda", "sm120")
        hypothesis = skill("tile", None)
        admitted = skill("mma", target)
        cards = (hypothesis, admitted)
        self.assertEqual(retrieve(cards, target), cards)
        self.assertEqual(
            retrieve(cards, target, skill_admission=SkillAdmission.ON),
            (admitted,),
        )

    def test_conflicting_ids_fail(self) -> None:
        card = skill("tile", None)
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            retrieve(
                (card, replace(card, carrier="different edit")),
                TargetContext("gemm", "cuda", "sm120"),
            )


if __name__ == "__main__":
    unittest.main()

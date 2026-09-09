import unittest
from dataclasses import replace

from kernel_fixtures import kernel, skill
from problem_fixtures import problem_spec

from klineage.memory import Scope, retrieve


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.target = kernel()
        self.card = skill(
            "staging",
            declared=Scope(cases=("gemm",), languages=("cuda",), platforms=("sm120",)),
        )

    def test_applicable_card_is_returned(self):
        self.assertEqual(retrieve((self.card,), self.target), (self.card,))
        self.assertEqual(retrieve((self.card,), self.target.problem), (self.card,))

    def test_exclusion_and_duplicates(self):
        other = skill("mma")
        wildcard = skill("wildcard", declared=Scope())
        self.assertEqual(
            retrieve(
                (self.card, other, wildcard, self.card),
                self.target,
                exclude_skills=("mma",),
            ),
            (self.card, wildcard),
        )

    def test_conflicting_ids_fail(self):
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            retrieve(
                (self.card, replace(self.card, intent="different edit")), self.target
            )

    def test_scope_matches_target(self):
        for problem in (
            problem_spec("topk"),
            problem_spec(language="python"),
            problem_spec(platform="sm90"),
        ):
            with self.subTest(problem=problem):
                self.assertEqual(retrieve((self.card,), problem), ())

    def test_rejects_invalid_inputs(self):
        for cards, target in (
            ("staging", self.target),
            (("staging",), self.target),
            ((self.card,), "gemm"),
        ):
            with self.subTest(cards=cards, target=target), self.assertRaises(TypeError):
                retrieve(cards, target)


if __name__ == "__main__":
    unittest.main()

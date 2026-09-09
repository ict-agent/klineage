import unittest
from dataclasses import replace

from problem_fixtures import problem_spec

from klineage.contract import ABIValue, ValueRole


class ProblemSignatureTests(unittest.TestCase):
    def test_resolves_axes_in_order(self):
        problem = problem_spec()
        definition = {
            **problem.definition,
            "axes": {"N": {"type": "var"}, "K": {"type": "const", "value": 16}},
            "inputs": {
                "z": {
                    "shape": ["N", "K"],
                    "dtype": "float32",
                    "description": "Matrix.",
                },
                "a": {"shape": None, "dtype": "int32", "description": None},
            },
            "outputs": {"result": {"shape": ["N"], "dtype": "float32"}},
        }
        problem = replace(
            problem,
            definition=definition,
            workload={**problem.workload, "axes": {"N": 7}},
        )
        self.assertEqual(
            problem.values(ValueRole.INPUTS),
            (
                ABIValue("z", "float32", (7, 16), "Matrix."),
                ABIValue("a", "int32"),
            ),
        )
        self.assertEqual(
            problem.values(ValueRole.OUTPUTS), (ABIValue("result", "float32", (7,)),)
        )
        smaller = replace(problem, workload={**problem.workload, "axes": {"N": 0}})
        self.assertEqual(smaller.values(ValueRole.INPUTS)[0].shape, (0, 16))

    def test_rejects_invalid_axes(self):
        problem = problem_spec()
        for axes in (
            {},
            {"M": -1, "N": 8, "K": 8},
            {"M": True, "N": 8, "K": 8},
            {"M": "8", "N": 8, "K": 8},
        ):
            with self.subTest(axes=axes), self.assertRaises(ValueError):
                replace(problem, workload={**problem.workload, "axes": axes}).values(
                    ValueRole.INPUTS
                )

        for axes in (
            {},
            {"M": {"type": "unknown"}},
            {"M": {"type": "const", "value": 8}},
        ):
            with self.subTest(axes=axes), self.assertRaises(ValueError):
                replace(
                    problem, definition={**problem.definition, "axes": axes}
                ).values(ValueRole.INPUTS)

    def test_keeps_empty_and_rank_zero(self):
        problem = problem_spec()
        definition = {
            **problem.definition,
            "inputs": {},
            "outputs": {"result": {"shape": [], "dtype": "float32"}},
        }
        problem = replace(problem, definition=definition)
        self.assertEqual(problem.values(ValueRole.INPUTS), ())
        self.assertEqual(
            problem.values(ValueRole.OUTPUTS), (ABIValue("result", "float32"),)
        )

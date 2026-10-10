import unittest

from problem_fixtures import problem_spec


class ProblemSignatureTests(unittest.TestCase):
    def test_fields_are_detached(self):
        problem = problem_spec()
        values = problem.values("inputs")
        values["a"]["shape"][0] = 1
        values["a"]["dtype"] = "int32"
        self.assertEqual(problem.values("inputs")["a"]["shape"], [128, 128])
        self.assertEqual(problem.definition["inputs"]["a"]["shape"], ["M", "K"])
        self.assertEqual(problem.definition["inputs"]["a"]["dtype"], "float16")

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
        problem = problem.replace(definition=definition, workloads=({**problem.workloads[0], 'axes': {'N': 7}},))
        self.assertEqual(
            problem.values("inputs"),
            {
                "z": {"dtype": "float32", "shape": [7, 16], "description": "Matrix."},
                "a": {"dtype": "int32", "shape": None, "description": None},
            },
        )
        self.assertEqual(tuple(problem.values("inputs")), ("z", "a"))
        self.assertEqual(
            problem.values("outputs"), {"result": {"dtype": "float32", "shape": [7]}}
        )
        smaller = problem.replace(workloads=({**problem.workloads[0], 'axes': {'N': 0}},))
        self.assertEqual(smaller.values("inputs")["z"]["shape"], [0, 16])

    def test_rejects_invalid_axes(self):
        problem = problem_spec()
        for axes in (
            {},
            {"M": -1, "N": 8, "K": 8},
            {"M": True, "N": 8, "K": 8},
            {"M": "8", "N": 8, "K": 8},
        ):
            with self.subTest(axes=axes), self.assertRaises(ValueError):
                problem.replace(workloads=({**problem.workloads[0], 'axes': axes},)).values(
                    "inputs"
                )

        for axes in (
            {},
            {"M": {"type": "unknown"}},
            {"M": {"type": "const", "value": 8}},
        ):
            with self.subTest(axes=axes), self.assertRaises(ValueError):
                problem.replace(definition={**problem.definition, 'axes': axes}).values("inputs")

    def test_keeps_empty_and_rank_zero(self):
        problem = problem_spec()
        definition = {
            **problem.definition,
            "inputs": {},
            "outputs": {"result": {"shape": [], "dtype": "float32"}},
        }
        problem = problem.replace(definition=definition)
        self.assertEqual(problem.values("inputs"), {})
        self.assertEqual(
            problem.values("outputs"), {"result": {"dtype": "float32", "shape": []}}
        )

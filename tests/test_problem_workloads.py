from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from problem_fixtures import problem_spec

from klineage.artifact import Node, Problem
from klineage.artifact.problem import load_problem, load_trace
from klineage.backend import get_backend
from klineage.serialization import from_json, to_json

DEFINITION = {
    "name": "increment",
    "op_type": "elementwise",
    "axes": {"N": {"type": "var"}, "K": {"type": "const", "value": 4}},
    "inputs": {"x": {"shape": ["N", "K"], "dtype": "float32"}},
    "outputs": {"y": {"shape": ["N", "K"], "dtype": "float32"}},
    "reference": "def run(x): return x + 1\n",
}


def workload(identifier, size=2, path=None):
    source = {"type": "random"}
    if path is not None:
        source = {"type": "safetensors", "path": str(path), "tensor_key": "x"}
    return {"uuid": identifier, "axes": {"N": size}, "inputs": {"x": source}}


def problem(*workloads):
    return Problem(
        "increment", DEFINITION, workloads, language="cuda", platform="sm90"
    )


class ProblemWorkloadsTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_custom_problem_layout(self):
        class ExtendedProblem(Problem):
            @classmethod
            def problem_spec(cls):
                return super().problem_spec().with_child(Node.file("notes.txt"))

        item = problem()
        Problem.problem_spec().check(item.tree)
        with self.assertRaisesRegex(ValueError, "notes.txt"):
            ExtendedProblem(
                "increment", DEFINITION, language="cuda", platform="sm90"
            )
        with self.assertRaisesRegex(ValueError, "notes.txt"):
            from_json(to_json(item), ExtendedProblem)

        tree = item.tree.with_child(Node.file("notes.txt", "Custom layout."))
        extended = ExtendedProblem.from_tree(item.kind, tree)
        self.assertEqual(from_json(to_json(extended), ExtendedProblem), extended)

    def test_definition_without_workloads(self):
        item = Problem(
            "increment", DEFINITION, language="cuda", platform="sm90"
        )
        self.assertEqual(item.workloads, ())
        self.assertEqual(item.resolve_inputs(), ())
        self.assertEqual(tuple(node.name for node in item.tree.children), (
            "target.json", "definition.json", "workloads.jsonl",
        ))
        self.assertEqual(item.find("workloads.jsonl").content, "")
        self.assertFalse(hasattr(item, "workload"))
        self.assertEqual(item.values("inputs")["x"]["shape"], ["N", 4])
        self.assertEqual(from_json(to_json(item), Problem), item)

    def test_order_and_roundtrip(self):
        cases = (workload("small", 2), workload("large", 8))
        item = problem(*cases)
        self.assertEqual(item.workloads, cases)
        records = item.find("workloads.jsonl").content.splitlines()
        self.assertEqual(len(records), 2)
        self.assertEqual([json.loads(line)["workload"]["uuid"] for line in records],
                         ["small", "large"])
        restored = from_json(to_json(item), Problem)
        self.assertEqual(restored.workloads, cases)
        self.assertEqual(restored.resolve_inputs(), cases)
        self.assertEqual(restored.resolve_inputs("large"), (cases[1],))
        restored.workloads[0]["axes"]["N"] = 99
        self.assertEqual(restored.workloads, cases)

    def test_symbolic_and_explicit_shapes(self):
        small, large = workload("small", 2), workload("large", 8)
        item = problem(small, large)
        self.assertEqual(item.values("inputs")["x"]["shape"], ["N", 4])
        self.assertEqual(item.values("inputs", small)["x"]["shape"], [2, 4])
        self.assertEqual(item.values("outputs", large)["y"]["shape"], [8, 4])
        self.assertEqual(problem(small).values("inputs")["x"]["shape"], [2, 4])

    def test_shared_input_deduplication(self):
        first, second = self.root / "a.safetensors", self.root / "b.safetensors"
        first.write_bytes(b"same fixed data")
        second.write_bytes(first.read_bytes())
        item = problem(workload("first", path=first), workload("second", path=second))
        self.assertEqual(len(item.find("inputs").children), 1)
        self.assertEqual(len(item.resolve_inputs()), 2)
        restored = from_json(to_json(item), Problem)
        first.unlink()
        second.unlink()
        self.assertEqual([case["uuid"] for case in restored.workloads], ["first", "second"])
        with self.assertRaises(ValueError):
            restored.resolve_inputs()

    def test_selection_checks_only_its_inputs(self):
        first, second = self.root / "a.safetensors", self.root / "b.safetensors"
        first.write_bytes(b"first fixed data")
        second.write_bytes(b"second fixed data")
        item = problem(workload("first", path=first), workload("second", path=second))
        first.write_bytes(b"changed first input")
        self.assertEqual(item.resolve_inputs("second")[0]["uuid"], "second")
        with self.assertRaisesRegex(ValueError, "changed"):
            item.resolve_inputs()
        with self.assertRaisesRegex(ValueError, "changed"):
            item.replace(platform="sm120")
        replaced = item.replace(workloads=(workload("replacement", 3),))
        self.assertEqual(replaced.resolve_inputs()[0]["uuid"], "replacement")

    def test_directory_input_paths(self):
        (self.root / "definition.json").write_text(json.dumps(DEFINITION))
        inputs = self.root / "inputs"
        inputs.mkdir()
        data = inputs / "fixed.safetensors"
        data.write_bytes(b"fixed input")
        record = {"definition": "increment", "workload": workload("fixed", path="inputs/fixed.safetensors")}
        (self.root / "workloads.jsonl").write_text(json.dumps(record) + "\n")
        module = load_trace(self.root)
        self.assertEqual(module.workloads[0]["inputs"]["x"]["path"], str(data))
        with patch.object(type(get_backend("cuda")), "info", return_value={"platform": "sm90"}):
            loaded = load_problem(self.root, backend=get_backend("cuda"))
        self.assertEqual(loaded.resolve_inputs()[0]["inputs"]["x"]["path"], str(data))

    def test_directory_target(self):
        (self.root / "definition.json").write_text(json.dumps(DEFINITION))
        (self.root / "workloads.jsonl").write_text("")
        target = {"name": "custom", "language": "cuda", "platform": "sm90"}
        (self.root / "target.json").write_text(json.dumps(target))
        with patch("klineage.backend.detect_backend", side_effect=AssertionError("unneeded device probe")):
            loaded = load_problem(self.root)
        self.assertEqual(loaded.target(), target)
        self.assertEqual(loaded.workloads, ())

    def test_loading_does_not_run_oracle(self):
        definition = DEFINITION | {"reference": "raise RuntimeError('oracle executed')"}
        (self.root / "definition.json").write_text(json.dumps(definition))
        (self.root / "workloads.jsonl").write_text("")
        (self.root / "target.json").write_text(json.dumps(problem().target()))
        self.assertEqual(load_problem(self.root).definition, definition)


class ProblemInputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "input.safetensors"
        self.path.write_bytes(b"fixed input")
        problem = problem_spec()
        workload = problem.workloads[0]
        workload["inputs"]["a"] = {
            "type": "safetensors", "path": str(self.path), "tensor_key": "a",
        }
        self.problem = problem.replace(workloads=(workload,))

    def test_runtime_checks_input_digest(self):
        self.assertEqual(self.problem.resolve_inputs(), self.problem.workloads)
        self.path.write_bytes(b"changed input")
        self.assertEqual(self.problem.workloads[0]["axes"]["M"], 128)
        with self.assertRaisesRegex(ValueError, "changed"):
            self.problem.resolve_inputs()

    def test_input_path_scope(self):
        record = json.loads(self.problem.find("workloads.jsonl").content)
        descriptor = record["workload"]["inputs"]["a"]
        filename = Path(descriptor["path"]).name
        for prefix in ("other", "../inputs", "inputs/nested", "inputs/..", "/inputs", "inputs/", "inputs/."):
            with self.subTest(prefix=prefix):
                descriptor["path"] = f"{prefix}/{filename}"
                tree = self.problem.tree.with_child(Node.file("workloads.jsonl", json.dumps(record)))
                with self.assertRaisesRegex(ValueError, "fixed input path"):
                    Problem.from_tree("problem", tree)

    def test_replace_keeps_input_capture(self):
        updated = self.problem.replace(platform="sm90")
        self.assertEqual(updated.resolve_inputs(), self.problem.resolve_inputs())
        self.path.unlink()
        with self.assertRaises(ValueError):
            self.problem.replace(platform="sm90")


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

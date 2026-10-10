"""Problems preserve zero or more independent workload cases."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from klineage.artifact import Artifact, Node, Problem, json_node
from klineage.artifact import problem as problem_module
from klineage.artifact.problem import (
    load_problem,
    load_problem_module,
    load_trace,
    trace_module,
)
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
        self.assertEqual(item.node("workloads.jsonl").content, "")
        self.assertFalse(hasattr(item, "workload"))
        self.assertEqual(item.values("inputs")["x"]["shape"], ["N", 4])
        self.assertEqual(from_json(to_json(item), Problem), item)

    def test_workloads_file_required(self):
        item = problem()
        tree = Node.directory(".", (item.node("target.json"), item.node("definition.json")))
        with self.assertRaisesRegex(ValueError, "workloads.jsonl"):
            Problem.from_tree("problem", tree)
        for text in ("", "\n"):
            with self.subTest(text=text):
                tree = Node.directory(".", (
                    item.node("target.json"), item.node("definition.json"),
                    Node.file("workloads.jsonl", text),
                ))
                restored = Problem.from_tree("problem", tree)
                self.assertEqual(restored.workloads, ())

    def test_definition_checked_without_cases(self):
        item = problem()
        for text in ("invalid JSON", "[]"):
            with self.subTest(text=text):
                tree = item.tree.with_child(Node.file("definition.json", text))
                with self.assertRaises((TypeError, ValueError)):
                    Problem.from_tree("problem", tree)

    def test_order_and_roundtrip(self):
        cases = (workload("small", 2), workload("large", 8))
        item = problem(*cases)
        self.assertEqual(item.workloads, cases)
        records = item.node("workloads.jsonl").content.splitlines()
        self.assertEqual(len(records), 2)
        self.assertEqual([json.loads(line)["workload"]["uuid"] for line in records],
                         ["small", "large"])
        restored = from_json(to_json(item), Problem)
        self.assertEqual(restored.workloads, cases)
        self.assertEqual(restored.resolve_inputs(), cases)
        self.assertEqual(restored.resolve_inputs("large"), (cases[1],))
        restored.workloads[0]["axes"]["N"] = 99
        self.assertEqual(restored.workloads, cases)

    def test_uuid_validation(self):
        for cases in (
            (workload("same"), workload("same")),
            (workload(""),),
            (workload("   "),),
            ({"axes": {"N": 2}, "inputs": {"x": {"type": "random"}}},),
        ):
            with self.subTest(cases=cases), self.assertRaisesRegex(
                (TypeError, ValueError), "uuid"
            ):
                problem(*cases)

        item = problem(workload("first"), workload("second"))
        records = item.node("workloads.jsonl").content.splitlines()
        tree = item.tree.with_child(Node.file("workloads.jsonl", "\n".join((records[0], records[0]))))
        with self.assertRaisesRegex(ValueError, "duplicate.*uuid|uuid.*duplicate"):
            Problem.from_tree("problem", tree)

    def test_unknown_selection_rejected(self):
        for item in (problem(), problem(workload("first"))):
            with self.assertRaisesRegex(ValueError, "workload.*missing|unknown workload"):
                item.resolve_inputs("missing")

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
        self.assertEqual(len(item.node("inputs").children), 1)
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

    def test_trace_directory_cases(self):
        source = self.root / "definition.json"
        source.write_text(json.dumps(DEFINITION))
        with self.assertRaisesRegex(FileNotFoundError, "workloads.jsonl"):
            load_trace(self.root)
        records = self.root / "workloads.jsonl"
        records.write_text("\n")
        self.assertEqual(load_trace(self.root).workloads, ())
        cases = (workload("small", 2), workload("large", 8))
        records.write_text("\n".join(json.dumps({
            "definition": "increment", "workload": case,
        }) for case in cases))
        module = load_trace(self.root)
        self.assertEqual(module.workloads, cases)
        self.assertFalse(hasattr(module, "workload"))
        with patch.object(type(get_backend("cuda")), "info", return_value={"platform": "sm90"}):
            loaded = load_problem(self.root, backend=get_backend("cuda"))
        self.assertEqual(loaded.workloads, cases)
        inspected, inspected_path = load_problem_module(self.root)
        self.assertEqual(inspected.workloads, cases)
        self.assertEqual(inspected_path, source)

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

    def test_kernel_directory_reference(self):
        (self.root / "definition.json").write_text(json.dumps(DEFINITION))
        (self.root / "workloads.jsonl").write_text("")
        (self.root / "target.json").write_text(json.dumps({
            "name": "other", "language": "hip", "platform": "hygon",
        }))
        reference = {"path": str(self.root), **problem().target()}
        artifact = Artifact("custom", Node.directory(".", (
            json_node("metadata.json", {"problem": reference}),
        )))
        with patch("klineage.backend.detect_backend", side_effect=AssertionError("local probe")):
            self.assertEqual(load_problem(artifact), problem())
            self.assertEqual(load_problem(reference), problem())

    def test_problem_reference_is_directory(self):
        linked = self.root / "link"
        actual = self.root / "problem"
        actual.mkdir()
        linked.symlink_to(actual, target_is_directory=True)
        reference = problem_module.problem_reference(linked, problem())
        self.assertEqual(reference, {"path": str(actual), **problem().target()})
        filename = self.root / "file"
        filename.touch()
        with self.assertRaises(NotADirectoryError):
            problem_module.problem_reference(filename, problem())

    def test_invalid_target_rejected(self):
        (self.root / "definition.json").write_text(json.dumps(DEFINITION))
        (self.root / "workloads.jsonl").write_text("")
        (self.root / "target.json").mkdir()
        with patch("klineage.backend.detect_backend", side_effect=AssertionError("unneeded device probe")):
            with self.assertRaises(IsADirectoryError):
                load_problem(self.root)

    def test_bare_definition_rejected(self):
        path = self.root / "definition.json"
        path.write_text(json.dumps(DEFINITION))
        with self.assertRaisesRegex((ValueError, NotADirectoryError), "directory"):
            load_trace(path)
        with self.assertRaisesRegex((ValueError, NotADirectoryError), "directory"):
            load_problem(path, backend=get_backend("cuda"))
        path.write_text(to_json(problem(workload("one"))))
        self.assertEqual(load_problem(path), problem(workload("one")))

    def test_python_reference_module(self):
        path = self.root / "reference.py"
        path.write_text("def make_inputs(): return {}\ndef torch_ref(): return ()\n")
        module, source = load_problem_module(path)
        self.assertEqual(module.make_inputs(), {})
        self.assertEqual(source, path)

    def test_trace_module_workloads(self):
        self.assertEqual(trace_module(DEFINITION).workloads, ())
        cases = (workload("one"), workload("two", 4))
        module = trace_module(DEFINITION, cases)
        self.assertEqual(module.workloads, cases)
        self.assertEqual(module.run(3), 4)
        with self.assertRaisesRegex(ValueError, "duplicate.*uuid|uuid.*duplicate"):
            trace_module(DEFINITION, (cases[0], copy.deepcopy(cases[0])))


if __name__ == "__main__":
    unittest.main()

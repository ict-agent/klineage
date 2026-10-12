from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kernel_fixtures import edit_kernel, kernel, make_kernel, skill
from problem_fixtures import problem_spec

from klineage.action import same_implementation
from klineage.artifact import Artifact, Node, file_ref, source_files
from klineage.artifact.problem import Problem
from klineage.serialization import from_json, raw_text, to_json


class ArtifactTreeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_find_names(self):
        first = Node.file("item", "first")
        parent = Node.directory("item", (first,))
        last = Node.file("item", "last")
        unique = Node.file("unique.txt", "only")
        tree = Node.directory(".", (
            parent,
            Node.directory("other", (last, unique)),
        ))
        self.assertEqual(tree.find("item"), [parent, first, last])
        self.assertIs(tree.find("unique.txt"), unique)
        self.assertIs(tree.find("."), tree)
        self.assertIs(first.find("item"), first)
        self.assertEqual(tree.find("missing"), [])
        self.assertEqual(tree.find("other/unique.txt"), [])
        self.assertEqual(tree.find(""), [])
        with self.assertRaisesRegex(TypeError, "string"):
            tree.find(Path("unique.txt"))

        artifact = Artifact("example", tree)
        self.assertEqual(artifact.find("item"), [parent, first, last])
        self.assertIs(artifact.find("unique.txt"), unique)
        self.assertEqual(artifact.find("missing"), [])

    def test_part_needs_unique_name(self):
        class Note(Artifact):
            artifact_kind = "note"

        note = Node.directory("note", (Node.file("text", "hello"),), artifact_type="note")
        value = Artifact("bundle", Node.directory(".", (
            Node.directory("nested", (note,)),
            Node.file("plain", "text"),
        )))
        part = value.part("note", Note)
        self.assertIsInstance(part, Note)
        self.assertEqual(part.find("text").content, "hello")
        self.assertEqual(part.tree.name, ".")
        with self.assertRaisesRegex(KeyError, "missing"):
            value.part("missing")
        with self.assertRaisesRegex(ValueError, "plain.*typed artifact"):
            value.part("plain")

        ambiguous = Artifact("bundle", value.tree.with_child(note))
        with self.assertRaisesRegex(ValueError, "note.*2 matches"):
            ambiguous.part("note", Note)

    def test_check_structure(self):
        expected = Node.directory(".", (
            Node.directory("src", (Node.file("main.cu", "expected"),)),
            Node.file("config.toml", "expected"),
        ))
        actual = Node.directory(".", (
            Node.file("config.toml", "different"),
            Node.directory("src", (Node.file("main.cu", "different"),)),
        ), description="Different annotations.")
        self.assertIsNone(expected.check(actual))
        self.assertEqual(expected.find("main.cu").content, "expected")

    def test_check_mismatch_paths(self):
        expected = Node.directory(".", (
            Node.directory("src", (Node.file("main.cu", "source"),)),
        ))
        cases = (
            (Node.directory("other", expected.children), "name"),
            (Node.directory("."), "src.*missing"),
            (Node.directory(".", (Node.file("src", "source"),)), "src.*directory"),
            (Node.directory(".", (Node.directory("src"),)), "src/main.cu.*missing"),
            (Node.directory(".", (Node.directory("src", (Node.file("other.cu", "source"),)),)), "src/main.cu.*missing"),
            (expected.with_child(Node.file("extra.txt", "extra")), "extra.txt.*unexpected"),
        )
        for actual, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                expected.check(actual)
        with self.assertRaises(TypeError):
            expected.check("not a tree")

    def test_kernel_roundtrip(self):
        original = edit_kernel(kernel(), compile_flags=("-lineinfo",))
        restored = from_json(to_json(original), type(original))
        self.assertEqual(restored, original)
        self.assertEqual(
            restored.find("kernel.cu").content,
            source_files(original.find("submission"))["solution/kernel.cu"],
        )
        self.assertEqual(restored.tree.description, original.tree.description)
        self.assertNotIn(
            "source_files", json.loads(restored.find("metadata.json").content)
        )

    def test_reference_digest(self):
        source = self.root / "data.bin"
        source.write_bytes(b"before")
        node = Node.file("data.bin", reference=file_ref(source))
        result = Artifact("example", Node.directory(".", (node,)))
        source.write_bytes(b"edited")
        with self.assertRaisesRegex(ValueError, "digest|changed"):
            raw_text(result.find("data.bin"))

    def test_invalid_nodes(self):
        for name in ("../escape", "/absolute", "a/b", "a\\b", ".."):
            with self.subTest(name=name), self.assertRaises(ValueError):
                Node.file(name, "text")
        with self.assertRaises(ValueError):
            Node.directory(".", (Node.file("a", "1"), Node.file("a", "2")))

    def test_skill_roundtrip(self):
        original = skill("tiling")
        self.assertEqual(from_json(to_json(original), type(original)), original)
        self.assertIn("preconditions", original.find("SKILL.md").content)


CUDA_SOURCE = "__global__ void kernel() {}"


def problem() -> Problem:
    return Problem(
        name="vector-add",
        definition={
            "name": "vector-add",
            "op_type": "elementwise",
            "description": "Increment every element of x and return the result.",
            "axes": {"N": {"type": "var"}},
            "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
            "outputs": {"result": {"shape": ["N"], "dtype": "float32"}},
            "reference": "def run(x):\n    return x + 1\n",
        },
        workloads=({
            "uuid": "c03709e8-79f7-4d50-b75f-5c82884b13b2",
            "axes": {"N": 1024},
            "inputs": {"x": {"type": "random"}},
        },),
        language="cuda",
        platform="sm90",
    )


class ContractTests(unittest.TestCase):

    def test_identity_binds_io_order(self):
        spec = problem_spec()
        current = make_kernel("gemm", spec, source_files={"kernel.cu": CUDA_SOURCE})
        reordered = {
            **spec.definition,
            "inputs": dict(reversed(spec.definition["inputs"].items())),
        }
        changed = edit_kernel(current, problem=spec.replace(definition=reordered))
        self.assertFalse(same_implementation(current, changed))

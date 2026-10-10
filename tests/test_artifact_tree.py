import json
import tempfile
import unittest
from pathlib import Path

from kernel_fixtures import edit_kernel, kernel, skill

from klineage.action import same_implementation
from klineage.action.action import capture_output
from klineage.artifact import (
    Artifact,
    Node,
    file_ref,
    source_files,
)
from klineage.artifact.problem import load_problem
from klineage.serialization import from_json, raw_text, to_json


class ArtifactTreeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_explicit_artifact_class(self):
        class Report(Artifact):
            artifact_kind = "report.direct"
            schema = Node.directory(".", (Node.file("result.txt"),))

        value = Report(Report.artifact_kind, Node.directory(".", (
            Node.file("result.txt", "measured"),
        )))
        restored = from_json(to_json(value), Report)
        self.assertIs(type(restored), Report)
        self.assertEqual(restored, value)
        generic = from_json(to_json(kernel()), Artifact)
        self.assertIs(type(generic), Artifact)
        with self.assertRaisesRegex(ValueError, "result.txt.*missing"):
            Report(Report.artifact_kind, Node.directory("."))

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
        self.assertEqual(expected.child("src/main.cu").content, "expected")

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

    def test_check_has_no_file_access(self):
        reference = self.root / "input.bin"
        reference.write_bytes(b"data")
        actual = Node.file("input.bin", reference=file_ref(reference))
        reference.unlink()
        self.assertIsNone(Node.file("input.bin", "").check(actual))

    def test_check_empty_is_exact(self):
        with self.assertRaisesRegex(ValueError, "unexpected"):
            Node.directory(".").check(Node.directory(".", (Node.file("file", ""),)))
        with self.assertRaisesRegex(ValueError, "name"):
            Node.file("*", "").check(Node.file("file", ""))

    def test_layout_requires_file_data(self):
        layout = Node.directory(".", (
            Node.directory("src", (Node.file("main.cu"),)),
        ))
        with self.assertRaisesRegex(ValueError, "src/main.cu.*content"):
            Artifact("example.layout", layout)
        value = Node.directory(".", (
            Node.directory("src", (Node.file("main.cu", ""),)),
        ))
        layout.check(value)
        self.assertEqual(Artifact("example.layout", value).node("src/main.cu").content, "")

    def test_kernel_roundtrip(self):
        original = edit_kernel(kernel(), compile_flags=("-lineinfo",))
        restored = from_json(to_json(original), type(original))
        self.assertEqual(restored, original)
        self.assertEqual(
            restored.node("submission/solution/kernel.cu").content,
            source_files(original.node("submission"))["solution/kernel.cu"],
        )
        self.assertEqual(restored.tree.description, original.tree.description)
        self.assertNotIn(
            "source_files", json.loads(restored.node("metadata.json").content)
        )

    def test_reference_digest(self):
        source = self.root / "data.bin"
        source.write_bytes(b"before")
        node = Node.file("data.bin", reference=file_ref(source))
        result = Artifact("example", Node.directory(".", (node,)))
        source.write_bytes(b"edited")
        with self.assertRaisesRegex(ValueError, "digest|changed"):
            raw_text(result.node("data.bin"))

    def test_invalid_nodes(self):
        for name in ("../escape", "/absolute", "a/b", "a\\b", ".."):
            with self.subTest(name=name), self.assertRaises(ValueError):
                Node.file(name, "text")
        with self.assertRaises(ValueError):
            Node.directory(".", (Node.file("a", "1"), Node.file("a", "2")))

    def test_skill_roundtrip(self):
        original = skill("tiling")
        self.assertEqual(from_json(to_json(original), type(original)), original)
        self.assertIn("preconditions", original.node("SKILL.md").content)

    def test_exact_schema_collects_files(self):
        (self.root / "report.txt").write_text("measured result")
        (self.root / "data").mkdir()
        (self.root / "data" / "result.json").write_text('{"latency_ms": 1}')
        (self.root / ".klineage").mkdir()
        (self.root / "AGENTS.md").write_text("runner instructions")
        schema = Node.directory(".", (
            Node.file("report.txt"),
            Node.directory("data", (Node.file("result.json"),)),
        ), description="Report files.")
        tree = capture_output(self.root, schema)
        schema.check(tree)
        result = Artifact("report.files", tree)
        self.assertEqual({node.name for node in result.tree.children}, {"report.txt", "data"})
        self.assertEqual(result.node("data/result.json").content, '{"latency_ms": 1}')
        (self.root / "extra.txt").write_text("unexpected")
        with self.assertRaisesRegex(ValueError, "extra.txt.*unexpected"):
            schema.check(capture_output(self.root, schema))

    def test_backend_bundles(self):
        for language, platform, filename in (
            ("cuda", "nvidia-sm90a-cuda13", "kernel.cu"),
            ("hip", "hygon", "kernel.hip"),
            ("ascendc", "ascend", "kernel.cpp"),
        ):
            with self.subTest(language=language):
                current = kernel()
                current = edit_kernel(current, problem=load_problem(current).replace(
                        language=language, platform=platform
                    ), source_files={filename: "native code"})
                result = from_json(to_json(current), type(current))
                self.assertEqual(result.kind, f"kernel.{language}")
                self.assertEqual(result, current)

    def test_flags_change_identity(self):
        current = kernel()
        changed = edit_kernel(current, compile_flags=("--use_fast_math",))
        self.assertFalse(same_implementation(current, changed))

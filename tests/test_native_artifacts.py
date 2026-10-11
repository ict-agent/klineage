import json
import tempfile
import unittest
from pathlib import Path

from kernel_fixtures import edit_kernel, kernel, skill
from problem_fixtures import problem_spec

from klineage.artifact import (
    Artifact,
    Node,
    file_ref,
    source_files,
)
from klineage.artifact.problem import load_problem
from klineage.serialization import from_json, raw_text, restore, to_json


class NativeArtifactTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_domain_is_artifact(self):
        for value in (kernel(), problem_spec(), skill("tiling")):
            with self.subTest(type=type(value).__name__):
                self.assertIsInstance(value, Artifact)
                restored = from_json(to_json(value), type(value))
                self.assertIs(type(restored), type(value))
                self.assertEqual(restored, value)

    def test_tree_is_authoritative(self):
        original = kernel()
        payload = json.loads(to_json(original))
        restored = restore(payload, type(original))
        self.assertEqual(
            source_files(restored.find("submission")),
            source_files(original.find("submission")),
        )
        self.assertIsInstance(load_problem(restored), Artifact)
        changed = edit_kernel(restored, source_files={"kernel.cu": "new source"})
        self.assertEqual(changed.find("submission").find("solution").find("kernel.cu").content, "new source")
        self.assertNotEqual(changed.find("submission"), original.find("submission"))
        self.assertEqual(source_files(original.find("submission")), source_files(restored.find("submission")))

    def test_json_changes_are_visible(self):
        original = kernel()
        payload = json.loads(to_json(original))
        metadata = next(
            node
            for node in payload["tree"]["children"]
            if node["name"] == "metadata.json"
        )
        value = json.loads(metadata["content"])
        value["name"] = "renamed"
        metadata["content"] = json.dumps(value)
        restored = restore(payload, type(original))
        self.assertEqual(
            json.loads(raw_text(restored.find("metadata.json")))["name"], "renamed"
        )
        self.assertNotEqual(json.loads(raw_text(original.find("metadata.json")))["name"], json.loads(raw_text(restored.find("metadata.json")))["name"])

    def test_example_reference(self):
        path = self.root / "example.txt"
        path.write_text("one complete example\n")
        example = file_ref(path)
        spec = Node.file("kernel.cu", description="Kernel source.", one_shot=example)
        node = Node.file("kernel.cu", "source", description="Kernel source.", one_shot=example)
        spec.check(node)
        self.assertEqual(node.description, "Kernel source.")
        self.assertEqual(node.one_shot, example)
        self.assertEqual(raw_text(node.one_shot), "one complete example\n")
        self.assertEqual(
            raw_text(spec), f"kernel.cu ---> Kernel source. (See: {path} as an example)"
        )
        self.assertNotIn("rules", json.loads(to_json(node)))
        self.assertEqual(from_json(to_json(node), Node), node)
        path.write_text("changed example\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            raw_text(node.one_shot)
        self.assertEqual(
            raw_text(spec), f"kernel.cu ---> Kernel source. (See: {path} as an example)"
        )

    def test_example_keeps_bytes(self):
        path = self.root / "example.txt"
        path.write_bytes(b"example\r\n")
        node = Node.file("source.txt", "source", one_shot=file_ref(path))
        self.assertEqual(raw_text(node.one_shot), "example\r\n")

    def test_check_preserves_annotations(self):
        spec = Node.file("file.txt", description="Description.", one_shot="example")
        node = Node.file("file.txt", "data", description="Actual.", one_shot="actual")
        self.assertIsNone(spec.check(node))
        self.assertEqual(node.description, "Actual.")
        self.assertEqual(raw_text(node.one_shot), "actual")

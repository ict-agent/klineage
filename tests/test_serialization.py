from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from kernel_fixtures import kernel, measured, skill
from problem_fixtures import problem_spec

from klineage.artifact import Artifact, FileRef, Kernel, Node, file_ref, json_node
from klineage.backend import TimingPolicy, TimingResult
from klineage.memory import Scope
from klineage.serialization import (
    SerializationHandler,
    from_json,
    raw_text,
    register_handler,
    restore,
    to_json,
)
from klineage.server.protocol import Evaluation, source_digest, submission


class SerializationTests(unittest.TestCase):
    def test_raw_text_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.txt"
            path.write_bytes(b"example\r\n")
            reference = file_ref(path)
            node = Node.file("example.txt", reference=reference)
            for value in ("example\r\n", Node.file("inline.txt", "example\r\n"), reference, node):
                with self.subTest(type=type(value).__name__):
                    self.assertEqual(raw_text(value), "example\r\n")
            self.assertEqual(raw_text(Node.file("empty.txt", "")), "")
            path.write_bytes(b"altered\r\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                raw_text(reference)
        self.assertEqual(raw_text(Node.directory(".")), "./")

    def test_raw_text_layout(self):
        reference = FileRef("/examples/main.cu", "0" * 64, 4)
        layout = Node.directory(".", (
            Node.directory("src", (
                Node.file(
                    "main.cu", "source contents are omitted",
                    description="Host entry.\nInclude launch parameters.",
                    one_shot=reference,
                ),
                Node.directory("empty"),
            ), description="Sources."),
            Node.file("config.toml", reference=reference, description="Build settings."),
        ), description="Kernel.")
        self.assertEqual(raw_text(layout), (
            "./ ---> Kernel.\n"
            "├── src/ ---> Sources.\n"
            "│   ├── main.cu ---> Host entry.\n"
            "│   │                Include launch parameters."
            " (See: /examples/main.cu as an example)\n"
            "│   └── empty/\n"
            "└── config.toml ---> Build settings."
        ))

    def test_dataclass_roundtrips(self):
        values = (
            FileRef("/missing/input", "0" * 64, 4),
            Scope(languages=("cuda", "hip")),
            TimingPolicy(warmup=1, repeat=2, trials=1),
            TimingResult(((1.0, 2.0),), 1.5, "cupti", {"cupti_version": "13.0"}),
        )
        for value in values:
            with self.subTest(type=type(value).__name__):
                self.assertEqual(from_json(to_json(value), type(value)), value)

    def test_schema_roundtrip(self):
        schema = Node.directory(
            ".",
            description="Report.",
            children=(
                Node.file(
                    "result.txt",
                    description="Result.",
                    one_shot="example",
                ),
            ),
        )
        self.assertEqual(from_json(to_json(schema), Node), schema)
        self.assertEqual(
            raw_text(schema),
            "./ ---> Report.\n└── result.txt ---> Result. (See: example as an example)",
        )

    def test_artifact_types_and_format(self):
        for value in (kernel(), problem_spec(), skill("tiling")):
            with self.subTest(type=type(value).__name__):
                data = json.loads(to_json(value))
                self.assertEqual(set(data), {"artifact_type", "tree"})
                self.assertFalse(hasattr(value, "version"))
                generic = from_json(to_json(value), Artifact)
                self.assertEqual(generic.tree, value.tree)
                self.assertIs(type(generic), Artifact)
                self.assertEqual(restore(data, type(value)), value)
        with self.assertRaisesRegex(TypeError, "not a Kernel"):
            from_json(to_json(problem_spec()), Kernel)
        with self.assertRaisesRegex(ValueError, "envelope"):
            restore({"artifact_type": "problem"}, Artifact)

    def test_runtime_callable_is_omitted(self):
        current = kernel()
        object.__setattr__(current, "function", lambda value: value + 1)
        restored = from_json(to_json(current), Kernel)
        self.assertIsNone(restored.function)
        self.assertEqual(restored, current)

    def test_evaluation_transport(self):
        current = kernel()
        validation = measured(current)
        metadata = json.loads(current.find("metadata.json").content)
        current = type(current)(
            current.tree.with_child(
                json_node(
                    "metadata.json",
                    metadata | {"validation": validation},
                )
            )
        )
        result = Evaluation(
            "job",
            "problem",
            source_digest("problem", submission(current)),
            current,
            validation,
            {"worker.log": "measured"},
        )
        data = json.loads(to_json(result))
        self.assertIsNone(data["error"])
        self.assertEqual(restore(data, Evaluation), result)
        metadata = next(
            node
            for node in data["kernel"]["tree"]["children"]
            if node["name"] == "metadata.json"
        )
        metadata.pop("content")
        metadata["reference"] = {
            "path": "/private/metadata.json",
            "sha256": "0" * 64,
            "size": 4,
        }
        with patch(
            "klineage.serialization.FileRefHandler.raw_text", side_effect=AssertionError("untrusted read")
        ):
            with self.assertRaisesRegex(ValueError, "must be inline"):
                restore(data, Evaluation)

    def test_custom_handler_inheritance(self):
        @dataclass
        class Label:
            text: str

        class DetailedLabel(Label):
            pass

        class LabelHandler(SerializationHandler):
            def default(self, value):
                return {"label": value.text}

            def restore(self, data, object_type):
                return object_type(data["label"])

            def raw_text(self, value):
                return value.text

        register_handler(Label, LabelHandler())
        value = DetailedLabel("compile")
        self.assertEqual(json.loads(to_json(value)), {"label": "compile"})
        self.assertEqual(from_json(to_json(value), DetailedLabel), value)
        self.assertEqual(raw_text(value), "compile")
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_handler(Label, LabelHandler())

    def test_nonfinite_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                to_json({"value": value})


if __name__ == "__main__":
    unittest.main()

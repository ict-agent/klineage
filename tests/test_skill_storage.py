import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import yaml

from klineage.memory import (
    Scope,
    SkillCard,
    load_memory,
    load_skill,
    save_memory,
    save_skill,
)
from klineage.prompts import render_prompt

SKILL_BODY = """# Overview

Load each operand tile once and reuse it across threads.

# Precondition

- Shared memory is available.

# Scope

- Cases: *
- Languages: *
- Platforms: *

# Code Change Snippet

## Before

```cuda
read_global();
```

## After

```cuda
read_shared("输入");
```
"""


class SkillStorageTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.card = SkillCard(
            skill_id="gemm.staging",
            intent="Stage operands in shared memory for reuse across threads.",
            preconditions=("Shared memory is available.",),
            scope=Scope(),
            body=SKILL_BODY,
        )

    def test_markdown_roundtrip(self):
        path = self.root / "SKILL.md"
        save_skill(self.card, path)
        self.assertEqual(load_skill(path), self.card)
        text = path.read_text()
        _, frontmatter, body = text.split("---\n", 2)
        self.assertEqual(yaml.safe_load(frontmatter), self.card.to_metadata())
        self.assertNotIn("read_shared", frontmatter)
        self.assertNotIn("Before", frontmatter)
        self.assertEqual(body.strip(), self.card.body)

    def test_intent_is_single_line(self):
        with self.assertRaisesRegex(ValueError, "intent"):
            replace(self.card, intent=self.card.intent + "\n\n" + SKILL_BODY)

    def test_rejects_extra_payload(self):
        path = self.root / "SKILL.md"
        save_skill(self.card, path)
        path.write_text(
            path.read_text().replace("skill_id:", "unexpected: {}\nskill_id:", 1)
        )
        with self.assertRaises(ValueError):
            load_skill(path)

    def test_multiline_roundtrip(self):
        card = replace(
            self.card,
            body=self.card.body + "\n\n## Notes\n\n```text\n---\n```",
            preconditions=('Layouts: "row-major".\nKeep strides.',),
        )
        path = self.root / "SKILL.md"
        save_skill(card, path)
        self.assertEqual(load_skill(path), card)

    def test_memory_roundtrip(self):
        path = self.root / "memory.json"
        save_memory((self.card, self.card), path)
        self.assertEqual(load_memory(path), (self.card,))
        payload = json.loads(path.read_text())[0]
        self.assertEqual(
            set(payload), {"skill_id", "intent", "preconditions", "scope", "body"}
        )
        self.assertEqual(SkillCard.from_dict(payload), self.card)

        with self.assertRaises(TypeError):
            save_memory([[self.card]], path)
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            save_memory((self.card, replace(self.card, intent="conflict")), path)
        self.assertEqual(load_memory(path), (self.card,))

    def test_rejects_invalid_memory(self):
        path = self.root / "memory.json"
        for payload in ({}, [[self.card.to_dict()]], [{"skill_id": "incomplete"}]):
            with self.subTest(payload=payload):
                path.write_text(json.dumps(payload))
                with self.assertRaises((TypeError, ValueError)):
                    load_memory(path)

    def test_rejects_invalid_fields(self):
        for changes in (
            {"skill_id": 1},
            {"body": None},
            {"preconditions": "shared"},
            {"preconditions": [1]},
            {"scope": {**self.card.scope.to_dict(), "languages": "cuda"}},
            {"unexpected": True},
        ):
            with (
                self.subTest(changes=changes),
                self.assertRaises((TypeError, ValueError)),
            ):
                SkillCard.from_dict({**self.card.to_dict(), **changes})

    def test_prompt_example_roundtrip(self):
        prompt = render_prompt("decompose_gemm")
        document = prompt.split("````markdown\n", 1)[1].rsplit("\n````", 1)[0]
        path = self.root / "SKILL.md"
        path.write_text(document)
        card = load_skill(path)
        self.assertEqual(card.skill_id, "gemm.shared-memory-staging")
        self.assertEqual(len(card.intent.splitlines()), 1)
        self.assertNotIn("```", card.intent)
        self.assertIn("__syncthreads();", card.body)
        save_skill(card, path)
        self.assertEqual(load_skill(path), card)

    def test_preserves_body_edits(self):
        path = self.root / "SKILL.md"
        save_skill(self.card, path)
        header, body = path.read_text().split("\n---\n", 1)
        changed = body.replace("read_shared", "read_shared_tile")
        path.write_text(header + "\n---\n" + changed)
        restored = load_skill(path)
        self.assertEqual(restored.intent, self.card.intent)
        self.assertEqual(restored.body, changed.strip())
        save_skill(restored, path)
        self.assertEqual(load_skill(path), restored)

    def test_rejects_yaml_objects(self):
        path = self.root / "SKILL.md"
        path.write_text("---\n!!python/object:builtins.object {}\n---\n# unsafe\n")
        with self.assertRaises(ValueError):
            load_skill(path)

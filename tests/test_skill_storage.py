from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml
from kernel_fixtures import kernel, skill
from problem_fixtures import problem_spec

from klineage.artifact.problem import load_problem
from klineage.memory import Scope, SkillCard, load_skill, retrieve, save_skill
from klineage.serialization import to_json

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

    def test_multiline_roundtrip(self):
        card = self.card.replace(body=self.card.body + '\n\n## Notes\n\n```text\n---\n```', preconditions=('Layouts: "row-major".\nKeep strides.',))
        path = self.root / "SKILL.md"
        save_skill(card, path)
        self.assertEqual(load_skill(path), card)

    def test_rejects_invalid_fields(self):
        for changes in (
            {"skill_id": 1},
            {"body": None},
            {"preconditions": "shared"},
            {"preconditions": [1]},
            {"scope": {**json.loads(to_json(self.card.scope)), "languages": "cuda"}},
            {"unexpected": True},
        ):
            with (
                self.subTest(changes=changes),
                self.assertRaises((TypeError, ValueError)),
            ):
                self.card.replace(**changes)

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


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.target = kernel()
        self.card = skill(
            "staging",
            declared=Scope(cases=("gemm",), languages=("cuda",), platforms=("sm120",)),
        )

    def test_applicable_card_is_returned(self):
        self.assertEqual(retrieve((self.card,), self.target), (self.card,))
        self.assertEqual(retrieve((self.card,), load_problem(self.target)), (self.card,))

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
                (self.card, self.card.replace(intent='different edit')), self.target
            )

    def test_scope_matches_target(self):
        for problem in (
            problem_spec("topk"),
            problem_spec(language="python"),
            problem_spec(platform="sm90"),
        ):
            with self.subTest(problem=problem):
                self.assertEqual(retrieve((self.card,), problem), ())

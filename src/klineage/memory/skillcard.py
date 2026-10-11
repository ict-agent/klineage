"""Optimization recipes and their artifact format."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import yaml

from klineage.artifact.tree import (
    Artifact,
    ArtifactKind,
    Node,
)
from klineage.serialization import raw_text, restore
from klineage.utils import mapping, nonempty, string_tuple


def string_sequence(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence of strings")
    return string_tuple(value, label)


@dataclass(frozen=True, slots=True)
class Scope:
    """Nonempty target selectors used to retrieve an optimization recipe.

    Retrieval requires a match in all three dimensions. Each dimension accepts
    exact values or the whole-value wildcard ``*``; these are not glob patterns.
    Matching scope does not establish that source-level preconditions hold.

    Attributes:
        cases: Operator types from ``definition['op_type']``.
        languages: Backend language identifiers.
        platforms: Backend platform identifiers.
    """

    cases: tuple[str, ...] = ("*",)
    languages: tuple[str, ...] = ("*",)
    platforms: tuple[str, ...] = ("*",)

    def __post_init__(self):
        object.__setattr__(self, "cases", string_tuple(self.cases, "scope case"))
        object.__setattr__(
            self,
            "languages",
            string_tuple(self.languages, "scope language"),
        )
        object.__setattr__(
            self,
            "platforms",
            string_tuple(self.platforms, "scope platform"),
        )
        if not self.cases or not self.languages or not self.platforms:
            raise ValueError("scope case, language, and platform cannot be empty")


FRONTMATTER = "---\n"


SKILLCARD = Node.directory(
    ".",
    description="One independently reusable forward optimization.",
    children=(
        Node.file(
            "SKILL.md",
            description="\n".join(
                (
                    "YAML metadata followed by the standalone Markdown recipe.",
                    "Frontmatter has exactly skill_id, intent, preconditions and scope.",
                    "Intent is a brief single line. Scope has nonempty cases, languages and platforms arrays.",
                    "Use Overview, Precondition, Scope and Code Change Snippet sections with Before and After code.",
                    "Cover Data types, Layout, Storage, Pipeline and Hardware in both metadata and prose.",
                    "Each prerequisite needs a dependency in the transformation; example settings belong in Overview.",
                    "Pipeline covers ordering, visibility and safe buffer reuse; a fixed stage count needs a dependency.",
                    "The recipe must restore the optimization from the predecessor alone, without reading the expert.",
                    "Keep SKILL.md outside the kernel source bundle.",
                )
            ),
        ),
    ),
)


class SkillCard(Artifact):
    """One reusable forward optimization stored entirely in ``SKILL.md``.

    YAML frontmatter holds exactly ``skill_id``, ``intent``, ``preconditions``
    and ``scope``; the remaining Markdown is the recipe. Properties parse this
    document, so metadata has no separate mutable representation. The artifact
    specification describes recommended recipe sections; runtime validation
    checks metadata and a nonempty body, not the recipe's semantic correctness.

    Args:
        skill_id (str): Nonempty stable identifier used for retrieval and deduplication.
        intent (str): Nonempty single-line description of the transformation.
        preconditions (Sequence[str]): Nonempty source or execution requirements.
        scope: Target selectors for retrieval.
        body (str): Nonempty Markdown explaining how to restore the optimization.

    Examples:
        >>> card = SkillCard("reuse", "Reuse a buffer", ("Lifetimes do not overlap",),
        ...                  Scope(languages=("cuda",)), "## Overview\\nReuse storage.")
        >>> card.scope.languages
        ('cuda',)
    """

    __slots__ = ()
    artifact_kind = ArtifactKind.SKILLCARD
    schema = SKILLCARD

    def __init__(self, skill_id, intent, preconditions, scope: Scope, body):
        if not isinstance(scope, Scope):
            raise TypeError("skill scope must be a Scope")
        header = yaml.safe_dump(
            {
                "skill_id": nonempty(skill_id, "skill_id"),
                "intent": nonempty(intent, "intent"),
                "preconditions": list(string_sequence(preconditions, "preconditions")),
                "scope": asdict(scope),
            },
            sort_keys=False,
            allow_unicode=True,
        )
        text = f"{FRONTMATTER}{header}{FRONTMATTER}\n{nonempty(body, 'body')}\n"
        Artifact.__init__(
            self,
            self.artifact_kind,
            Node.directory(".", (Node.file("SKILL.md", text),)),
        )

    def document(self):
        """Parse and return fresh frontmatter plus the stripped Markdown body."""
        text = raw_text(self.find("SKILL.md"))
        header, separator, body = text[len(FRONTMATTER) :].partition(
            f"\n{FRONTMATTER}\n"
        )
        if not text.startswith(FRONTMATTER) or not separator:
            raise ValueError("SKILL.md requires YAML frontmatter and Markdown sections")
        try:
            header = yaml.safe_load(header)
        except yaml.YAMLError as error:
            raise ValueError("invalid SKILL.md YAML") from error
        if not isinstance(header, dict):
            raise TypeError("SKILL.md frontmatter must be a mapping")
        if set(header) != {"skill_id", "intent", "preconditions", "scope"}:
            raise ValueError("SKILL.md frontmatter requires four metadata fields")
        return header, body.strip()

    def validate(self):
        """Check frontmatter fields, scope and body without evaluating the recipe."""
        super().validate()
        header, body = self.document()
        for name in ("skill_id", "intent"):
            nonempty(header[name], name)
        if len(header["intent"].splitlines()) != 1:
            raise ValueError("intent must be a brief single-line description")
        if not string_sequence(header["preconditions"], "preconditions"):
            raise ValueError("a SkillCard needs at least one precondition")
        restore(mapping(header["scope"], "skill scope"), Scope)
        nonempty(body, "body")

    @property
    def skill_id(self):
        """Stable identifier stored in frontmatter."""
        return self.document()[0]["skill_id"]

    @property
    def intent(self):
        """Single-line transformation description."""
        return self.document()[0]["intent"]

    @property
    def preconditions(self):
        """Requirements the caller must check against the current implementation."""
        return tuple(self.document()[0]["preconditions"])

    @property
    def scope(self):
        """Target selectors reconstructed from frontmatter."""
        return restore(self.document()[0]["scope"], Scope)

    @property
    def body(self):
        """Standalone Markdown recipe without frontmatter."""
        return self.document()[1]

    def to_metadata(self):
        """Return parsed frontmatter without the Markdown body."""
        return self.document()[0]

    def replace(self, **changes):
        """Return a new card with selected constructor fields replaced."""
        fields = {
            name: getattr(self, name)
            for name in ("skill_id", "intent", "preconditions", "scope", "body")
        }
        return type(self)(**(fields | changes))

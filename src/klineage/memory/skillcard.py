"""Portable optimization recipes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from klineage._utils import mapping, nonempty, string_tuple


def _strings(value: Any, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a sequence of strings")
    return string_tuple(value, label)


@dataclass(frozen=True, slots=True)
class Scope:
    """The target contexts where a skill is expected to apply."""

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": list(self.cases),
            "languages": list(self.languages),
            "platforms": list(self.platforms),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Scope:
        if set(value) != {"cases", "languages", "platforms"}:
            raise ValueError("scope requires cases, languages, and platforms")
        return cls(
            cases=_strings(value["cases"], "scope cases"),
            languages=_strings(value["languages"], "scope languages"),
            platforms=_strings(value["platforms"], "scope platforms"),
        )


@dataclass(frozen=True, slots=True)
class SkillCard:
    """Brief optimization metadata and an independent Markdown recipe."""

    skill_id: str
    intent: str
    preconditions: tuple[str, ...]
    scope: Scope
    body: str

    def __post_init__(self):
        for name in ("skill_id", "intent", "body"):
            object.__setattr__(self, name, nonempty(getattr(self, name), name))

        if len(self.intent.splitlines()) != 1:
            raise ValueError("intent must be a brief single-line description")

        object.__setattr__(
            self,
            "preconditions",
            string_tuple(self.preconditions, "preconditions"),
        )
        if not self.preconditions:
            raise ValueError("a SkillCard needs at least one precondition")

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_metadata(), "body": self.body}

    def to_metadata(self) -> dict[str, Any]:
        """Return the four SKILL.md frontmatter fields."""

        return {
            "skill_id": self.skill_id,
            "intent": self.intent,
            "preconditions": list(self.preconditions),
            "scope": self.scope.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SkillCard:
        if set(value) != {"skill_id", "intent", "preconditions", "scope", "body"}:
            raise ValueError("payload differs from the SkillCard schema")
        return cls(
            skill_id=value["skill_id"],
            intent=value["intent"],
            preconditions=_strings(value["preconditions"], "preconditions"),
            scope=Scope.from_dict(mapping(value["scope"], "skill scope")),
            body=value["body"],
        )


__all__ = [
    "Scope",
    "SkillCard",
]

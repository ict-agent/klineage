"""Validated kernel transitions arranged as an executable lineage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from klineage._utils import mapping
from klineage.kernel import Kernel
from klineage.memory.skillcard import SkillCard, TransitionEvidence


class Termination(StrEnum):
    UNKNOWN = "unknown"
    COMPLETE = "complete"
    STEP_LIMIT = "step_limit"
    REJECTION_LIMIT = "rejection_limit"


@dataclass(frozen=True, slots=True)
class Lineage:
    """An executable curriculum from a naive state to an expert state."""

    states: tuple[Kernel, ...]
    transitions: tuple[TransitionEvidence, ...]
    skills: tuple[SkillCard, ...]
    termination: Termination = Termination.UNKNOWN
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "transitions", tuple(self.transitions))
        object.__setattr__(self, "skills", tuple(self.skills))
        if not isinstance(self.termination, Termination):
            raise TypeError("termination must be a Termination")
        if not self.states:
            raise ValueError("a lineage must contain at least one kernel state")
        if len(self.states) != len(self.transitions) + 1:
            raise ValueError("a lineage must have exactly one more state than edges")
        for index, transition in enumerate(self.transitions):
            if transition.before_fingerprint != self.states[index].fingerprint:
                raise ValueError(f"transition {index} does not start at state {index}")
            if transition.after_fingerprint != self.states[index + 1].fingerprint:
                message = f"transition {index} does not end at state {index + 1}"
                raise ValueError(message)

    @property
    def naive_kernel(self) -> Kernel:
        return self.states[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "states": [state.to_dict() for state in self.states],
            "transitions": [transition.to_dict() for transition in self.transitions],
            "skills": [card.to_dict() for card in self.skills],
            "termination": self.termination.value,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Lineage:
        return cls(
            states=tuple(
                Kernel.from_dict(mapping(item, "lineage state"))
                for item in value["states"]
            ),
            transitions=tuple(
                TransitionEvidence.from_dict(mapping(item, "lineage transition"))
                for item in value["transitions"]
            ),
            skills=tuple(
                SkillCard.from_dict(mapping(item, "lineage skill"))
                for item in value["skills"]
            ),
            termination=Termination(value.get("termination", "unknown")),
            reason=str(value.get("reason", "")),
        )


__all__ = ["Lineage", "Termination"]

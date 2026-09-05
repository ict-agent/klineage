"""The paper's SkillCard representation and verification evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from klineage._utils import boolean, mapping, nonempty, string_tuple
from klineage.harness.eval import ValidationResult
from klineage.kernel import TargetContext


class SkillAdmission(StrEnum):
    OFF = "off"
    ON = "on"


@dataclass(frozen=True, slots=True)
class Scope:
    """The target contexts where a skill is expected to apply."""

    cases: tuple[str, ...] = ("*",)
    languages: tuple[str, ...] = ("*",)
    platforms: tuple[str, ...] = ("*",)
    prior_actions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
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
        object.__setattr__(
            self,
            "prior_actions",
            string_tuple(self.prior_actions, "scope prior action"),
        )
        if not self.cases or not self.languages or not self.platforms:
            raise ValueError("scope case, language, and platform cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": list(self.cases),
            "languages": list(self.languages),
            "platforms": list(self.platforms),
            "prior_actions": list(self.prior_actions),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Scope:
        return cls(
            cases=tuple(str(item) for item in value.get("cases", ("*",))),
            languages=tuple(str(item) for item in value.get("languages", ("*",))),
            platforms=tuple(str(item) for item in value.get("platforms", ("*",))),
            prior_actions=tuple(str(item) for item in value.get("prior_actions", ())),
        )


@dataclass(frozen=True, slots=True)
class TransitionEvidence:
    """Evidence attached to one validated concrete forward transition."""

    action_category: str
    locus: str
    before_fingerprint: str
    after_fingerprint: str
    backward_edit: str
    forward_edit: str
    predecessor_validation: ValidationResult
    roundtrip_validation: ValidationResult
    rejected_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "action_category",
            "locus",
            "before_fingerprint",
            "after_fingerprint",
            "backward_edit",
            "forward_edit",
        ):
            object.__setattr__(self, name, nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "rejected_evidence",
            string_tuple(self.rejected_evidence, "rejected evidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_category": self.action_category,
            "locus": self.locus,
            "before_fingerprint": self.before_fingerprint,
            "after_fingerprint": self.after_fingerprint,
            "backward_edit": self.backward_edit,
            "forward_edit": self.forward_edit,
            "predecessor_validation": self.predecessor_validation.to_dict(),
            "roundtrip_validation": self.roundtrip_validation.to_dict(),
            "rejected_evidence": list(self.rejected_evidence),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TransitionEvidence:
        return cls(
            action_category=str(value["action_category"]),
            locus=str(value["locus"]),
            before_fingerprint=str(value["before_fingerprint"]),
            after_fingerprint=str(value["after_fingerprint"]),
            backward_edit=str(value["backward_edit"]),
            forward_edit=str(value["forward_edit"]),
            predecessor_validation=ValidationResult.from_dict(
                mapping(
                    value["predecessor_validation"],
                    "predecessor validation",
                )
            ),
            roundtrip_validation=ValidationResult.from_dict(
                mapping(value["roundtrip_validation"], "roundtrip validation")
            ),
            rejected_evidence=tuple(
                str(item) for item in value.get("rejected_evidence", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class VerificationTrial:
    """One materialization trial recorded in a SkillCard verification log."""

    target: TargetContext
    validation: ValidationResult
    expected_effect_observed: bool
    held_out: bool

    def __post_init__(self) -> None:
        if not isinstance(self.expected_effect_observed, bool):
            raise TypeError("expected_effect_observed must be a bool")
        if not isinstance(self.held_out, bool):
            raise TypeError("held_out must be a bool")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "validation": self.validation.to_dict(),
            "expected_effect_observed": self.expected_effect_observed,
            "held_out": self.held_out,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VerificationTrial:
        return cls(
            target=TargetContext.from_dict(
                mapping(value["target"], "verification target")
            ),
            validation=ValidationResult.from_dict(
                mapping(value["validation"], "verification validation")
            ),
            expected_effect_observed=boolean(
                value["expected_effect_observed"],
                "expected_effect_observed",
            ),
            held_out=boolean(value["held_out"], "held_out"),
        )


@dataclass(frozen=True, slots=True)
class SkillCard:
    """The paper's nine-field reusable optimization representation.

    The nine serialized paper fields are intent, anchor, carrier, precondition,
    effect, evidence, risk, scope, and verification_log. ``skill_id`` is the
    stable handle used by callers and storage.
    """

    skill_id: str
    intent: str
    anchor: str
    carrier: str
    preconditions: tuple[str, ...]
    effects: tuple[str, ...]
    evidence: tuple[TransitionEvidence, ...]
    risks: tuple[str, ...]
    scope: Scope
    verification_log: tuple[VerificationTrial, ...] = ()

    def __post_init__(self) -> None:
        for name in ("skill_id", "intent", "anchor", "carrier"):
            object.__setattr__(self, name, nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "preconditions",
            string_tuple(self.preconditions, "precondition"),
        )
        object.__setattr__(self, "effects", string_tuple(self.effects, "effect"))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "risks", string_tuple(self.risks, "risk"))
        object.__setattr__(self, "verification_log", tuple(self.verification_log))
        if not self.evidence:
            raise ValueError("a SkillCard must be anchored by transition evidence")
        if not self.preconditions or not self.effects:
            raise ValueError("a SkillCard needs at least one precondition and effect")
        categories = {item.action_category for item in self.evidence}
        if len(categories) != 1:
            raise ValueError("a SkillCard's evidence must share one action category")

    @property
    def admitted(self) -> bool:
        """A hypothesis is reusable only after a successful held-out trial."""

        return any(
            trial.held_out
            and trial.validation.accepted
            and trial.expected_effect_observed
            for trial in self.verification_log
        )

    @property
    def action_category(self) -> str:
        return self.evidence[0].action_category

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "intent": self.intent,
            "anchor": self.anchor,
            "carrier": self.carrier,
            "precondition": list(self.preconditions),
            "effect": list(self.effects),
            "evidence": [item.to_dict() for item in self.evidence],
            "risk": list(self.risks),
            "scope": self.scope.to_dict(),
            "verification_log": [trial.to_dict() for trial in self.verification_log],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SkillCard:
        return cls(
            skill_id=str(value["skill_id"]),
            intent=str(value["intent"]),
            anchor=str(value["anchor"]),
            carrier=str(value["carrier"]),
            preconditions=tuple(str(item) for item in value["precondition"]),
            effects=tuple(str(item) for item in value["effect"]),
            evidence=tuple(
                TransitionEvidence.from_dict(mapping(item, "skill evidence"))
                for item in value["evidence"]
            ),
            risks=tuple(str(item) for item in value["risk"]),
            scope=Scope.from_dict(mapping(value["scope"], "skill scope")),
            verification_log=tuple(
                VerificationTrial.from_dict(mapping(item, "skill verification trial"))
                for item in value.get("verification_log", ())
            ),
        )


__all__ = [
    "Scope",
    "SkillAdmission",
    "SkillCard",
    "TransitionEvidence",
    "VerificationTrial",
]

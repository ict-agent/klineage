"""External validation contracts for generated kernels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from klineage._utils import boolean, mapping, optional_float

if TYPE_CHECKING:
    from klineage.kernel import Kernel
    from klineage.memory.skillcard import SkillCard


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """One compile/correctness/profile gate result.

    ``reference_latency_ms`` is the latency of the comparison kernel, so a
    ``relative_performance`` greater than one means the candidate is faster.
    """

    compile_passed: bool
    correctness_passed: bool
    profile_passed: bool
    latency_ms: float | None = None
    reference_latency_ms: float | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("compile_passed", "correctness_passed", "profile_passed"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        for name in ("latency_ms", "reference_latency_ms"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        object.__setattr__(self, "details", dict(self.details))

    @property
    def accepted(self) -> bool:
        return self.compile_passed and self.correctness_passed and self.profile_passed

    @property
    def relative_performance(self) -> float | None:
        if self.latency_ms is None or self.reference_latency_ms is None:
            return None
        return self.reference_latency_ms / self.latency_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "compile_passed": self.compile_passed,
            "correctness_passed": self.correctness_passed,
            "profile_passed": self.profile_passed,
            "latency_ms": self.latency_ms,
            "reference_latency_ms": self.reference_latency_ms,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ValidationResult:
        return cls(
            compile_passed=boolean(value["compile_passed"], "compile_passed"),
            correctness_passed=boolean(
                value["correctness_passed"],
                "correctness_passed",
            ),
            profile_passed=boolean(value["profile_passed"], "profile_passed"),
            latency_ms=optional_float(value.get("latency_ms")),
            reference_latency_ms=optional_float(value.get("reference_latency_ms")),
            details=dict(mapping(value.get("details", {}), "validation details")),
        )


class KernelEvaluator(Protocol):
    """Compile, check correctness, and profile a kernel."""

    def evaluate(
        self,
        kernel: Kernel,
        *,
        reference: Kernel | None = None,
    ) -> ValidationResult: ...


class EffectVerifier(Protocol):
    """Verify that a materialized skill reproduced its expected effect."""

    def __call__(
        self,
        skill: SkillCard,
        baseline: Kernel,
        candidate: Kernel,
    ) -> bool: ...


__all__ = ["EffectVerifier", "KernelEvaluator", "ValidationResult"]

"""NCU capture options and evidence bound to one kernel state."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from klineage.kernel import Kernel

_SETS = frozenset({"basic", "detailed", "full"})
_SECTION = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_METRIC_FIELDS = frozenset({"kernel", "launch_id", "section", "metric", "unit", "value"})


@dataclass(frozen=True, slots=True)
class ProfileOptions:
    set: str = "detailed"
    sections: tuple[str, ...] = ()
    kernel_filter: str | None = None
    timeout_seconds: int = 180

    def __post_init__(self) -> None:
        if self.set not in _SETS:
            raise ValueError("unknown NCU set")
        sections = tuple(self.sections)
        if any(not isinstance(s, str) or not _SECTION.fullmatch(s) for s in sections):
            raise ValueError("invalid NCU section identifier")
        object.__setattr__(self, "sections", sections)
        if self.kernel_filter is not None:
            if not isinstance(self.kernel_filter, str) or not self.kernel_filter.strip():
                raise ValueError("kernel_filter must be a nonempty regex")
            re.compile(self.kernel_filter)
        if type(self.timeout_seconds) is not int or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive integer")


@dataclass(frozen=True, slots=True)
class KernelProfile:
    kernel_fingerprint: str
    target: Mapping[str, Any]
    options: ProfileOptions
    tool: str
    tool_version: str
    device: Mapping[str, Any]
    metrics: tuple[Mapping[str, str], ...]
    report_path: str
    raw_path: str
    collected_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.options, ProfileOptions):
            raise TypeError("profile options must be ProfileOptions")
        if self.tool != "ncu":
            raise ValueError("unsupported profiler")
        for name in ("kernel_fingerprint", "tool_version", "report_path", "raw_path",
                     "collected_at"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"profile {name} must be nonempty")
        for name in ("target", "device"):
            object.__setattr__(self, name, dict(getattr(self, name)))
        metrics = tuple(dict(row) for row in self.metrics)
        if not metrics or any(not _METRIC_FIELDS.issubset(row) for row in metrics):
            raise ValueError("profile requires captured kernel metrics")
        if any(not row["kernel"] or not row["metric"] for row in metrics):
            raise ValueError("profile metrics require kernel and metric names")
        object.__setattr__(self, "metrics", metrics)

    def matches(self, kernel: "Kernel") -> bool:
        target = kernel.context.to_dict()
        target.pop("prior_actions")
        return self.kernel_fingerprint == kernel.fingerprint and self.target == target

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["options"]["sections"] = list(self.options.sections)
        value["metrics"] = list(value["metrics"])
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KernelProfile":
        return cls(**{**value, "options": ProfileOptions(**value["options"])})


__all__ = ["KernelProfile", "ProfileOptions"]

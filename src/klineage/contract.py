"""Derive call signatures from Trace problems; keep artifact loading separate."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any

from klineage.constants import BUNDLE_CONFIG, BUNDLE_SOLUTION
from klineage.utils import mapping, nonempty


class ValueRole(StrEnum):
    INPUTS = "inputs"
    OUTPUTS = "outputs"


def json_value(value: Any, label: str) -> Any:
    """Return a detached JSON value and reject lossy/non-portable values."""

    def check(item: Any, location: str):
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float):
            if not (-float("inf") < item < float("inf")):
                raise ValueError(f"{location} must not contain NaN or infinity")
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise TypeError(f"{location} keys must be strings")
                check(child, f"{location}.{key}")
            return
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for index, child in enumerate(item):
                check(child, f"{location}[{index}]")
            return
        raise TypeError(f"{location} must contain only JSON values")

    check(value, label)
    # A JSON roundtrip both detaches nested mutable objects and normalizes
    # accepted Sequence implementations to ordinary lists.
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def json_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    normalized = json_value(mapping(value, label), label)
    assert isinstance(normalized, dict)
    return normalized


def description(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value.strip()


def relative_source_path(value: str, label: str = "source path") -> str:
    """Validate and normalize a portable relative path inside a source tree."""

    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    raw = value
    if not raw or raw != raw.strip():
        raise ValueError(f"{label} must be a normalized relative path")
    if "\x00" in raw:
        raise ValueError(f"{label} must not contain a NUL byte")
    if "\\" in raw:
        raise ValueError(f"{label} must use '/' path separators")
    path = PurePosixPath(raw)
    parts = raw.split("/")
    if (
        path.is_absolute()
        or ":" in parts[0]
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError(f"{label} must be a normalized relative path")
    normalized = path.as_posix()
    if normalized != raw:
        raise ValueError(f"{label} must be a normalized relative path")
    return normalized


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """A FlashInfer Trace definition and workload on one execution target."""

    name: str
    definition: Mapping[str, Any]
    workload: Mapping[str, Any]
    language: str
    platform: str

    def __post_init__(self):
        for name in ("name", "language", "platform"):
            object.__setattr__(
                self, name, nonempty(getattr(self, name), f"problem {name}")
            )
        for name in ("definition", "workload"):
            object.__setattr__(
                self, name, json_mapping(getattr(self, name), f"problem {name}")
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "definition": json_mapping(self.definition, "problem definition"),
            "workload": json_mapping(self.workload, "problem workload"),
            "language": self.language,
            "platform": self.platform,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProblemSpec:
        value = mapping(value, "problem spec")
        return cls(
            name=value["name"],
            definition=value["definition"],
            workload=value["workload"],
            language=value["language"],
            platform=value["platform"],
        )

    def values(self, role: ValueRole) -> tuple[ABIValue, ...]:
        """Resolve the ordered Trace signature for this workload."""

        if not isinstance(role, ValueRole):
            raise TypeError("role must be an ValueRole")
        specs = mapping(self.definition[role], f"definition {role}")
        axes = mapping(self.definition["axes"], "definition axes")
        bindings = mapping(self.workload["axes"], "workload axes")
        values = []
        for name, raw in specs.items():
            spec = mapping(raw, f"definition {role}.{name}")
            shape = spec["shape"]
            if shape is not None and not isinstance(shape, list):
                raise TypeError(
                    f"definition {role}.{name}.shape must be an array or null"
                )
            values.append(
                ABIValue(
                    name=name,
                    dtype=spec["dtype"],
                    shape=tuple(
                        axis_size(axis, axes, bindings) for axis in shape or ()
                    ),
                    description=spec.get("description") or "",
                )
            )
        return tuple(values)


def axis_size(axis: str, axes: Mapping[str, Any], bindings: Mapping[str, Any]) -> int:
    axis = nonempty(axis, "tensor axis")
    if axis not in axes:
        raise ValueError(f"undefined tensor axis {axis!r}")
    spec = mapping(axes[axis], f"axis {axis}")
    kind = spec["type"]
    if kind == "const":
        size = spec["value"]
    elif kind == "var":
        if axis not in bindings:
            raise ValueError(f"workload axis {axis!r} is missing")
        size = bindings[axis]
    else:
        raise ValueError(f"axis {axis!r} must be const or var")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"axis {axis!r} must resolve to a nonnegative integer")
    if kind == "const" and axis in bindings and bindings[axis] != size:
        raise ValueError(f"workload overrides constant axis {axis!r}")
    return size


@dataclass(frozen=True, slots=True)
class ABIValue:
    """One named value crossing the evaluator-to-submission call boundary."""

    name: str
    dtype: str | None = None
    shape: tuple[int | str, ...] = ()
    description: str = ""

    def __post_init__(self):
        object.__setattr__(self, "name", nonempty(self.name, "ABI value name"))
        if self.dtype is not None:
            object.__setattr__(self, "dtype", nonempty(self.dtype, "ABI value dtype"))
        if isinstance(self.shape, (str, bytes)):
            raise TypeError("ABI value shape must be a sequence of dimensions")
        shape: list[int | str] = []
        for dimension in self.shape:
            if isinstance(dimension, bool) or not isinstance(dimension, (int, str)):
                raise TypeError("ABI dimensions must be integers or symbolic strings")
            if isinstance(dimension, str):
                dimension = nonempty(dimension, "ABI symbolic dimension")
            shape.append(dimension)
        object.__setattr__(self, "shape", tuple(shape))
        object.__setattr__(
            self,
            "description",
            description(self.description, "ABI value description"),
        )


class OutputStyle(StrEnum):
    RETURN = "return"
    DESTINATION = "destination"


__all__ = [
    "BUNDLE_CONFIG",
    "BUNDLE_SOLUTION",
    "ABIValue",
    "OutputStyle",
    "ProblemSpec",
    "ValueRole",
]

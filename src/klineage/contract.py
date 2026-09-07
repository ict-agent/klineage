"""Keep I/O ABI separate from artifact config; legacy loading remains default."""

from __future__ import annotations

import json
import keyword
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any

from klineage._utils import mapping, nonempty

PYTHON_CALLABLE_PROTOCOL = "python-callable-v1"
CUDA_CONFIG = "config.toml"
CUDA_SOLUTION = "solution"
_ENTRY_SUFFIXES = {"cuda": (".cu", ".py"), "python": (".py",)}
RAW_CUDA_ABI = (
    'extern "C" cudaError_t klineage_launch('
    "const void* const* inputs, void* const* outputs, cudaStream_t stream)"
)


def _json_value(value: Any, label: str) -> Any:
    """Return a detached JSON value and reject lossy/non-portable values."""

    def check(item: Any, location: str) -> None:
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


def _json_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    normalized = _json_value(mapping(value, label), label)
    assert isinstance(normalized, dict)
    return normalized


def _description(value: str, label: str) -> str:
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
    """The complete, evaluator-independent statement of one kernel problem."""

    name: str
    statement: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", nonempty(self.name, "problem name"))
        object.__setattr__(
            self,
            "statement",
            nonempty(self.statement, "problem statement"),
        )
        object.__setattr__(
            self,
            "parameters",
            _json_mapping(self.parameters, "problem parameters"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "statement": self.statement,
            "parameters": _json_mapping(self.parameters, "problem parameters"),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProblemSpec:
        value = mapping(value, "problem spec")
        return cls(
            name=value["name"],
            statement=value["statement"],
            parameters=mapping(value.get("parameters", {}), "problem parameters"),
        )


@dataclass(frozen=True, slots=True)
class ABIValue:
    """One named value crossing the evaluator-to-submission call boundary."""

    name: str
    dtype: str | None = None
    shape: tuple[int | str, ...] = ()
    description: str = ""
    constraints: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
            _description(self.description, "ABI value description"),
        )
        object.__setattr__(
            self,
            "constraints",
            _json_mapping(self.constraints, "ABI value constraints"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "description": self.description,
            "constraints": _json_mapping(self.constraints, "ABI value constraints"),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ABIValue:
        value = mapping(value, "ABI value")
        raw_shape = value.get("shape", ())
        if not isinstance(raw_shape, Sequence) or isinstance(raw_shape, (str, bytes)):
            raise TypeError("ABI value shape must be an array")
        dtype = value.get("dtype")
        return cls(
            name=value["name"],
            dtype=dtype,
            shape=tuple(raw_shape),
            description=value.get("description", ""),
            constraints=mapping(
                value.get("constraints", {}),
                "ABI value constraints",
            ),
        )


@dataclass(frozen=True, slots=True)
class EvaluatorInterface:
    """A versioned loader entry point understood by external evaluators.

    For ``python-callable-v1``, the evaluator imports ``module`` from the
    submission root, calls the zero-argument function named by ``loader``, and
    requires that result to be callable.  It then invokes that callable with
    the ABI inputs, in order, and interprets its return value as the ABI
    outputs, in order.
    """

    module: str = "submission.py"
    loader: str = "load"

    def __post_init__(self) -> None:
        module = relative_source_path(self.module, "evaluator module")
        if PurePosixPath(module).suffix != ".py":
            raise ValueError("evaluator module must be a Python source file")
        object.__setattr__(self, "module", module)
        loader = nonempty(self.loader, "evaluator loader symbol")
        if not loader.isidentifier() or keyword.iskeyword(loader):
            raise ValueError("evaluator loader symbol must be a Python identifier")
        object.__setattr__(self, "loader", loader)

    def to_dict(self) -> dict[str, str]:
        return {
            "protocol": PYTHON_CALLABLE_PROTOCOL,
            "module": self.module,
            "loader": self.loader,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvaluatorInterface:
        value = mapping(value, "evaluator interface")
        protocol = value["protocol"]
        if protocol != PYTHON_CALLABLE_PROTOCOL:
            raise ValueError(
                f"evaluator interface protocol must be {PYTHON_CALLABLE_PROTOCOL!r}"
            )
        return cls(
            module=value.get("module", "submission.py"),
            loader=value.get("loader", "load"),
        )


class OutputStyle(StrEnum):
    RETURN = "return"
    DESTINATION = "destination"


@dataclass(frozen=True, slots=True)
class CudaBuild:
    """Build entry and output convention, separate from the kernel I/O ABI."""

    language: str
    entry_point: str
    output_style: OutputStyle = OutputStyle.RETURN

    def __post_init__(self) -> None:
        if self.language not in _ENTRY_SUFFIXES:
            raise ValueError("build language must be cuda or python")
        if not isinstance(self.output_style, OutputStyle):
            raise TypeError("output_style must be an OutputStyle")
        entry = nonempty(self.entry_point, "build entry_point")
        parts = entry.split("::")
        if entry != self.entry_point or len(parts) != 2:
            raise ValueError("entry_point must be relative/source.cu::symbol or .py::symbol")
        path, symbol = parts
        relative_source_path(path, "build entry source")
        if PurePosixPath(path).suffix not in _ENTRY_SUFFIXES[self.language]:
            raise ValueError("entry source suffix is incompatible with build language")
        if not symbol.isidentifier() or keyword.iskeyword(symbol):
            raise ValueError("entry symbol must be an identifier")

    @property
    def source_path(self) -> str:
        return f"{CUDA_SOLUTION}/{self.entry_point.split('::')[0]}"

    @property
    def symbol(self) -> str:
        return self.entry_point.split("::")[1]

    @classmethod
    def from_sources(cls, sources: Mapping[str, str]) -> CudaBuild:
        config = tomllib.loads(sources[CUDA_CONFIG])
        for name in sources:
            relative_source_path(name)
            if name != CUDA_CONFIG and not name.startswith(f"{CUDA_SOLUTION}/"):
                raise ValueError("CUDA bundles contain only config.toml and solution/ sources")
        solution = mapping(config[CUDA_SOLUTION], "solution metadata")
        for key in ("name", "definition", "author"):
            nonempty(solution[key], f"solution {key}")
        build = mapping(config["build"], "build metadata")
        destination = build.get("destination_passing_style", False)
        if type(destination) is not bool:
            raise TypeError("destination_passing_style must be a boolean")
        result = cls(
            language=build["language"], entry_point=build["entry_point"],
            output_style=OutputStyle.DESTINATION if destination else OutputStyle.RETURN,
        )
        if result.source_path not in sources:
            raise ValueError(f"missing entry source: {result.source_path}")
        nonempty(sources[result.source_path], "CUDA entry source")
        return result


def bundle_build(
    sources: Mapping[str, str], interface: EvaluatorInterface,
) -> CudaBuild | None:
    """Distinguish AKO metadata from a legacy entry's auxiliary config."""

    if CUDA_CONFIG not in sources:
        return None
    config = tomllib.loads(sources[CUDA_CONFIG])
    if interface.module in sources and not ({CUDA_SOLUTION, "build"} & config.keys()):
        return None
    return CudaBuild.from_sources(sources)


def source_entry(sources: Mapping[str, str], interface: EvaluatorInterface) -> str:
    """Resolve the artifact entry without changing its I/O ABI."""

    build = bundle_build(sources, interface)
    return build.source_path if build is not None else interface.module


@dataclass(frozen=True, slots=True)
class KernelABI:
    """Versioned input/output declaration and evaluator loading convention.

    Inputs are positional in declaration order.  A single declared output is
    returned directly; multiple outputs are returned as a tuple in declaration
    order, and an ABI with no outputs returns ``None``.
    """

    inputs: tuple[ABIValue, ...] = ()
    outputs: tuple[ABIValue, ...] = ()
    interface: EvaluatorInterface = field(default_factory=EvaluatorInterface)

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", self._values(self.inputs, "input"))
        object.__setattr__(self, "outputs", self._values(self.outputs, "output"))
        if not isinstance(self.interface, EvaluatorInterface):
            raise TypeError("kernel ABI interface must be an EvaluatorInterface")

    @staticmethod
    def _values(values: Sequence[ABIValue], role: str) -> tuple[ABIValue, ...]:
        if isinstance(values, (str, bytes)):
            raise TypeError(f"kernel ABI {role}s must be a sequence")
        normalized = tuple(values)
        if any(not isinstance(value, ABIValue) for value in normalized):
            raise TypeError(f"kernel ABI {role}s must contain ABIValue objects")
        names = [value.name for value in normalized]
        if len(set(names)) != len(names):
            raise ValueError(f"kernel ABI {role} names must be unique")
        return normalized

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "inputs": [value.to_dict() for value in self.inputs],
            "outputs": [value.to_dict() for value in self.outputs],
            "interface": self.interface.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> KernelABI:
        value = mapping(value, "kernel ABI")
        raw_inputs = value.get("inputs", ())
        raw_outputs = value.get("outputs", ())
        if not isinstance(raw_inputs, Sequence) or isinstance(raw_inputs, (str, bytes)):
            raise TypeError("kernel ABI inputs must be an array")
        if not isinstance(raw_outputs, Sequence) or isinstance(
            raw_outputs, (str, bytes)
        ):
            raise TypeError("kernel ABI outputs must be an array")
        version = value["version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise TypeError("kernel ABI version must be an integer")
        if version != 1:
            raise ValueError("kernel ABI version must be 1")
        return cls(
            inputs=tuple(
                ABIValue.from_dict(mapping(item, "kernel ABI input"))
                for item in raw_inputs
            ),
            outputs=tuple(
                ABIValue.from_dict(mapping(item, "kernel ABI output"))
                for item in raw_outputs
            ),
            interface=EvaluatorInterface.from_dict(
                mapping(value.get("interface", {}), "evaluator interface")
            ),
        )


__all__ = [
    "PYTHON_CALLABLE_PROTOCOL",
    "RAW_CUDA_ABI",
    "CUDA_CONFIG",
    "CUDA_SOLUTION",
    "ABIValue",
    "CudaBuild",
    "EvaluatorInterface",
    "KernelABI",
    "OutputStyle",
    "ProblemSpec",
    "bundle_build",
    "source_entry",
]

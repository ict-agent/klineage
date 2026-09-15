"""Kernel source bundles and their problem contracts."""

from __future__ import annotations

import hashlib
import json
import keyword
import math
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from klineage.backend import get_backend
from klineage.constants import (
    BUILD_DIRECTORY,
    BUNDLE_CONFIG,
    BUNDLE_SOLUTION,
    KERNEL_FILE,
)
from klineage.contract import (
    OutputStyle,
    ProblemSpec,
    relative_source_path,
)
from klineage.tools import agent_function
from klineage.utils import boolean, mapping, nonempty


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Measured validation attached to a kernel."""

    compile_passed: bool
    correctness_passed: bool
    profile_passed: bool
    latency_ms: float | None = None
    reference_latency_ms: float | None = None

    def __post_init__(self):
        for name in ("compile_passed", "correctness_passed", "profile_passed"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if self.profile_passed and self.latency_ms is None:
            raise ValueError("successful timing requires latency_ms")
        for name in ("latency_ms", "reference_latency_ms"):
            latency = getattr(self, name)
            if latency is None:
                continue
            if type(latency) not in (int, float):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def accepted(self) -> bool:
        return self.compile_passed and self.correctness_passed and self.profile_passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "compile_passed": self.compile_passed,
            "correctness_passed": self.correctness_passed,
            "profile_passed": self.profile_passed,
            "latency_ms": self.latency_ms,
            "reference_latency_ms": self.reference_latency_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ValidationResult:
        return cls(
            compile_passed=boolean(value["compile_passed"], "compile_passed"),
            correctness_passed=boolean(
                value["correctness_passed"], "correctness_passed"
            ),
            profile_passed=boolean(value["profile_passed"], "profile_passed"),
            latency_ms=value.get("latency_ms"),
            reference_latency_ms=value.get("reference_latency_ms"),
        )


@dataclass(frozen=True, slots=True)
class Kernel:
    name: str
    problem: ProblemSpec
    source_files: Mapping[str, str] | None = None
    validation: ValidationResult | None = None
    compile_flags: tuple[str, ...] = ()
    language: str | None = field(init=False)
    entry_point: str | None = field(init=False)
    output_style: OutputStyle | None = field(init=False)
    function: Callable[..., Any] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self):
        object.__setattr__(self, "name", nonempty(self.name, "kernel name"))
        if not isinstance(self.problem, ProblemSpec):
            raise TypeError("kernel problem must be a ProblemSpec")
        if self.validation is not None and not isinstance(
            self.validation, ValidationResult
        ):
            raise TypeError("kernel validation must be a ValidationResult")
        object.__setattr__(self, "compile_flags", tuple(self.compile_flags))

        sources = {}
        for name, text in mapping(self.source_files or {}, "kernel source_files").items():
            path = relative_source_path(name, "kernel source path")
            if not isinstance(text, str):
                raise TypeError("kernel source contents must be strings")
            if "\x00" in text:
                raise ValueError("kernel sources must not contain NUL bytes")
            sources[path] = text
        object.__setattr__(self, "source_files", sources)
        self.read_build()

    def read_build(self):
        # A source-free placeholder carries only the problem contract. Its build
        # identity stays unknown until the caller supplies sources.
        if not self.source_files:
            for name in ("language", "entry_point", "output_style"):
                object.__setattr__(self, name, None)
            return

        if BUNDLE_CONFIG not in self.source_files:
            backend = get_backend(self.problem.language, self.problem.platform)
            object.__setattr__(self, "language", self.problem.language)
            object.__setattr__(
                self, "entry_point", f"{backend.raw_source}::klineage_launch"
            )
            object.__setattr__(self, "output_style", OutputStyle.DESTINATION)
            return

        config = tomllib.loads(self.source_files[BUNDLE_CONFIG])
        for name in self.source_files:
            if name != BUNDLE_CONFIG and not name.startswith(f"{BUNDLE_SOLUTION}/"):
                raise ValueError(
                    "bundles contain only config.toml and solution/ sources"
                )
        solution = mapping(config[BUNDLE_SOLUTION], "solution metadata")
        for key in ("name", "definition", "author"):
            nonempty(solution[key], f"solution {key}")
        build = mapping(config["build"], "build metadata")
        language = build["language"]
        suffixes = (".py",)
        if language != "python":
            backend = get_backend(self.problem.language, self.problem.platform)
            if language != backend.language:
                raise ValueError(
                    f"build language must be {backend.language} or python for {backend.kind}"
                )
            suffixes = backend.entry_suffixes
        entry = nonempty(build["entry_point"], "build entry_point")
        parts = entry.split("::")
        if entry != build["entry_point"] or len(parts) != 2:
            raise ValueError("entry_point must be relative/source::symbol")
        path, symbol = parts
        relative_source_path(path, "build entry source")
        if PurePosixPath(path).suffix not in suffixes:
            raise ValueError("entry source suffix is incompatible with build language")
        if not symbol.isidentifier() or keyword.iskeyword(symbol):
            raise ValueError("entry symbol must be an identifier")
        destination = build.get("destination_passing_style", False)
        if type(destination) is not bool:
            raise TypeError("destination_passing_style must be a boolean")
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "entry_point", entry)
        object.__setattr__(
            self,
            "output_style",
            OutputStyle.DESTINATION if destination else OutputStyle.RETURN,
        )
        if self.source_path not in self.source_files:
            raise ValueError(f"missing entry source: {self.source_path}")
        nonempty(self.source_files[self.source_path], "bundle entry source")

    @property
    def source_path(self) -> str:
        path = self.entry_point.split("::")[0]
        return (
            f"{BUNDLE_SOLUTION}/{path}" if BUNDLE_CONFIG in self.source_files else path
        )

    @property
    def symbol(self) -> str:
        return self.entry_point.split("::")[1]

    @classmethod
    @agent_function
    def from_sources(
        cls,
        source_files: Mapping[str, str],
        problem: ProblemSpec,
        *,
        name: str | None = None,
        build_root: Path = BUILD_DIRECTORY,
        include_paths: Sequence[Path] = (),
    ) -> Kernel:
        """Construct and build a callable kernel from its complete sources."""
        kernel = cls(name or problem.name, problem, source_files)
        kernel.build(build_root, include_paths=include_paths)
        return kernel

    @agent_function
    def build(
        self,
        build_root: Path = BUILD_DIRECTORY,
        *,
        include_paths: Sequence[Path] = (),
        strides: Mapping[tuple[str, str], tuple[int, ...]] | None = None,
    ):
        """Build in this process; serialized kernels contain no runtime handles."""
        object.__setattr__(self, "function", None)
        from klineage.artifact.bundle import BundleLoader

        function = BundleLoader(build_root, include_paths=include_paths).build(
            self, strides=strides
        )
        object.__setattr__(self, "function", function)

    def __call__(self, *inputs: Any) -> Any:
        if self.function is None:
            raise RuntimeError("Kernel is not built; use from_sources() or build()")
        return self.function(*inputs)

    @property
    def fingerprint(self) -> str:
        value = {
            "problem": self.problem.to_dict(),
            "definition_io": [
                list(self.problem.definition[role]) for role in ("inputs", "outputs")
            ],
            "source_files": dict(sorted(self.source_files.items())),
        }
        return hashlib.sha256(
            json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize sources, problem, and validation; omit compiled runtime handles."""
        payload = {
            "name": self.name,
            "problem": self.problem.to_dict(),
            "source_files": dict(sorted(self.source_files.items())),
            "validation": self.validation.to_dict() if self.validation else None,
        }
        if self.compile_flags:
            payload["compile_flags"] = list(self.compile_flags)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Kernel:
        """Restore serialized Kernel metadata without building or executing sources."""
        validation = value.get("validation")
        return cls(
            name=value["name"],
            problem=ProblemSpec.from_dict(mapping(value["problem"], "kernel problem")),
            source_files=mapping(value["source_files"], "kernel source_files"),
            compile_flags=tuple(value.get("compile_flags") or ()),
            validation=ValidationResult.from_dict(
                mapping(validation, "kernel validation")
            )
            if validation is not None
            else None,
        )


@agent_function
def save_kernel(kernel: Kernel, workdir: Path) -> None:
    """Write kernel.json in an existing workdir, replacing its prior metadata.

    Serializes the Kernel's sources, problem, and validation; it does not reread
    edited disk sources or compile. Reconstruct the Kernel after source edits.
    """
    path = workdir / KERNEL_FILE
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(kernel.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@agent_function
def load_kernel(workdir: Path) -> Kernel:
    """Restore kernel.json from a directory without compiling or running it.

    Returns a Kernel with embedded sources and problem. Evaluation builds it in
    a worker; call Kernel.build before direct execution in this process.
    """
    return Kernel.from_dict(
        json.loads((workdir / KERNEL_FILE).read_text(encoding="utf-8"))
    )


__all__ = ["Kernel", "ValidationResult", "load_kernel", "save_kernel"]

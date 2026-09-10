"""Kernel source bundles and their problem contracts."""

from __future__ import annotations

import hashlib
import json
import keyword
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from klineage.backend import get_backend
from klineage.constants import BUILD_DIRECTORY, BUNDLE_CONFIG, BUNDLE_SOLUTION
from klineage.contract import (
    OutputStyle,
    ProblemSpec,
    relative_source_path,
)
from klineage.harness.eval import ValidationResult
from klineage.tools import agent_function
from klineage.utils import mapping, nonempty


@dataclass(frozen=True, slots=True)
class Kernel:
    name: str
    problem: ProblemSpec
    source_files: Mapping[str, str]
    validation: ValidationResult | None = None
    language: str = field(init=False)
    entry_point: str = field(init=False)
    output_style: OutputStyle = field(init=False)
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

        sources = {}
        for name, text in mapping(self.source_files, "kernel source_files").items():
            path = relative_source_path(name, "kernel source path")
            if not isinstance(text, str):
                raise TypeError("kernel source contents must be strings")
            if "\x00" in text:
                raise ValueError("kernel sources must not contain NUL bytes")
            sources[path] = text
        if not sources or not any(text.strip() for text in sources.values()):
            raise ValueError("kernel source_files must contain source text")
        object.__setattr__(self, "source_files", sources)
        self.read_build()

    def read_build(self):
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
        from klineage.harness.artifacts import BundleLoader

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
        return {
            "name": self.name,
            "problem": self.problem.to_dict(),
            "source_files": dict(sorted(self.source_files.items())),
            "validation": self.validation.to_dict() if self.validation else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Kernel:
        """Restore serialized Kernel metadata without building or executing sources."""
        validation = value.get("validation")
        return cls(
            name=value["name"],
            problem=ProblemSpec.from_dict(mapping(value["problem"], "kernel problem")),
            source_files=mapping(value["source_files"], "kernel source_files"),
            validation=ValidationResult.from_dict(
                mapping(validation, "kernel validation")
            )
            if validation is not None
            else None,
        )


__all__ = ["Kernel"]

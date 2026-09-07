"""Kernel states and their target execution context."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from klineage._utils import mapping, nonempty, string_tuple
from klineage.contract import KernelABI, ProblemSpec, relative_source_path, source_entry
from klineage.harness.eval import ValidationResult
from klineage.profiling import KernelProfile


@dataclass(frozen=True, slots=True, order=True)
class Feature:
    """A mechanism at a stable semantic locus, e.g. ``gemm.main / mma``."""

    locus: str
    name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "locus", nonempty(self.locus, "feature locus"))
        object.__setattr__(self, "name", nonempty(self.name, "feature name"))

    def to_dict(self) -> dict[str, str]:
        return {"locus": self.locus, "name": self.name}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Feature:
        return cls(locus=value["locus"], name=value["name"])


@dataclass(frozen=True, slots=True)
class TargetContext:
    """The paper's target tuple ``T = (case, language, platform, actions)``."""

    case: str
    language: str
    platform: str
    prior_actions: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "case", nonempty(self.case, "case"))
        object.__setattr__(self, "language", nonempty(self.language, "language"))
        object.__setattr__(self, "platform", nonempty(self.platform, "platform"))
        object.__setattr__(
            self,
            "prior_actions",
            string_tuple(self.prior_actions, "prior action"),
        )
        object.__setattr__(
            self, "capabilities", string_tuple(self.capabilities, "capability"),
        )

    def with_actions(self, actions: Iterable[str]) -> TargetContext:
        return replace(
            self,
            prior_actions=string_tuple(
                (*self.prior_actions, *actions),
                "prior action",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "language": self.language,
            "platform": self.platform,
            "prior_actions": list(self.prior_actions),
            "capabilities": list(self.capabilities),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TargetContext:
        return cls(
            case=str(value["case"]),
            language=str(value["language"]),
            platform=str(value["platform"]),
            prior_actions=tuple(str(item) for item in value.get("prior_actions", ())),
            capabilities=tuple(str(item) for item in value.get("capabilities", ())),
        )


@dataclass(frozen=True, slots=True)
class Kernel:
    """A source-bearing kernel state in a lineage."""

    name: str
    source: str
    context: TargetContext
    artifact_path: Path | None = None
    validation: ValidationResult | None = None
    problem: ProblemSpec | None = None
    abi: KernelABI | None = None
    source_files: Mapping[str, str] | None = None
    features: tuple[Feature, ...] = ()
    profile: KernelProfile | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", nonempty(self.name, "kernel name"))
        features = tuple(self.features)
        if any(not isinstance(item, Feature) for item in features):
            raise TypeError("kernel features must contain Feature objects")
        object.__setattr__(self, "features", tuple(dict.fromkeys(features)))
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("kernel source must be non-empty")
        if self.artifact_path is not None:
            object.__setattr__(
                self,
                "artifact_path",
                Path(self.artifact_path).expanduser().absolute(),
            )
        if self.problem is not None and not isinstance(self.problem, ProblemSpec):
            raise TypeError("kernel problem must be a ProblemSpec")
        if self.abi is not None and not isinstance(self.abi, KernelABI):
            raise TypeError("kernel ABI must be a KernelABI")
        if self.source_files is not None:
            if not isinstance(self.source_files, Mapping):
                raise TypeError("kernel source_files must be a mapping")
            source_files: dict[str, str] = {}
            for raw_path, contents in self.source_files.items():
                if not isinstance(raw_path, str):
                    raise TypeError("kernel source file paths must be strings")
                path = relative_source_path(raw_path, "kernel source file path")
                if not isinstance(contents, str):
                    raise TypeError("kernel source file contents must be strings")
                if "\x00" in contents:
                    raise ValueError("kernel source files must not contain NUL bytes")
                source_files[path] = contents
            entry = source_entry(source_files, self.abi.interface) if self.abi else None
            if entry is not None and entry not in source_files:
                raise ValueError(
                    "kernel source_files do not contain the ABI interface module "
                    f"{entry!r}"
                )
            if entry is not None and source_files[entry] != self.source:
                raise ValueError("kernel source must match its ABI interface module")
            object.__setattr__(self, "source_files", source_files)

        if self.profile is not None:
            if not isinstance(self.profile, KernelProfile):
                raise TypeError("kernel profile must be a KernelProfile")
            # Source, contract, or target edits invalidate captured evidence.
            if not self.profile.matches(self):
                object.__setattr__(self, "profile", None)

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.source.encode())
        if (
            self.problem is not None
            or self.abi is not None
            or self.source_files is not None
        ):
            contract = {
                "problem": self.problem.to_dict() if self.problem else None,
                "abi": self.abi.to_dict() if self.abi else None,
                "source_files": (
                    dict(sorted(self.source_files.items()))
                    if self.source_files is not None
                    else None
                ),
            }
            digest.update(b"\0")
            digest.update(
                json.dumps(
                    contract,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            )
        return digest.hexdigest()

    def with_validation(self, validation: ValidationResult) -> Kernel:
        return replace(self, validation=validation)

    def _prompt_input(self) -> dict[str, Any]:
        """Send the contract and source once, without host bookkeeping."""

        value: dict[str, Any] = {
            "context": self.context.to_dict(),
            "problem": self.problem.to_dict() if self.problem else None,
            "abi": self.abi.to_dict() if self.abi else None,
        }
        if self.source_files is None:
            value["source"] = self.source
        else:
            value["source_files"] = dict(sorted(self.source_files.items()))
        if self.features:
            value["features"] = [item.to_dict() for item in self.features]
        if self.profile is not None:
            value["profile"] = self.profile.to_dict()
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "context": self.context.to_dict(),
            "artifact_path": str(self.artifact_path) if self.artifact_path else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "profile": self.profile.to_dict() if self.profile else None,
            "problem": self.problem.to_dict() if self.problem else None,
            "abi": self.abi.to_dict() if self.abi else None,
            "source": self.source,
            "features": [item.to_dict() for item in self.features],
            "source_files": (
                dict(sorted(self.source_files.items()))
                if self.source_files is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Kernel:
        validation = value.get("validation")
        artifact_path = value.get("artifact_path")
        problem = value.get("problem")
        abi = value.get("abi")
        source_files = value.get("source_files")
        profile = value.get("profile")
        return cls(
            name=str(value["name"]),
            source=str(value["source"]),
            context=TargetContext.from_dict(
                mapping(value["context"], "kernel context")
            ),
            artifact_path=Path(str(artifact_path)) if artifact_path else None,
            validation=(
                ValidationResult.from_dict(mapping(validation, "kernel validation"))
                if validation is not None
                else None
            ),
            problem=(
                ProblemSpec.from_dict(mapping(problem, "kernel problem"))
                if problem is not None
                else None
            ),
            abi=(
                KernelABI.from_dict(mapping(abi, "kernel ABI"))
                if abi is not None
                else None
            ),
            source_files=(
                dict(mapping(source_files, "kernel source_files"))
                if source_files is not None
                else None
            ),
            features=tuple(
                Feature.from_dict(mapping(item, "kernel feature"))
                for item in value.get("features", ())
            ),
            profile=KernelProfile.from_dict(profile) if profile is not None else None,
        )


__all__ = ["Feature", "Kernel", "TargetContext"]

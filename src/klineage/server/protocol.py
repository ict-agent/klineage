"""Bounded source transport and authoritative evaluation records."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from klineage.serialization import raw_text
from klineage.artifact import Artifact, Kernel, NodeKind, ValidationResult
from klineage.artifact.tree import (
    SOURCE_BYTE_LIMIT,
    SOURCE_FILE_LIMIT,
    relative_source_path,
)

DEFAULT_TIMEOUT = 900.0
REQUEST_LIMIT = 2 * SOURCE_BYTE_LIMIT + 1024 * 1024
RESPONSE_LIMIT = 4 * REQUEST_LIMIT
DIAGNOSTIC_LIMIT = 64 * 1024
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ServerError(RuntimeError):
    """A remote request failed before a measurement result was available."""


def identifier(value: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError("invalid problem or job identifier")
    return value


def strict_json(data):
    def object_pairs(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("duplicate JSON field")
            result[name] = value
        return result

    def invalid_constant(value):
        raise ValueError("nonfinite JSON number")

    return json.loads(
        data, object_pairs_hook=object_pairs, parse_constant=invalid_constant
    )


def transport_artifact(value):
    """Require inline metadata/sources before typed deserialization can read files."""
    input_root = "inputs/" if value.get("artifact_type") == "problem" else None

    def visit(node, path=""):
        node = dict(node)
        if node.get("reference") is not None and not (
            input_root is not None and path.startswith(input_root)
        ):
            raise ValueError(f"{path}: transported artifact files must be inline")
        # Examples are descriptions; resolve only the receiving package's spec examples.
        if isinstance(node.get("one_shot"), dict):
            node["one_shot"] = None
        if "children" in node:
            node["children"] = [
                visit(
                    child,
                    path
                    + child["name"]
                    + ("/" if child["kind"] == "directory" else ""),
                )
                for child in node["children"]
            ]
        return node

    return {**value, "tree": visit(value["tree"])}


def deadline_seconds(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("timeout must be a number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("timeout must be finite and positive")
    return float(value)


def check_submission(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "name",
        "kind",
        "sources",
        "compile_flags",
    }:
        raise ValueError("submission requires name, kind, sources and compile_flags")
    for key in ("name", "kind"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"submission.{key} must be nonempty text")
    sources = value["sources"]
    if not isinstance(sources, dict) or not sources:
        raise ValueError("submission.sources must contain source text")
    if len(sources) > SOURCE_FILE_LIMIT:
        raise ValueError("source bundle exceeds its file limit")
    size = 0
    for name, content in sources.items():
        relative_source_path(name)
        if not isinstance(content, str) or "\0" in content:
            raise ValueError(f"{name}: source must be UTF-8 text")
        size += len(content.encode("utf-8"))
    if size > SOURCE_BYTE_LIMIT:
        raise ValueError("source bundle exceeds its byte limit")
    flags = value["compile_flags"]
    if not isinstance(flags, list) or any(
        not isinstance(flag, str) or "\0" in flag for flag in flags
    ):
        raise ValueError("compile_flags must be an array of strings")
    return value


def submission(value: Artifact | Path, kind: str | None = None) -> dict[str, Any]:
    from klineage.artifact import load_artifact

    if not isinstance(value, Artifact):
        path = Path(value)
        if path.is_dir():
            from klineage.artifact.kernel import capture_kernel

            if kind is None:
                raise ValueError("directory submissions require a kernel kind")
            value = Kernel.from_tree(
                kind, capture_kernel(path)
            )
        else:
            value = load_artifact(path, Kernel)
    if not isinstance(value, Kernel):
        raise TypeError("submission must be a Kernel artifact or directory")
    metadata_node = value.find("metadata.json")
    if metadata_node.content is None:
        raise ValueError("metadata.json must contain inline text")
    metadata = json.loads(raw_text(metadata_node))
    sources = {}

    def visit(node, prefix=""):
        for child in node.children:
            name = prefix + child.name
            if child.kind is NodeKind.DIRECTORY:
                visit(child, name + "/")
                continue
            if child.content is None:
                raise ValueError(f"{name}: source FileRefs are not accepted")
            sources[name] = child.content

    visit(value.find("submission"))
    return check_submission(
        {
            "name": metadata["name"],
            "kind": value.kind,
            "sources": sources,
            "compile_flags": metadata.get("compile_flags", []),
        }
    )


def source_digest(problem_id: str, value: dict[str, Any]) -> str:
    payload = {"problem_id": problem_id, **check_submission(value)}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class Evaluation:
    """A measured kernel and the server record identifying its submitted sources.

    Construction checks the source digest, validation consistency and diagnostic
    paths. Use `Client.verify()` to match a local handoff to the server record.

    Attributes:
        job_id: Server-issued measurement identifier.
        problem_id: Registered problem identifier.
        digest: Digest of the problem identifier and submitted kernel sources,
            name, kind and compile flags.
        kernel: Artifact containing measured sources and validation; not a built
            local callable.
        validation: Build, correctness and timing gate results.
        diagnostics: Relative diagnostic paths mapped to text.
        error: Worker or scheduling failure, if one was recorded.
    """

    job_id: str
    problem_id: str
    digest: str
    kernel: Kernel
    validation: ValidationResult
    diagnostics: dict[str, str]
    error: str | None = None

    def __post_init__(self):
        identifier(self.job_id)
        identifier(self.problem_id)
        if self.digest != source_digest(self.problem_id, submission(self.kernel)):
            raise ValueError("result digest differs from its sources")
        if self.kernel.validation != self.validation:
            raise ValueError("result validation differs from its kernel")
        if not isinstance(self.diagnostics, dict):
            raise ValueError("diagnostics must be a text mapping")
        for name, content in self.diagnostics.items():
            relative_source_path(name, "diagnostic path")
            if not isinstance(content, str):
                raise ValueError("diagnostics must contain text")
        if self.error is not None and not isinstance(self.error, str):
            raise ValueError("result error must be text")

    @property
    def accepted(self) -> bool:
        """Whether every gate passed and no job-level error was recorded."""
        return self.error is None and self.validation.accepted

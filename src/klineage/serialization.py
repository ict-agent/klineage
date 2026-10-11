"""Registered JSON and raw-text handlers for artifacts and execution records."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import fields
from functools import lru_cache
from pathlib import Path
from typing import Any

from klineage.utils import mapping


class SerializationHandler(ABC):
    """Define serialization for a registered Python type.

    ``default`` returns a shallow JSON-compatible value. Registered child
    objects may remain in it; the JSON encoder processes them recursively.
    ``restore`` receives parsed data and the requested concrete Python type.
    Override ``raw_text`` or ``raw_bytes`` for text or file-byte representations.
    """

    @abstractmethod
    def default(self, value: Any) -> Any:
        """Return this object's JSON representation without serializing children."""
        raise NotImplementedError

    @abstractmethod
    def restore(self, data: Any, object_type: type) -> Any:
        """Restore and validate parsed data as the requested type."""
        raise NotImplementedError

    def raw_text(self, value: Any) -> str:
        """Return text for this type, or reject unsupported text conversion."""
        raise TypeError(f"no raw-text serialization for {type(value).__name__}")

    def raw_bytes(self, value: Any) -> bytes:
        """Return file bytes, or reject unsupported binary conversion."""
        raise TypeError(f"no byte serialization for {type(value).__name__}")


class DataclassHandler(SerializationHandler):
    def default(self, value):
        return {field.name: getattr(value, field.name) for field in fields(value)}

    def restore(self, data, object_type):
        return object_type(**mapping(data, object_type.__name__))


class FileRefHandler(DataclassHandler):
    def raw_text(self, value):
        return self.raw_bytes(value).decode("utf-8")

    def raw_bytes(self, value):
        path = Path(value.path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"reference must be a regular file: {path}")
        with path.open("rb") as stream:
            data = stream.read(value.size + 1)
        if len(data) != value.size or hashlib.sha256(data).hexdigest() != value.sha256:
            raise ValueError(f"referenced file changed: {path}")
        return data


class ScopeHandler(DataclassHandler):
    def restore(self, data, object_type):
        from klineage.memory.skillcard import string_sequence

        names = ("cases", "languages", "platforms")
        if set(data) != set(names):
            raise ValueError("scope requires cases, languages, and platforms")
        return object_type(
            **{name: string_sequence(data[name], f"scope {name}") for name in names}
        )


class ArtifactHandler(SerializationHandler):
    def raw_text(self, value):
        return raw_text(value.tree)

    def default(self, value):
        return {
            "artifact_type": value.kind,
            "tree": value.tree,
        }

    def restore(self, data, object_type):
        from klineage.artifact.tree import Node

        if set(data) != {"artifact_type", "tree"}:
            raise ValueError("invalid artifact envelope")
        return object_type.from_tree(
            data["artifact_type"], restore(data["tree"], Node)
        )


class NodeHandler(SerializationHandler):
    def raw_bytes(self, value):
        from klineage.artifact.tree import NodeKind

        if value.kind is not NodeKind.FILE:
            raise TypeError("byte serialization requires a file node")
        if value.content is not None:
            return value.content.encode("utf-8")
        if value.reference is not None:
            return raw_bytes(value.reference)
        raise ValueError(f"{value.name}: file requires content or a reference")

    def raw_text(self, value):
        from klineage.artifact.tree import FileRef, NodeKind

        if value.kind is NodeKind.FILE:
            if value.content is not None:
                return value.content
            if value.reference is not None:
                return raw_text(value.reference)

        def render(node, prefix="", branch=""):
            name = node.name + ("/" if node.kind is NodeKind.DIRECTORY else "")
            label = prefix + branch + name
            continuation = prefix
            if branch:
                continuation += "│   " if branch == "├── " else "    "

            description = node.description.rstrip()
            if node.one_shot is not None:
                example = node.one_shot
                if isinstance(example, FileRef):
                    example = example.path
                example = example.replace("\r", "\\r").replace("\n", "\\n")
                description += (" " if description else "") + f"(See: {example} as an example)"

            lines = description.splitlines() or [""]
            separator = " ---> "
            yield label + (separator + lines[0] if lines[0] else "")
            padding = " " * (len(label) + len(separator) - len(continuation))
            for line in lines[1:]:
                yield continuation + padding + line

            for index, child in enumerate(node.children):
                branch = "└── " if index == len(node.children) - 1 else "├── "
                yield from render(child, continuation, branch)

        return "\n".join(render(value))

    def default(self, value):
        from klineage.artifact.tree import NodeKind

        result = {
            "name": value.name,
            "kind": value.kind.value,
            "description": value.description,
            "one_shot": value.one_shot,
        }
        if value.artifact_type is not None:
            result["artifact_type"] = value.artifact_type
        if value.kind is NodeKind.DIRECTORY:
            result["children"] = value.children
        elif value.reference is not None:
            result["reference"] = value.reference
        elif value.content is not None:
            result["content"] = value.content
        return result

    def restore(self, data, object_type):
        from klineage.artifact.tree import FileRef, NodeKind

        allowed = {
            "name",
            "kind",
            "description",
            "one_shot",
            "artifact_type",
            "children",
            "content",
            "reference",
        }
        if set(data) - allowed:
            raise ValueError("unknown tree node fields")
        reference, example = data.get("reference"), data.get("one_shot")
        return object_type(
            data["name"],
            NodeKind(data["kind"]),
            data.get("description", ""),
            restore(example, FileRef) if isinstance(example, dict) else example,
            tuple(restore(child, object_type) for child in data.get("children", ())),
            data.get("content"),
            restore(reference, FileRef) if reference is not None else None,
            data.get("artifact_type"),
        )


class ValidationHandler(SerializationHandler):
    def default(self, value):
        result = {
            "compile_passed": value.compile_passed,
            "correctness_passed": value.correctness_passed,
            "profile_passed": value.profile_passed,
            "latency_ms": value.latency_ms,
            "reference_latency_ms": value.reference_latency_ms,
        }
        if value.workloads:
            result["workloads"] = dict(value.workloads)
        return result

    def restore(self, data, object_type):
        workloads = data.get("workloads", {})
        if not isinstance(workloads, Mapping):
            raise TypeError("workloads must map UUIDs to validation results")
        return object_type(
            compile_passed=data["compile_passed"],
            correctness_passed=data["correctness_passed"],
            profile_passed=data["profile_passed"],
            latency_ms=data.get("latency_ms"),
            reference_latency_ms=data.get("reference_latency_ms"),
            workloads={
                name: restore(value, object_type) for name, value in workloads.items()
            },
        )


class EvaluationHandler(SerializationHandler):
    def default(self, value):
        return {
            "job_id": value.job_id,
            "problem_id": value.problem_id,
            "digest": value.digest,
            "kernel": value.kernel,
            "validation": value.validation,
            "diagnostics": value.diagnostics,
            "error": value.error,
        }

    def restore(self, data, object_type):
        from klineage.artifact.kernel import Kernel, ValidationResult
        from klineage.server.protocol import transport_artifact

        return object_type(
            data["job_id"],
            data["problem_id"],
            data["digest"],
            restore(transport_artifact(data["kernel"]), Kernel),
            restore(data["validation"], ValidationResult),
            data["diagnostics"],
            data.get("error"),
        )


@lru_cache(maxsize=1)
def handlers() -> dict[type, SerializationHandler]:
    """Initialize built-in handlers without importing vendor runtimes."""
    from klineage.artifact.kernel import ValidationResult
    from klineage.artifact.tree import Artifact, FileRef, Node
    from klineage.backend.timing import TimingPolicy, TimingResult
    from klineage.memory.skillcard import Scope
    from klineage.server.protocol import Evaluation

    dataclass_handler = DataclassHandler()
    return {
        FileRef: FileRefHandler(),
        Scope: ScopeHandler(),
        TimingPolicy: dataclass_handler,
        TimingResult: dataclass_handler,
        Artifact: ArtifactHandler(),
        Node: NodeHandler(),
        ValidationResult: ValidationHandler(),
        Evaluation: EvaluationHandler(),
    }


def register_handler(object_type: type, handler: SerializationHandler) -> None:
    """Register a process-local handler; reject duplicate types.

    Args:
        object_type: Type whose representations the handler owns.
        handler: Implements ``default`` and ``restore``; optionally ``raw_text``
            and ``raw_bytes``.
    """
    if not isinstance(object_type, type) or not isinstance(
        handler, SerializationHandler
    ):
        raise TypeError("registration requires a type and SerializationHandler")
    registered = handlers()
    if object_type in registered:
        raise ValueError(
            f"serialization handler already registered: {object_type.__name__}"
        )
    registered[object_type] = handler


def handler_for(object_type: type) -> SerializationHandler:
    """Select the nearest registered class in the requested type's MRO."""
    if not isinstance(object_type, type):
        raise TypeError("restoration requires a Python type")
    registered = handlers()
    for base in object_type.__mro__:
        if base in registered:
            return registered[base]
    raise TypeError(f"no serialization handler for {object_type.__name__}")


def to_json(value: Any, *, indent: int | None = 2) -> str:
    """Encode a value and registered nested objects in one JSON traversal.

    Args:
        value: JSON data, registered objects, or containers combining them.
        indent: Indentation width, or None for a single-line representation.
    """
    return json.dumps(
        value,
        default=lambda item: handler_for(type(item)).default(item),
        ensure_ascii=False,
        allow_nan=False,
        indent=indent,
    )


def raw_text(value: Any) -> str:
    """Serialize text through the value's registered handler.

    Strings pass through unchanged. File nodes return their contents; FileRef
    reads verify the file's type, size and hash before decoding UTF-8. Artifacts,
    directories and files without payload render annotated file trees. Example
    references appear as paths without being read; inline examples escape line breaks.
    JSON serialization preserves references without reading them.

    Raises:
        TypeError: The value has no supported raw-text representation.
        ValueError: A referenced file is missing, a symlink, or its pin changed.
        UnicodeDecodeError: Referenced content is not UTF-8.
    """
    if isinstance(value, str):
        return value
    return handler_for(type(value)).raw_text(value)


def raw_bytes(value: Any) -> bytes:
    """Return file bytes through a handler, checking FileRef size and digest.

    File nodes encode inline text as UTF-8 or read their pinned references.
    Directories and files without payload cannot be converted to file bytes.
    """
    return handler_for(type(value)).raw_bytes(value)


def restore[T](data: Any, object_type: type[T]) -> T:
    """Restore already-parsed data using the requested type's registered handler."""
    return handler_for(object_type).restore(data, object_type)


def from_json[T](text: str, object_type: type[T]) -> T:
    """Parse JSON once and restore the requested type through its handler."""
    return restore(json.loads(text), object_type)


__all__ = [
    "SerializationHandler",
    "register_handler",
    "to_json",
    "raw_text",
    "raw_bytes",
    "from_json",
    "restore",
]

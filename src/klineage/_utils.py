"""Small validation and identity helpers shared by klineage modules."""

from __future__ import annotations

import re
import secrets
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_WORKSPACE = "agent-workspace"


def nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def string_tuple(values: Iterable[str], label: str) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values,)
    result: list[str] = []
    for value in values:
        normalized = nonempty(value, label)
        if normalized not in result:
            result.append(normalized)
    return tuple(result)


def mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a bool")
    return value


def safe_name(value: str, *, default: str) -> str:
    normalized = _SAFE_NAME_RE.sub("-", value).strip(".-_")
    return (normalized or default)[:80]


def operation_id(purpose: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{safe_name(purpose, default='action')}-{timestamp}-{secrets.token_hex(3)}"


def new_workdir(name: str) -> Path:
    return Path.cwd() / _WORKSPACE / operation_id(name)


__all__ = [
    "boolean",
    "mapping",
    "new_workdir",
    "nonempty",
    "operation_id",
    "safe_name",
    "string_tuple",
]

"""Shared validation, identity, path, and JSON helpers."""

from __future__ import annotations

import json
import os
import re
import secrets
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from klineage.constants import WORKSPACE_DIR

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


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
    return Path.cwd() / WORKSPACE_DIR / operation_id(name)


def optional_directory(value: str | os.PathLike[str] | None) -> Path | None:
    """Return None for absent/blank input; otherwise require an existing directory."""

    if value is None or isinstance(value, str) and not value.strip():
        return None
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise NotADirectoryError(f"not a directory: {path}")
    return path


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=True)


def make_log_dir(path: Path) -> Path:
    directory = Path(path).expanduser().absolute()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return directory


def write_json(path: Path, value: Any):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def existing_dir(value: Any, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{label} must be a path string")
    path = resolve_path(value)
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")
    return path


__all__ = [
    "boolean",
    "existing_dir",
    "make_log_dir",
    "mapping",
    "new_workdir",
    "nonempty",
    "operation_id",
    "optional_directory",
    "resolve_path",
    "safe_name",
    "string_tuple",
    "write_json",
]

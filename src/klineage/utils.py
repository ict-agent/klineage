"""Shared validation, identity, path, and JSON helpers."""

from __future__ import annotations

import os
import re
import secrets
import stat
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from klineage.constants import WORKSPACE_DIR

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class DirectoryMode(Enum):
    EXISTING = "existing"
    CREATE = "create"


@contextmanager
def safe_directory(path: Path, mode=DirectoryMode.EXISTING):
    """Open each directory without following links; retain the final directory fd."""
    path = Path(path).expanduser().absolute()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            if part in (".", ".."):
                raise ValueError("directory path must not contain traversal")
            if mode is DirectoryMode.CREATE:
                with suppress(FileExistsError):
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def safe_file(path: Path, mode="r", *, errors="strict"):
    """Open a regular, single-link file beneath no-follow directory descriptors."""
    flags = {
        "r": os.O_RDONLY,
        "w": os.O_WRONLY | os.O_CREAT,
        "w+": os.O_RDWR | os.O_CREAT,
        "a": os.O_WRONLY | os.O_CREAT | os.O_APPEND,
    }[mode] | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    path = Path(path)
    directory_mode = DirectoryMode.EXISTING if mode == "r" else DirectoryMode.CREATE
    with safe_directory(path.parent, directory_mode) as parent:
        descriptor = os.open(path.name, flags, 0o600, dir_fd=parent)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise OSError(f"expected a regular single-link file: {path}")
        # Validate hard links before truncation, unlike open(..., O_TRUNC).
        if mode in ("w", "w+"):
            os.ftruncate(descriptor, 0)
        stream = os.fdopen(descriptor, mode, encoding="utf-8", errors=errors, newline="")
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        yield stream


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


def json_value(value: Any, label: str) -> Any:
    """Return a detached JSON value and reject lossy/non-portable values."""

    def normalize(item: Any, location: str):
        if item is None or isinstance(item, bool):
            return item
        if isinstance(item, str):
            return str(item)
        if isinstance(item, int):
            return int(item)
        if isinstance(item, float):
            if not (-float("inf") < item < float("inf")):
                raise ValueError(f"{location} must not contain NaN or infinity")
            return float(item)
        if isinstance(item, Mapping):
            result = {}
            for key, child in item.items():
                if not isinstance(key, str):
                    raise TypeError(f"{location} keys must be strings")
                result[str(key)] = normalize(child, f"{location}.{key}")
            return result
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            return [
                normalize(child, f"{location}[{index}]")
                for index, child in enumerate(item)
            ]
        raise TypeError(f"{location} must contain only JSON values")

    return normalize(value, label)


def json_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    normalized = json_value(mapping(value, label), label)
    assert isinstance(normalized, dict)
    return normalized


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
    from klineage.serialization import to_json

    path.write_text(
        to_json(value) + "\n",
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
    "existing_dir",
    "make_log_dir",
    "json_value",
    "json_mapping",
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

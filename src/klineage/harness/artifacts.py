"""Safe paths and source checks for artifacts written by coding agents."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from klineage._utils import nonempty, signature
from klineage.contract import EvaluatorInterface, relative_source_path
from klineage.errors import StructuredOutputError, ValidationGateError

_COMMENT_RE = re.compile(r"//[^\n]*|/\*[\s\S]*?\*/")
_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+(.+)$', re.MULTILINE)
_LIBRARY_RE = re.compile(r"\b(?:cutlass|cute)\s*(?:::|/)|\bcublas\w*", re.IGNORECASE)
_CUDA_TOKEN_RE = re.compile(r"\b(?:__global__|__device__)\b")
_NON_CUDA_MARKERS = ("@triton.jit", "import torch", "from torch", "def forward(")
_MAX_SOURCE_BYTES = 16 * 1024 * 1024
_MAX_SOURCE_FILES = 1024


def read_generated_source(
    expected_path: Path,
    label: str,
    *,
    allowed_root: Path,
) -> str:
    """Read an agent artifact without following replaced paths or symlinks."""

    root = allowed_root.absolute()
    path = expected_path.absolute()
    if path.parent != root:
        raise StructuredOutputError(f"{label} path is outside its action directory")
    if root.is_symlink() or root.resolve(strict=True) != root:
        raise StructuredOutputError(f"{label} action directory was replaced")
    if path.is_symlink():
        raise StructuredOutputError(f"{label} cannot be a symbolic link")

    if not path.exists():
        raise StructuredOutputError(f"agent did not write {label}")
    if not path.is_file() or not path.resolve(strict=True).is_relative_to(root):
        raise StructuredOutputError(f"{label} is not a regular action file")
    if path.stat().st_size > _MAX_SOURCE_BYTES:
        raise StructuredOutputError(f"{label} exceeds the 16 MiB limit")
    source = path.read_text(encoding="utf-8")
    if "\x00" in source:
        raise StructuredOutputError(f"{label} contains a NUL byte")
    return nonempty(source, label)


def _read_source_tree(directory: Path, label: str) -> dict[str, str]:
    files: dict[str, str] = {}
    total_bytes = 0
    pending = [directory]
    while pending:
        parent = pending.pop()
        for path in sorted(parent.iterdir(), key=lambda item: item.name):
            if path.is_symlink():
                raise StructuredOutputError(f"{label} cannot contain symbolic links")
            if path.is_dir():
                pending.append(path)
                continue
            if not path.is_file():
                raise StructuredOutputError(
                    f"{label} can contain only regular files and directories"
                )
            if len(files) >= _MAX_SOURCE_FILES:
                raise StructuredOutputError(
                    f"{label} contains more than {_MAX_SOURCE_FILES} files"
                )
            try:
                size = path.stat(follow_symlinks=False).st_size
            except OSError as error:
                raise StructuredOutputError(f"cannot stat {label} file") from error
            if total_bytes + size > _MAX_SOURCE_BYTES:
                raise StructuredOutputError(f"{label} exceeds the 16 MiB total limit")
            try:
                relative = path.relative_to(directory).as_posix()
            except ValueError as error:
                raise StructuredOutputError(
                    f"{label} file is outside its submission directory"
                ) from error
            try:
                resolved = path.resolve(strict=True)
            except OSError as error:
                raise StructuredOutputError(f"cannot resolve {label} file") from error
            if not resolved.is_relative_to(directory):
                raise StructuredOutputError(
                    f"{label} file is outside its submission directory"
                )
            try:
                portable_path = relative_source_path(relative, f"{label} path")
                source = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as error:
                raise StructuredOutputError(
                    f"{label} file {relative!r} is not UTF-8 text"
                ) from error
            except (TypeError, ValueError) as error:
                raise StructuredOutputError(str(error)) from error
            if "\x00" in source:
                raise StructuredOutputError(
                    f"{label} file {relative!r} contains a NUL byte"
                )
            total_bytes += len(source.encode("utf-8"))
            if total_bytes > _MAX_SOURCE_BYTES:
                raise StructuredOutputError(f"{label} exceeds the 16 MiB total limit")
            files[portable_path] = source
    return dict(sorted(files.items()))


def read_generated_source_bundle(
    expected_directory: Path,
    label: str,
    *,
    allowed_root: Path,
    interface: EvaluatorInterface,
) -> dict[str, str]:
    """Read a complete, path-safe generated source tree."""

    root = allowed_root.absolute()
    directory = expected_directory.absolute()
    if directory.parent != root:
        raise StructuredOutputError(f"{label} path is outside its action directory")
    if root.is_symlink() or root.resolve(strict=True) != root:
        raise StructuredOutputError(f"{label} action directory was replaced")
    if directory.is_symlink():
        raise StructuredOutputError(f"{label} cannot be a symbolic link")
    if not directory.exists():
        raise StructuredOutputError(f"agent did not write {label}")
    if not directory.is_dir() or directory.resolve(strict=True) != directory:
        raise StructuredOutputError(f"{label} is not a regular action directory")

    source_files = _read_source_tree(directory, label)
    if not source_files:
        raise StructuredOutputError(f"agent did not write {label}")

    if interface.module not in source_files:
        raise StructuredOutputError(
            f"{label} does not contain ABI interface module {interface.module!r}"
        )
    try:
        nonempty(source_files[interface.module], "ABI interface module")
    except ValueError as error:
        raise StructuredOutputError(str(error)) from error
    return source_files


def require_pure_cuda(source: str, *, repository: Path | None = None) -> None:
    """Reject obvious framework delegation and non-CUDA output."""

    code = _COMMENT_RE.sub("", source)
    lowered = code.lower()
    _require_standalone(code, repository)

    if not _CUDA_TOKEN_RE.search(code):
        raise ValidationGateError(
            "generated source does not contain a CUDA __global__ or __device__ kernel"
        )
    marker = next((item for item in _NON_CUDA_MARKERS if item in lowered), None)
    if marker is not None:
        raise ValidationGateError(
            f"generated source contains a non-CUDA fallback marker: {marker}"
        )


def require_cuda_source_bundle(
    source_files: Mapping[str, str], *, repository: Path | None = None,
) -> None:
    """Require CUDA implementation code while allowing Python loader glue."""

    sources = tuple(source_files.values())
    for source in sources:
        _require_standalone(_COMMENT_RE.sub("", source), repository)
    if not any(_CUDA_TOKEN_RE.search(source) for source in sources):
        raise ValidationGateError(
            "generated source bundle does not contain a CUDA __global__ or "
            "__device__ kernel"
        )
    if any("@triton.jit" in source.lower() for source in sources):
        raise ValidationGateError(
            "generated source bundle contains a non-CUDA fallback marker: @triton.jit"
        )


def _require_standalone(code: str, repository: Path | None) -> None:
    if _LIBRARY_RE.search(code):
        raise ValidationGateError("raw CUDA cannot delegate to an expert library")
    roots = () if repository is None else (
        repository, repository / "include", repository / "tools/util/include",
    )
    for include in _INCLUDE_RE.findall(code):
        match = re.fullmatch(r'[<"]([^>"]+)[>"]\s*', include)
        if match is None:
            raise ValidationGateError("raw CUDA requires literal include paths")
        name = Path(match[1])
        if (
            name.is_absolute() or ".." in name.parts
            or any((root / name).is_file() for root in roots)
        ):
            raise ValidationGateError("raw CUDA cannot include repository implementation files")


def is_cuda_language(language: str) -> bool:
    normalized = signature(language)
    return "cuda" in normalized or normalized in {"cu", "cuda c", "cuda c++"}


__all__ = [
    "is_cuda_language",
    "read_generated_source",
    "read_generated_source_bundle",
    "require_cuda_source_bundle",
    "require_pure_cuda",
]

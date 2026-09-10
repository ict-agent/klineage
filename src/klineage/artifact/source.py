"""Read and freeze complete source bundles."""

from __future__ import annotations

import errno
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory

from klineage.contract import relative_source_path
from klineage.errors import StructuredOutputError
from klineage.tools import agent_function

#: Maximum combined source size accepted by the bundle reader.
_MAX_SOURCE_BYTES = 16 * 1024 * 1024
#: Maximum file count accepted by the bundle reader.
_MAX_SOURCE_FILES = 1024


@agent_function
def read_source_tree(directory: Path, label: str) -> dict[str, str]:
    """Read a complete source directory into relative-path → exact-text entries.

    Reject symlinks, binary files, and escaping paths. Use label to identify the
    bundle in errors. Pass the returned map to Kernel.from_sources.
    """
    directory = directory.expanduser().absolute()
    if directory.is_symlink():
        raise StructuredOutputError(f"{label} cannot be a symbolic link")
    directory = directory.resolve(strict=True)
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
                source = path.read_bytes().decode("utf-8")
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


def snapshot(work: Path, sources: Mapping[str, str]) -> str:
    sources = {
        relative_source_path(name, "snapshot source"): source
        for name, source in sources.items()
    }
    digest = hashlib.sha256(
        json.dumps(sources, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    directory = work / f"source-{digest}"
    if not directory.exists():
        work.mkdir(parents=True, exist_ok=True)
        # Publish complete sources atomically; concurrent builds reuse the winner.
        with TemporaryDirectory(prefix=".source-", dir=work) as temporary:
            staging = Path(temporary)
            for name, source in sources.items():
                path = staging / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source.encode("utf-8"))
            try:
                staging.rename(directory)
            except OSError as error:
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise

    if read_source_tree(directory, "source snapshot") != sources:
        raise StructuredOutputError(f"source snapshot was modified: {directory}")
    return str(directory)

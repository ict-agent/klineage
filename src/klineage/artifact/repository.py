"""Prepare source repositories for an action."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from klineage.errors import ActionError
from klineage.tools import agent_function


@agent_function
def stage_repository(
    repo: str | os.PathLike[str],
    destination: Path,
) -> Path:
    """Copy a local repository or shallow-clone a Git URL into a fresh destination.

    Returns its resolved path. The destination must not exist; inspect and reuse
    an intact staged repository on retries instead of calling this again.
    """

    raw = os.fspath(repo)
    local = Path(raw).expanduser()
    if local.exists():
        source = local.resolve(strict=True)
        if not source.is_dir():
            raise NotADirectoryError(source)
        shutil.copytree(
            source,
            destination,
            symlinks=True,
            ignore=repository_ignore(destination),
            ignore_dangling_symlinks=True,
        )
        return destination.resolve(strict=True)

    if not looks_like_git_url(raw):
        raise FileNotFoundError(f"repository does not exist: {raw}")
    git = shutil.which("git")
    if git is None:
        raise ActionError("git is required to stage a repository URL")
    completed = subprocess.run(
        (git, "clone", "--depth", "1", "--", raw, str(destination)),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise ActionError(
            f"could not clone repository {raw!r}: {completed.stdout[-2000:]}"
        )
    return destination.resolve(strict=True)


def repository_ignore(destination: Path):
    destination = destination.expanduser().resolve()
    ignored_names = {".git", ".hg", ".svn", ".venv", "__pycache__", ".klineage"}

    def ignore(directory: str, names: list[str]) -> set[str]:
        base = Path(directory).resolve(strict=True)
        ignored = {name for name in names if name in ignored_names}
        for name in names:
            child = (base / name).resolve(strict=False)
            if (
                destination == child
                or destination.is_relative_to(child)
                or child.is_relative_to(destination)
            ):
                ignored.add(name)
        return ignored

    return ignore


def looks_like_git_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https", "ssh", "git"} or value.startswith("git@")


__all__ = ["stage_repository"]

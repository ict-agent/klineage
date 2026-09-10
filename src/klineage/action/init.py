"""Initialize an expert kernel from a problem and repository."""

from __future__ import annotations

import os
from pathlib import Path

from klineage.action.action import Action
from klineage.artifact.repository import looks_like_git_url
from klineage.constants import MAX_RETRIES, TIMEOUT, RunKind
from klineage.prompts import render_prompt
from klineage.utils import new_workdir


def init(
    problem: str | os.PathLike[str],
    repo: str | os.PathLike[str],
    expert_kernel: str | os.PathLike[str],
):
    action = Init(problem, repo, expert_kernel)
    action.run()


class Init(Action):
    verify_prompt = render_prompt(f"{RunKind.VERIFY}_{RunKind.INIT}")

    def __init__(
        self,
        problem: str | os.PathLike[str],
        repo: str | os.PathLike[str],
        expert_kernel: str | os.PathLike[str],
        *,
        workdir: Path | None = None,
        enable_verifier: bool = True,
        max_retries: int = MAX_RETRIES,
        timeout: int = TIMEOUT,
    ):

        workdir = Path(workdir or new_workdir(RunKind.INIT)).expanduser().resolve()
        problem_path = Path(problem).expanduser().resolve(strict=True)
        repository = (
            str(repo)
            if looks_like_git_url(str(repo))
            else str(Path(repo).expanduser().resolve())
        )
        prompt = render_prompt(
            RunKind.INIT,
            problem=str(problem_path),
            repository=repository,
            expert_kernel=str(expert_kernel),
        )
        super().__init__(
            prompt,
            workdir,
            enable_verifier,
            timeout,
            max_retries=max_retries,
        )


__all__ = ["Init", "init"]

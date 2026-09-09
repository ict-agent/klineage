"""Capture CUDA metrics and interpret them through the shared runner."""

from dataclasses import asdict
from pathlib import Path

from klineage._utils import new_workdir
from klineage.action.action import MAX_RETRIES, TIMEOUT, Action
from klineage.kernel import Kernel
from klineage.profiling import ProfileOptions
from klineage.prompts import render_prompt

DEFAULT_OPTIONS = ProfileOptions()


def profile(kernel: Kernel | Path, *, options: ProfileOptions = DEFAULT_OPTIONS):
    action = Profile(kernel, options=options)
    action.run()


class Profile(Action):
    verify_prompt = render_prompt("verify_profile")

    def __init__(
        self,
        kernel: Kernel | Path,
        *,
        options: ProfileOptions = DEFAULT_OPTIONS,
        workdir: Path | None = None,
        enable_verifier: bool = True,
        max_retries: int = MAX_RETRIES,
        timeout: int = TIMEOUT,
    ):
        workdir = workdir or new_workdir("profile")
        prompt = render_prompt(
            "profile",
            kernel=(
                kernel.to_dict()
                if isinstance(kernel, Kernel)
                else str(Path(kernel).expanduser().resolve())
            ),
            options=asdict(options),
        )
        super().__init__(
            prompt,
            workdir,
            enable_verifier,
            timeout,
            max_retries=max_retries,
        )


__all__ = ["Profile", "profile"]

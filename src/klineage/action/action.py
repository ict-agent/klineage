"""Action execution, LLM verification, and bounded retries."""

from __future__ import annotations

from pathlib import Path

from klineage.constants import MAX_RETRIES, TIMEOUT, RunKind
from klineage.errors import StructuredOutputError, ValidationGateError
from klineage.harness.codex_runner import CodexRunner
from klineage.prompts import render_prompt
from klineage.utils import operation_id

VERIFY_PROMPT = """Inspect actual artifacts in the working directory.
Return true only when the requested outputs satisfy the task; uncertainty means false.
"""


class Action:
    verify_prompt = VERIFY_PROMPT

    def __init__(
        self,
        prompt: str,
        workdir: Path,
        enable_verifier: bool = True,
        timeout: int = TIMEOUT,
        *,
        max_retries: int = MAX_RETRIES,
    ):
        if type(enable_verifier) is not bool:
            raise TypeError("enable_verifier must be a boolean")
        if type(max_retries) is not int:
            raise TypeError("max_retries must be an integer")
        if max_retries < 0:
            raise ValueError("max_retries must be nonnegative")

        self.prompt = prompt
        self.workdir = Path(workdir).expanduser().resolve()
        self.enable_verifier = enable_verifier
        self.timeout = timeout
        self.max_retries = max_retries
        self.runner = CodexRunner(self.workdir, timeout=timeout)
        self.response: str | None = None
        self.run_id: str | None = None
        self.attempt = 0

    def run(self):
        retries = self.max_retries if self.enable_verifier else 0
        prompt = self.prompt
        for retry in range(retries + 1):
            self.attempt = retry + 1
            self.response = None
            self.run_id = operation_id(type(self).__name__)
            try:
                result = self.runner(prompt, run_id=self.run_id)
                self.response = result.final_message
                if not self.enable_verifier:
                    return
                passed = self.verify(self.verify_prompt)
                if type(passed) is not bool:
                    raise StructuredOutputError("verification must return a boolean")
                if not passed:
                    raise ValidationGateError(
                        f"verification failed after {self.attempt} attempts"
                    )
                return
            except Exception as error:
                if retry == retries:
                    raise
                prompt = (
                    f"Previous attempt failed: {type(error).__name__}: {error}\n"
                    "Read the prior generation and verification records in this workdir. "
                    "Correct the failed artifacts, remove obsolete source files, and "
                    "rewrite all required outputs before finishing.\n\n"
                    f"{self.prompt}"
                )

    def verify(self, prompt: str) -> bool:
        # Verify inherits Action, so load it after Action is defined.
        from klineage.action.verify import verify

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        return verify(
            render_prompt(RunKind.VERIFY, criteria=prompt, run_id=self.run_id),
            self.workdir,
            timeout=self.timeout,
        )


__all__ = ["Action"]

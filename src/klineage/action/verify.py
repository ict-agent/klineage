"""A Codex action whose output is a strict boolean."""

from pathlib import Path

from klineage.action.action import TIMEOUT, Action
from klineage.errors import StructuredOutputError

BOOLEAN_PROMPT = "\nReturn exactly true or false. No explanation or Markdown."


class Verify(Action):
    def __init__(self, prompt: str, workdir: Path, *, timeout: int = TIMEOUT):
        super().__init__(
            prompt + BOOLEAN_PROMPT, workdir, enable_verifier=False, timeout=timeout
        )
        self.passed = False

    def run(self):
        self.passed = False
        super().run()
        response = self.response.strip() if isinstance(self.response, str) else ""
        if response not in ("true", "false"):
            raise StructuredOutputError("verification must return true or false")
        self.passed = response == "true"


def verify(prompt: str, workdir: Path, *, timeout: int = TIMEOUT) -> bool:
    action = Verify(prompt, workdir, timeout=timeout)
    action.run()
    return action.passed


__all__ = ["Verify", "verify"]

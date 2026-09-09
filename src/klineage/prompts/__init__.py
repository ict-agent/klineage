"""Load packaged Jinja prompt templates."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from jinja2 import (
    ChoiceLoader,
    Environment,
    PackageLoader,
    StrictUndefined,
    select_autoescape,
)

PROMPT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@lru_cache(maxsize=1)
def prompt_environment() -> Environment:
    environment = Environment(
        loader=ChoiceLoader(
            [
                PackageLoader("klineage", "prompts"),
                PackageLoader("klineage", ""),
            ]
        ),
        autoescape=select_autoescape(default=False),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
    )
    # Tensor argument order is part of the ABI.
    environment.policies["json.dumps_kwargs"] = {
        "sort_keys": False,
        "ensure_ascii": False,
    }
    return environment


def render_prompt(name: str, /, **context: Any) -> str:
    """Render ``klineage/prompts/<name>.j4`` with strict variables."""

    if not PROMPT_NAME_RE.fullmatch(name):
        raise ValueError(f"invalid prompt template name: {name!r}")
    return prompt_environment().get_template(f"{name}.j4").render(**context).strip()


__all__ = ["render_prompt"]

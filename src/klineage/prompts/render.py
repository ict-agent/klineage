"""Load packaged Jinja prompt templates."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from jinja2 import (
    Environment,
    PackageLoader,
    StrictUndefined,
    select_autoescape,
)

from klineage.artifact.kernel import KERNEL
from klineage.artifact.tree import Node
from klineage.memory import SkillCard
from klineage.serialization import to_json

PROMPT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(?:/[a-z][a-z0-9_]*)?$")


def action_spec(stage: str) -> Node:
    """Describe the artifacts an action asks the agent to produce."""
    common = (
        "Action.run returns an Artifact; JSON is serialization data, not an output-directory file.",
        "Use current_kernel/ for trial sources. klineage.evaluate and klineage.result with work automatically append complete results under versions/0, versions/1, ...; agents do not save Kernels.",
        "Action uses the largest numbered saved version, including failed results, without automatic fallback. Retrieve an earlier accepted job with klineage.result(job_id, work=...) to append it as the latest version.",
        "With explicitly supplied devices and tools, write diagnostic scripts and use vendor profilers directly; their outputs are not validation.",
        "Do not run diagnostics and service benchmarks concurrently on the same device.",
        "The framework includes the measured Kernel as the kernel part of the returned Artifact.",
        "Treat upstream artifacts and mounted skills as read-only.",
        "Consume a result only after run succeeds; partial files are not a handoff.",
        "Read validation and evaluation diagnostics from the saved Kernel metadata.json or the reply. Explain source evidence in the final response; build caches are not output artifacts.",
        "Every attempt retains prompt.txt, trace.jsonl, stats.jsonl, final_message.txt and stderr.log in its runner record.",
    )
    descriptions = {
        "generation": (
            "Create a kernel implementing the supplied Problem and every workload in its configured language and backend.",
            "Evaluate standalone candidates against the problem oracle and retain the best accepted result.",
            "Obtain a correct measured implementation before optional performance experiments.",
            "Preserve the complete definition, ordered ABI, workload set and execution target.",
            "Return a measured Kernel without a SkillCard.",
        ),
        "init": (
            "Reproduce the fixed expert instance without expert-library dependencies in the delivered bundle.",
            "Stage the repository under workspace/repository/ and the original expert adapter under workspace/expert/.",
            "Check correctness, paired performance and mechanism fidelity; explain source evidence in the final response.",
            "Preserve upstream release build settings and record the exact expert specialization and host call.",
        ),
        "decompose": (
            "Remove exactly one independent optimization and preserve the complete problem and ordered ABI.",
            "A removal emits a SKILL.md via save_skill; the framework nests it under skillcard/.",
            "An unchanged naive input preserves its name and emits no SKILL.md; evaluating it automatically saves a complete new version.",
            "A step reaching naive still emits a card. Budget exhaustion does not prove naive status.",
            "Check correctness, actual mechanism removal and recipe prerequisites; a slowdown is not required.",
        ),
        "apply": (
            "One round may combine techniques and run multiple measured trials.",
            "Keep the round input fixed as the paired performance reference whenever it has sources.",
            "Return the best validated candidate; a changed implementation needs a measured speedup.",
            "For a source-free start, validate against the problem oracle and retain the best measured candidate.",
            "An unchanged result preserves name and sources, uses fresh service validation, and is saved automatically as a complete version without SKILL.md.",
            "Never emit a SkillCard. Empty memory does not stop optimization; prerequisites depend on current code.",
            "Performance inspection may use the supplied backend's tools; official correctness and acceptance timing remain service-owned.",
        ),
    }
    children = []
    if stage in descriptions:
        children.append(
            Node.directory("kernel", KERNEL.children, description=KERNEL.description)
        )
    if stage == "decompose":
        children.append(
            Node.directory("skillcard", SkillCard.schema.children, description=SkillCard.schema.description)
        )
    return Node.directory(
        ".",
        description="\n".join(
            (
                f"{stage.capitalize()} action output.",
                *common,
                *descriptions.get(stage, ()),
            )
        ),
        children=children,
    )


@lru_cache(maxsize=1)
def prompt_environment() -> Environment:
    environment = Environment(
        loader=PackageLoader("klineage.prompts", "templates"),
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
    environment.policies["json.dumps_function"] = (
        lambda value, **options: to_json(value, indent=options.get("indent"))
    )
    return environment


def render_prompt(name: str, /, **context: Any) -> str:
    """Render a task template with strict variables."""

    if not PROMPT_NAME_RE.fullmatch(name):
        raise ValueError(f"invalid prompt template name: {name!r}")
    return prompt_environment().get_template(f"{name}.j4").render(**context).strip()


__all__ = ["action_spec", "render_prompt"]

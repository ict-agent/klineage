"""Generate a kernel from selected skills."""

from __future__ import annotations

from collections.abc import Sequence

from klineage.action._sandbox import _Kind, _next, _Sandbox
from klineage.errors import StructuredOutputError
from klineage.harness.artifacts import is_cuda_language, require_cuda_source_bundle
from klineage.kernel import Kernel
from klineage.memory.skillcard import SkillCard
from klineage.prompts import render_prompt


def code_gen(
    current_kernel: Kernel,
    skills: Sequence[SkillCard],
) -> Kernel:
    """Materialize SkillCards in a new sandbox without claiming validation."""

    cards = _skill_sequence(skills)
    with _next(_Kind.CODE_GEN, current_kernel) as sandbox:
        return _materialize(sandbox, current_kernel, cards)


def _materialize(
    sandbox: _Sandbox,
    current_kernel: Kernel,
    skills: Sequence[SkillCard],
    *,
    output: str = "submission",
) -> Kernel:
    if current_kernel.problem is None:
        raise ValueError("code generation requires a kernel problem specification")
    if current_kernel.abi is None:
        raise ValueError("code generation requires a kernel ABI contract")

    submission = sandbox._dir(output)
    payload = {
        "current_kernel": current_kernel._prompt_input(),
        # Admission evidence stays in memory; generation needs actionable edits.
        "skill_cards": [
            {
                "intent": card.intent,
                "anchor": card.anchor,
                "carrier": card.carrier,
                "precondition": list(card.preconditions),
                "effect": list(card.effects),
                "risk": list(card.risks),
                "scope": card.scope.to_dict(),
            }
            for card in skills
        ],
        "submission_dir": submission,
    }
    response = sandbox._ask(
        "materialize",
        render_prompt("code_gen", skill_count=len(skills)),
        payload,
    )
    if response != {"done": True}:
        raise StructuredOutputError("code_gen response must be exactly {'done': true}")

    source_files, artifact = sandbox._take_bundle(
        output,
        "generated source bundle",
        current_kernel.abi.interface,
    )
    if is_cuda_language(current_kernel.context.language):
        require_cuda_source_bundle(source_files)
    source = source_files[current_kernel.abi.interface.module]
    context = current_kernel.context.with_actions(
        card.action_category for card in skills
    )
    return Kernel(
        name=f"{current_kernel.name}.materialized",
        source=source,
        context=context,
        artifact_path=artifact,
        problem=current_kernel.problem,
        abi=current_kernel.abi,
        source_files=source_files,
    )


def _skill_sequence(skills: Sequence[SkillCard]) -> tuple[SkillCard, ...]:
    if isinstance(skills, (str, bytes)) or not isinstance(skills, Sequence):
        raise TypeError("skills must be a sequence of SkillCards")
    cards = tuple(skills)
    if any(not isinstance(card, SkillCard) for card in cards):
        raise TypeError("skills contains a non-SkillCard value")
    if not cards:
        raise ValueError("at least one SkillCard is required")
    if len({card.skill_id for card in cards}) != len(cards):
        raise ValueError("skills contains duplicate SkillCards")

    return cards

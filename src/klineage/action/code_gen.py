"""Generate a kernel from selected skills."""

from __future__ import annotations

from collections.abc import Sequence

from klineage.action._sandbox import _Kind, _next, _Sandbox
from klineage.errors import StructuredOutputError
from klineage.contract import KernelABI, ProblemSpec
from klineage.harness.artifacts import (
    is_cuda_language, require_cuda_source_bundle, require_pure_cuda,
)
from klineage.kernel import Kernel, TargetContext
from klineage.memory.paths import RetrievalPlan
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
                "locus": card.evidence[0].locus,
                "forward_edit": card.evidence[0].forward_edit,
                "carrier": card.carrier,
                "precondition": list(card.preconditions),
                "effect": list(card.effects),
                "risk": list(card.risks),
                "scope": card.scope.to_dict(),
                **{name: [feature.to_dict() for feature in getattr(card, name)]
                   for name in ("requires", "provides", "conflicts")
                   if getattr(card, name)},
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
        require_cuda_source_bundle(source_files, repository=sandbox._repo)
    source = source_files[current_kernel.abi.interface.module]
    return Kernel(
        name=f"{current_kernel.name}.materialized",
        source=source,
        context=current_kernel.context,
        artifact_path=artifact,
        problem=current_kernel.problem,
        abi=current_kernel.abi,
        source_files=source_files,
    )


def _seed(
    sandbox: _Sandbox, problem: ProblemSpec, abi: KernelABI,
    context: TargetContext, plan: RetrievalPlan,
    *, output: str = "baseline.cu",
) -> Kernel:
    entry = plan.lineage.states[plan.entry_index]
    response = sandbox._ask("seed-kernel", render_prompt("seed"), {
        "problem": problem.to_dict(), "kernel_abi": abi.to_dict(),
        "target": context.to_dict(),
        "initial_features": [item.to_dict() for item in entry.features],
        "source_example": entry.source_files or entry.source,
        "planned_intents": [card.intent for card in plan.skills],
        "output_path": sandbox._file(output),
    })
    if response != {"done": True}:
        raise StructuredOutputError("seed response must be exactly {'done': true}")
    source, artifact = sandbox._take_file(output, "baseline CUDA source")
    require_pure_cuda(source, repository=sandbox._repo)
    return Kernel(
        name=problem.name, source=source, context=context,
        artifact_path=artifact, problem=problem, abi=abi,
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

"""Apply selected skills and validate the generated kernel."""

from __future__ import annotations

from collections.abc import Sequence

from klineage.action._sandbox import _Kind, _next, _Sandbox
from klineage.action.code_gen import _materialize, _skill_sequence
from klineage.errors import ActionError, ValidationGateError
from klineage.kernel import Kernel
from klineage.memory.skillcard import SkillAdmission, SkillCard


def apply(
    current_kernel: Kernel,
    skills: Sequence[SkillCard],
    *,
    skill_admission: SkillAdmission = SkillAdmission.OFF,
) -> Kernel:
    """Materialize and validate selected skills in one action sandbox."""

    with _next(_Kind.APPLY, current_kernel) as sandbox:
        return _apply(sandbox, current_kernel, skills, skill_admission)


def _apply(
    sandbox: _Sandbox,
    current_kernel: Kernel,
    skills: Sequence[SkillCard],
    skill_admission: SkillAdmission,
) -> Kernel:
    if not isinstance(skill_admission, SkillAdmission):
        raise TypeError("skill_admission must be a SkillAdmission")
    cards = _skill_sequence(skills)
    if skill_admission is SkillAdmission.ON:
        hypotheses = [card.skill_id for card in cards if not card.admitted]
        if hypotheses:
            raise ActionError(
                "cannot apply unadmitted SkillCards: " + ", ".join(hypotheses)
            )

    candidate = _materialize(sandbox, current_kernel, cards)
    validation = sandbox._evaluate(candidate, reference=current_kernel)
    candidate = candidate.with_validation(validation)
    if validation.accepted:
        return candidate

    raise ValidationGateError(
        "materialized kernel failed the compile/correctness/profile gate",
        kernel=candidate,
    )

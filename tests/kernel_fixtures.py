"""Kernel and skill fixtures for domain tests."""

from problem_fixtures import problem_spec

from klineage.harness.eval import ValidationResult
from klineage.kernel import Kernel
from klineage.memory import Scope, SkillCard

CUDA_SOURCE = 'extern "C" __global__ void kernel(float *x) { x[0] += 1; }'


def accepted(latency: float = 1.0) -> ValidationResult:
    return ValidationResult(True, True, True, latency)


def kernel(
    name: str = "kernel",
    *,
    case: str = "gemm",
    language: str = "cuda",
    platform: str = "sm120",
    source: str = CUDA_SOURCE,
    validation: ValidationResult | None = None,
) -> Kernel:
    return Kernel(
        name=name,
        source_files={"kernel.cu": source},
        problem=problem_spec(case, language, platform),
        validation=validation,
    )


def skill(
    skill_id: str,
    *,
    declared: Scope | None = None,
) -> SkillCard:
    return SkillCard(
        skill_id=skill_id,
        intent=f"intent {skill_id}",
        preconditions=("precondition",),
        scope=declared or Scope(),
        body=f"# Overview\n\nApply {skill_id} to the current kernel.",
    )

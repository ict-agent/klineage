"""Initialize an expert kernel from a problem and repository."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from klineage.action._contract import _contract
from klineage.action._sandbox import _Kind, _new, _Sandbox
from klineage.contract import relative_source_path
from klineage.errors import StructuredOutputError, ValidationGateError
from klineage.harness.artifacts import (
    is_cuda_language,
    require_pure_cuda,
)
from klineage.harness.eval import ValidationResult
from klineage.harness.fidelity import verify_mechanism, verify_performance
from klineage.kernel import Kernel
from klineage.prompts import render_prompt

_MAX_ATTEMPTS = 6
_MINIMUM_EXPERT_RATIO = 0.99
_EXPERT_DIR = ".klineage-expert"


def init(
    problem: str | os.PathLike[str],
    repo: str | os.PathLike[str],
    expert_kernel: str | os.PathLike[str],
) -> Kernel:
    """Generate and verify raw CUDA from three experiment inputs."""

    with _new(_Kind.INIT, problem, repo) as sandbox:
        repository = sandbox._repo
        expert_path = _expert_path(repository, expert_kernel)
        expert_name = expert_path.relative_to(repository).as_posix()
        description = sandbox._inspect()
        problem_spec, abi, context = _contract(
            sandbox._problem,
            description,
        )
        sandbox._save(problem_spec, abi, context)
        expert = Kernel(
            name=expert_name,
            source=expert_path.read_text(encoding="utf-8"),
            context=context,
            artifact_path=expert_path,
            problem=problem_spec,
            abi=abi,
        )
        return _generate(sandbox, expert)


def _generate(
    sandbox: _Sandbox,
    expert_kernel: Kernel,
) -> Kernel:
    """Run the generation loop with already prepared private dependencies."""

    problem, abi = expert_kernel.problem, expert_kernel.abi
    context = expert_kernel.context
    if not is_cuda_language(context.language):
        raise ValueError("init requires a CUDA/C++ target context")

    expert_validation = sandbox._evaluate(expert_kernel, reference=None)
    expert_kernel = expert_kernel.with_validation(expert_validation)
    if not expert_validation.accepted:
        reason = _validation_reason(expert_validation)
        raise ValidationGateError(
            "specified expert kernel is not compilable, runnable, and ABI-correct"
            f": {reason}; see {sandbox._logs}",
            kernel=expert_kernel,
        )

    last_kernel: Kernel | None = None
    feedback: dict[str, Any] | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        artifact_name = f"candidate-{attempt:02d}.cu"
        payload = {
            "repository": str(sandbox._repo),
            "kernel_name": expert_kernel.name,
            "target": context.to_dict(),
            "problem": problem.to_dict(),
            "kernel_abi": abi.to_dict(),
            "minimum_expert_ratio": _MINIMUM_EXPERT_RATIO,
            "previous_gate_feedback": feedback,
            "output_path": sandbox._file(artifact_name),
        }
        response = sandbox._ask(
            "init-kernel",
            render_prompt("init"),
            payload,
        )
        if response != {"done": True}:
            raise StructuredOutputError("init response must be exactly {'done': true}")
        source, artifact_path = sandbox._take_file(
            artifact_name,
            "expert kernel source",
        )
        try:
            require_pure_cuda(source, repository=sandbox._repo)
        except ValidationGateError as error:
            feedback = {
                "artifact_gate": {
                    "error": str(error),
                    "error_type": type(error).__name__,
                }
            }
            continue
        candidate = Kernel(
            name=expert_kernel.name,
            source=source,
            context=context,
            problem=problem,
            abi=abi,
            artifact_path=artifact_path,
        )
        validation = _verify_candidate(
            sandbox._evaluate(candidate, reference=expert_kernel),
            expert_kernel=expert_kernel,
        )
        if validation.accepted:
            mechanism = verify_mechanism(
                sandbox._ask, expert_kernel, candidate, sandbox._repo,
            )
            validation = replace(
                validation,
                profile_passed=mechanism["passed"],
                details={**validation.details, "mechanism_verifier": mechanism},
            )
        if validation.accepted:
            # Remeasure the frozen artifact and expert in a fresh worker.
            confirmation = _verify_candidate(
                sandbox._evaluate(candidate, reference=expert_kernel),
                expert_kernel=expert_kernel,
            )
            validation = replace(
                confirmation,
                details={
                    **confirmation.details,
                    "mechanism_verifier": validation.details["mechanism_verifier"],
                    "confirmation": {
                        "independent": True, "selection": validation.to_dict(),
                    },
                },
            )
        candidate = candidate.with_validation(validation)
        last_kernel = candidate
        feedback = validation.to_dict()
        (sandbox._logs / f"fidelity-{attempt:02d}.json").write_text(
            json.dumps(feedback, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if validation.accepted:
            return candidate

    raise ValidationGateError(
        f"could not initialize a validated expert kernel in {_MAX_ATTEMPTS} attempts",
        kernel=last_kernel,
    )


def _verify_candidate(
    validation: ValidationResult,
    *,
    expert_kernel: Kernel,
) -> ValidationResult:
    validation = verify_performance(validation, _MINIMUM_EXPERT_RATIO)
    return replace(
        validation,
        details={
            **validation.details,
            "init_verifier": {
                "expert_fingerprint": expert_kernel.fingerprint,
                "minimum_expert_ratio": _MINIMUM_EXPERT_RATIO,
                "relative_performance": validation.relative_performance,
            },
        },
    )


def _validation_reason(validation: ValidationResult) -> str:
    for stage in ("build_load", "correctness", "timing", "worker"):
        value = validation.details.get(stage)
        if isinstance(value, Mapping):
            error = value.get("error")
            if isinstance(error, str) and error.strip():
                return error.strip()
    return "validation gate failed"


def _expert_path(
    repository: Path,
    value: str | os.PathLike[str],
) -> Path:
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return _copy_expert(repository, raw)

    name = relative_source_path(os.fspath(value), "expert kernel path")
    path = repository / name
    if path.is_symlink():
        raise ValueError("expert kernel cannot be a symbolic link")
    path = path.resolve(strict=True)
    if not path.is_relative_to(repository) or not path.is_file():
        raise ValueError("expert kernel must be a regular file inside repo")
    return path


def _copy_expert(repository: Path, source: Path) -> Path:
    if source.is_symlink():
        raise ValueError("expert kernel cannot be a symbolic link")
    source = source.resolve(strict=True)
    if not source.is_file():
        raise ValueError("expert kernel must be a regular file")

    directory = repository / _EXPERT_DIR
    directory.mkdir(mode=0o700)
    snapshot = directory / source.name
    snapshot.write_bytes(source.read_bytes())
    return snapshot.resolve(strict=True)


__all__ = ["init"]

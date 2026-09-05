"""Initialize an expert kernel from a problem and repository."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from klineage.action._sandbox import _Kind, _new, _Sandbox
from klineage.contract import RAW_CUDA_ABI, KernelABI, ProblemSpec, relative_source_path
from klineage.errors import StructuredOutputError, ValidationGateError
from klineage.harness.artifacts import (
    is_cuda_language,
    require_pure_cuda,
)
from klineage.harness.eval import ValidationResult
from klineage.kernel import Kernel, TargetContext
from klineage.prompts import render_prompt

_MAX_ATTEMPTS = 6
_MINIMUM_EXPERT_RATIO = 0.95
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
            require_pure_cuda(source)
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
        candidate = candidate.with_validation(validation)
        last_kernel = candidate
        if validation.accepted:
            return candidate
        feedback = validation.to_dict()

    raise ValidationGateError(
        f"could not initialize a validated expert kernel in {_MAX_ATTEMPTS} attempts",
        kernel=last_kernel,
    )


def _verify_candidate(
    validation: ValidationResult,
    *,
    expert_kernel: Kernel,
) -> ValidationResult:
    ratio = validation.relative_performance
    details = dict(validation.details)
    details["init_verifier"] = {
        "expert_fingerprint": expert_kernel.fingerprint,
        "minimum_expert_ratio": _MINIMUM_EXPERT_RATIO,
        "relative_performance": ratio,
    }
    return replace(
        validation,
        profile_passed=(
            validation.profile_passed
            and ratio is not None
            and ratio >= _MINIMUM_EXPERT_RATIO
        ),
        details=details,
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


def _contract(
    problem_path: Path,
    description: Mapping[str, Any],
) -> tuple[ProblemSpec, KernelABI, TargetContext]:
    try:
        name = description["problem_name"]
        raw_abi = description["abi"]
        platform = description["platform"]
    except KeyError as exc:
        raise ValueError(f"problem inspection omitted {exc.args[0]!r}") from exc
    if not isinstance(name, str) or not isinstance(platform, str):
        raise TypeError("problem inspection name and platform must be strings")
    if not isinstance(raw_abi, Mapping):
        raise TypeError("problem inspection ABI must be an object")

    source = problem_path.read_text(encoding="utf-8")
    relative = f"{problem_path.parent.name}/{problem_path.name}"
    statement = f"""Authoritative input problem ({relative}):

```python
{source}
```

Implement exactly those semantics with this standalone raw CUDA ABI:
    {RAW_CUDA_ABI}
`inputs[i]` and `outputs[i]` follow kernel_abi declaration order. Shapes and
dtypes are fixed by kernel_abi. Launch on the supplied stream, do not
synchronize it, and return the CUDA launch status. Do not call PyTorch, cuBLAS,
another framework/library implementation, a subprocess, or a precomputed
result.
""".strip()
    problem = ProblemSpec(name=name, statement=statement)
    abi = KernelABI.from_dict(raw_abi)
    context = TargetContext(
        case=problem.name,
        language="cuda",
        platform=platform,
    )
    return problem, abi, context


__all__ = ["init"]

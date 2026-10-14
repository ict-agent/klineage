from __future__ import annotations

import atexit
import hashlib
import json
import tempfile
from pathlib import Path

from problem_fixtures import problem_spec

from klineage.artifact import (
    Kernel,
    Node,
    ValidationResult,
    json_node,
    source_files,
    source_tree,
)
from klineage.artifact.kernel import kernel_layouts
from klineage.artifact.problem import load_problem, problem_reference
from klineage.backend import get_backend
from klineage.memory import Scope, SkillCard
from klineage.serialization import raw_text, to_json

PROBLEM_DATA = tempfile.TemporaryDirectory(prefix="klineage-test-problems-")
atexit.register(PROBLEM_DATA.cleanup)

CUDA_SOURCE = 'extern "C" __global__ void kernel(float *x) { x[0] += 1; }'


def accepted(latency: float = 1.0) -> ValidationResult:
    return ValidationResult(True, True, True, latency)


def measured(kernel, validation=None) -> ValidationResult:
    result = validation if validation is not None else accepted()
    return ValidationResult.combine({
        case["uuid"]: result for case in load_problem(kernel).workloads
    })


def kernel(
    name: str = "kernel",
    *,
    case: str = "gemm",
    language: str = "cuda",
    platform: str = "sm120",
    source: str = CUDA_SOURCE,
    validation: ValidationResult | None = None,
) -> Kernel:
    return make_kernel(
        name=name,
        source_files={get_backend(language, platform).kernel_type.kernel_spec().find("submission").find("solution").children[0].name: source},
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


def make_kernel(name, problem, source_files=None, validation=None, compile_flags=()):
    payload = to_json(problem)
    path = Path(PROBLEM_DATA.name) / hashlib.sha256(payload.encode()).hexdigest()
    path.mkdir(exist_ok=True)
    definition = problem.definition
    (path / "definition.json").write_text(to_json(definition) + "\n")
    (path / "workloads.jsonl").write_text(
        "".join(
            to_json({"definition": definition.get("name", problem.name), "workload": case}, indent=None) + "\n"
            for case in problem.workloads
        )
    )
    (path / "target.json").write_text(to_json(problem.target()) + "\n")
    reference = problem_reference(path, problem)
    sources = source_files or {}
    # Native fixtures supply the complete configured bundle expected from authors.
    if sources and "config.toml" not in sources:
        entry = next(iter(sources))
        sources = {
            "config.toml": (
                f'[solution]\nname="{name}"\ndefinition="{problem.name}"\nauthor="test"\n'
                f'[build]\nlanguage="{problem.language}"\nentry_point="{entry}::run"\n'
                + ("destination_passing_style = true\n" if problem.language != "python" else "")
            ),
            **{f"solution/{path}": text for path, text in sources.items()},
        }
    cls = get_backend(problem.language, problem.platform).kernel_type
    tree = Node.directory(
        ".",
        (
            json_node(
                "metadata.json",
                {
                    "name": name,
                    "problem": reference,
                    "validation": validation,
                    "compile_flags": list(compile_flags),
                },
            ),
            Node.directory("submission", source_tree(sources).children),
        ),
    )
    for layout in kernel_layouts(cls):
        try:
            layout.check(tree)
        except ValueError:
            continue
        return cls(tree)

    class FixtureKernel(cls):
        @classmethod
        def kernel_spec(cls):
            return tree

    return FixtureKernel(tree)


def edit_kernel(current, **changes):
    metadata = json.loads(raw_text(current.find("metadata.json")))
    metadata.pop("evaluation", None)
    values = {
        **metadata,
        "validation": current.validation,
        "problem": load_problem(current),
        "source_files": source_files(current.find("submission")),
    }
    if {"source_files", "compile_flags", "problem"} & changes.keys():
        values["validation"] = None
    return make_kernel(**(values | changes))


def load_entry(loader, kernel):
    from klineage.backend.runtime import bundle_entry

    problem = load_problem(kernel)
    backend = get_backend(problem.language, problem.platform)
    sources = source_files(kernel.find("submission"))
    flags = json.loads(raw_text(kernel.find("metadata.json"))).get("compile_flags", ())
    return loader.load_entry(sources, bundle_entry(sources, backend), backend, flags)


def write_kernel(current, directory):
    """Prepare authored test files, not a serialized artifact directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metadata.json").write_text(raw_text(current.find("metadata.json")))
    for name, content in source_files(current.find("submission")).items():
        path = directory / "submission" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return directory

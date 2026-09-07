from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from klineage.action.code_gen import _materialize
from klineage.contract import (
    PYTHON_CALLABLE_PROTOCOL,
    ABIValue,
    EvaluatorInterface,
    KernelABI,
    ProblemSpec,
)
from klineage.errors import StructuredOutputError, ValidationGateError
from klineage.harness.artifacts import (
    read_generated_source,
    read_generated_source_bundle,
)
from klineage.harness.eval import ValidationResult
from klineage.harness.structured import run_json
from klineage.kernel import Kernel, TargetContext
from klineage.memory import Scope, SkillCard, TransitionEvidence

ABI_SCHEMA_VERSION = 1
LOADER_SOURCE = "def load():\n    return object()\n"
CUDA_SOURCE = 'extern "C" __global__ void candidate(float *x) { x[0] += 1; }'
SOURCE_FILES = {"submission.py": LOADER_SOURCE, "src/candidate.cu": CUDA_SOURCE}
CUDA_FILES = {
    "config.toml": '''[solution]
name = "vector-add"
definition = "vector-add"
author = "klineage"
[build]
language = "cuda"
entry_point = "kernel.cu::candidate"
destination_passing_style = true
''',
    "solution/kernel.cu": CUDA_SOURCE,
}


def accepted() -> ValidationResult:
    return ValidationResult(True, True, True, 1.0, 2.0)


def problem() -> ProblemSpec:
    return ProblemSpec(
        name="vector-add",
        statement="Increment every element of x and return the result.",
        parameters={"N": 1024, "dtypes": ["float32"]},
    )


def abi(interface: EvaluatorInterface | None = None) -> KernelABI:
    return KernelABI(
        inputs=(ABIValue("x", "float32", ("N",)),),
        outputs=(ABIValue("result", "float32", ("N",)),),
        interface=interface or EvaluatorInterface(),
    )


def kernel() -> Kernel:
    return Kernel(
        name="baseline",
        source=LOADER_SOURCE,
        context=TargetContext("vector-add-N1024", "cuda", "sm90"),
        problem=problem(),
        abi=abi(),
        source_files=SOURCE_FILES,
    )


def skill() -> SkillCard:
    evidence = TransitionEvidence(
        action_category="tile",
        locus="output loop",
        before_fingerprint="before",
        after_fingerprint="after",
        backward_edit="remove tiling",
        forward_edit="restore tiling",
        predecessor_validation=accepted(),
        roundtrip_validation=accepted(),
    )
    return SkillCard(
        skill_id="tile",
        intent="tile the output",
        anchor="output loop",
        carrier="use tiled output traversal",
        preconditions=("contiguous output",),
        effects=("coalesced stores",),
        evidence=(evidence,),
        risks=("tail handling",),
        scope=Scope(),
    )


def write_tree(root: Path, files: dict[str, str]) -> None:
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)


def read_bundle(
    root: Path,
    submission: Path,
    interface: EvaluatorInterface | None = None,
) -> dict[str, str]:
    return read_generated_source_bundle(
        submission,
        "generated source bundle",
        allowed_root=root,
        interface=interface or EvaluatorInterface(),
    )


class FakeRunner:
    def __init__(self, sandbox: Path, files: dict[str, str]) -> None:
        self.sandbox_dir = sandbox.resolve()
        self.state_dir = self.sandbox_dir / ".klineage"
        self.state_dir.mkdir(parents=True)
        self.files = files
        self.payload: dict[str, object] = {}

    def __call__(self, prompt: str, **_: object) -> SimpleNamespace:
        self.payload = json.loads(prompt.split("INPUT_JSON:\n", 1)[1])

        if isinstance(output_path := self.payload.get("output_path"), str):
            path = self.sandbox_dir / output_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(CUDA_SOURCE)

        if isinstance(submission_dir := self.payload.get("submission_dir"), str):
            write_tree(self.sandbox_dir / submission_dir, self.files)

        trace_path = self.state_dir / "trace.jsonl"
        return SimpleNamespace(final_message='{"done": true}', trace_path=trace_path)


class SourceEvaluator:
    def __init__(self) -> None:
        self.calls: list[tuple[Kernel, Kernel | None]] = []

    def evaluate(
        self,
        candidate: Kernel,
        *,
        reference: Kernel | None = None,
    ) -> ValidationResult:
        self.calls.append((candidate, reference))
        return accepted()


class FakeSandbox:
    def __init__(
        self,
        runner: FakeRunner,
        evaluator: SourceEvaluator | None = None,
    ) -> None:
        self._runner = runner
        self._evaluator = evaluator or SourceEvaluator()
        self._repo = runner.sandbox_dir / "repository"
        self._repo.mkdir()
        self._logs = runner.state_dir / "evaluations"
        self._logs.mkdir()
        self._artifacts = runner.state_dir / "artifacts"
        self._artifacts.mkdir()

    def _file(self, name: str) -> str:
        path = self._runner.sandbox_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return name

    def _dir(self, name: str) -> str:
        (self._runner.sandbox_dir / name).mkdir(parents=True)
        return name

    def _ask(
        self,
        purpose: str,
        instructions: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        return run_json(self._runner, purpose, instructions, payload)  # type: ignore[arg-type]

    def _take_file(self, name: str, label: str) -> tuple[str, Path]:
        source = read_generated_source(
            self._runner.sandbox_dir / name,
            label,
            allowed_root=self._runner.sandbox_dir,
        )
        artifact = self._artifacts / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(source, encoding="utf-8")
        return source, artifact.resolve()

    def _take_bundle(
        self,
        name: str,
        label: str,
        interface: EvaluatorInterface,
    ) -> tuple[dict[str, str], Path]:
        files = read_generated_source_bundle(
            self._runner.sandbox_dir / name,
            label,
            allowed_root=self._runner.sandbox_dir,
            interface=interface,
        )
        artifact = self._artifacts / name
        artifact.mkdir()
        write_tree(artifact, files)
        return files, artifact.resolve()

    def _evaluate(
        self,
        candidate: Kernel,
        *,
        reference: Kernel | None = None,
    ) -> ValidationResult:
        return self._evaluator.evaluate(candidate, reference=reference)


class TempTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()


class ContractTests(unittest.TestCase):
    def test_problem_and_abi_json_schema(self) -> None:
        problem_value = problem()
        abi_value = abi(EvaluatorInterface("python/entrypoint.py", "load_candidate"))
        problem_json = problem_value.to_dict()
        abi_json = abi_value.to_dict()

        self.assertEqual(set(problem_json), {"name", "statement", "parameters"})
        self.assertEqual(abi_json["version"], ABI_SCHEMA_VERSION)
        self.assertEqual(
            abi_json["interface"]["protocol"],
            PYTHON_CALLABLE_PROTOCOL,
        )
        self.assertEqual(ProblemSpec.from_dict(problem_json), problem_value)
        self.assertEqual(KernelABI.from_dict(abi_json), abi_value)

    def test_abi_rejects_wrong_version_or_protocol(self) -> None:
        abi_json = abi().to_dict()
        abi_json["version"] = ABI_SCHEMA_VERSION + 1
        with self.assertRaises(ValueError):
            KernelABI.from_dict(abi_json)

        interface_json = EvaluatorInterface().to_dict()
        interface_json["protocol"] = "shared-library-v1"
        with self.assertRaises(ValueError):
            EvaluatorInterface.from_dict(interface_json)

        del abi_json["version"]
        with self.assertRaises(KeyError):
            KernelABI.from_dict(abi_json)
        del interface_json["protocol"]
        with self.assertRaises(KeyError):
            EvaluatorInterface.from_dict(interface_json)

    def test_interface_rejects_unsafe_paths_and_symbols(self) -> None:
        modules = ("/x.py", "../x.py", "a/../x.py", "C:/x.py", "x.txt")
        for module in modules:
            with self.subTest(module=module), self.assertRaises(ValueError):
                EvaluatorInterface(module=module)

        for loader in ("load.candidate", "class"):
            with self.subTest(loader=loader), self.assertRaises(ValueError):
                EvaluatorInterface(loader=loader)

    def test_kernel_source_must_match_its_interface_module(self) -> None:
        with self.assertRaisesRegex(ValueError, "interface module"):
            Kernel(
                name="candidate",
                source="def different():\n    pass\n",
                context=TargetContext("vector-add", "cuda", "sm90"),
                problem=problem(),
                abi=abi(),
                source_files=SOURCE_FILES,
            )


class SourceBundleTests(TempTest):
    def test_reads_cuda_config_entry(self) -> None:
        submission = self.root / "submission"
        submission.mkdir()
        write_tree(submission, CUDA_FILES)

        self.assertEqual(read_bundle(self.root, submission), CUDA_FILES)
        value = replace(kernel(), source=CUDA_SOURCE, source_files=CUDA_FILES)
        self.assertEqual(Kernel.from_dict(value.to_dict()), value)

    def test_reads_source_dir_with_nested_interface(self) -> None:
        interface = EvaluatorInterface("python/entrypoint.py", "load_candidate")
        files = {
            interface.module: LOADER_SOURCE,
            "src/candidate.cu": CUDA_SOURCE,
        }
        submission = self.root / "submission"
        submission.mkdir()
        write_tree(submission, files)
        result = read_bundle(self.root, submission, interface)
        self.assertEqual(result, files)

    def test_rejects_directory_outside_action_root(self) -> None:
        submission = self.root / "nested" / "submission"
        submission.mkdir(parents=True)
        write_tree(submission, SOURCE_FILES)
        with self.assertRaises(StructuredOutputError):
            read_bundle(self.root, submission)

    def test_rejects_symlinks(self) -> None:
        outside = self.root / "outside.py"
        outside.write_text(LOADER_SOURCE)
        submission = self.root / "submission"
        submission.mkdir()
        (submission / "submission.py").symlink_to(outside)

        with self.assertRaises(StructuredOutputError):
            read_bundle(self.root, submission)

    def test_requires_interface_module(self) -> None:
        submission = self.root / "submission"
        submission.mkdir()
        write_tree(submission, {"src/candidate.cu": CUDA_SOURCE})
        with self.assertRaises(StructuredOutputError):
            read_bundle(self.root, submission)


class MaterializeTests(TempTest):
    def test_requires_problem_and_abi(self) -> None:
        runner = FakeRunner(self.root, SOURCE_FILES)
        sandbox = FakeSandbox(runner)
        bare = Kernel(
            name="baseline",
            source=CUDA_SOURCE,
            context=TargetContext("vector-add", "cuda", "sm90"),
        )

        with self.assertRaises(ValueError):
            _materialize(sandbox, bare, (skill(),))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            _materialize(  # type: ignore[arg-type]
                sandbox,
                replace(bare, problem=problem()),
                (skill(),),
            )

    def test_writes_source_dir_and_embeds_contract(self) -> None:
        runner = FakeRunner(self.root, CUDA_FILES)
        sandbox = FakeSandbox(runner)
        generated = _materialize(  # type: ignore[arg-type]
            sandbox,
            kernel(),
            (skill(),),
        )

        current = runner.payload["current_kernel"]
        self.assertIsInstance(current, dict)
        self.assertEqual(current["problem"], problem().to_dict())
        self.assertEqual(current["abi"], abi().to_dict())
        self.assertTrue(generated.artifact_path.is_dir())
        self.assertEqual(generated.source_files, CUDA_FILES)
        self.assertEqual(generated.source, CUDA_SOURCE)

    def test_cuda_requires_build_config(self) -> None:
        runner = FakeRunner(self.root, SOURCE_FILES)
        with self.assertRaisesRegex((ValueError, StructuredOutputError), "config.toml"):
            _materialize(FakeSandbox(runner), kernel(), (skill(),))

    def test_cuda_bundle_rejects_cutlass(self) -> None:
        files = {**CUDA_FILES, "solution/kernel.cu":
                 "#include <cutlass/cutlass.h>\n" + CUDA_SOURCE}
        runner = FakeRunner(self.root, files)
        with self.assertRaisesRegex(ValidationGateError, "expert library"):
            _materialize(FakeSandbox(runner), kernel(), (skill(),))

    def test_requires_cuda_in_cuda_source_dir(self) -> None:
        runner = FakeRunner(self.root, {
            "config.toml": CUDA_FILES["config.toml"].replace(
                "kernel.cu::candidate", "binding.py::kernel"
            ).replace("true", "false"),
            "solution/binding.py": "def kernel(x):\n    return x\n",
        })
        sandbox = FakeSandbox(runner)
        with self.assertRaisesRegex(ValidationGateError, "does not contain a CUDA"):
            _materialize(  # type: ignore[arg-type]
                sandbox,
                kernel(),
                (skill(),),
            )


if __name__ == "__main__":
    unittest.main()

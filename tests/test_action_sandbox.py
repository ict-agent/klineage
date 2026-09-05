from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar, Self
from unittest.mock import patch

from klineage.action._sandbox import _Kind, _new, _next
from klineage.contract import EvaluatorInterface, KernelABI, ProblemSpec
from klineage.kernel import Kernel, TargetContext

CUDA_SOURCE = 'extern "C" __global__ void kernel(float *x) { x[0] += 1; }'


class FakeRunner:
    instances: ClassVar[list[FakeRunner]] = []

    def __init__(
        self,
        sandbox_dir: Path,
        *,
        read_roots: tuple[Path, ...] = (),
        **_: object,
    ) -> None:
        self.sandbox_dir = Path(sandbox_dir).resolve()
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir = self.sandbox_dir / ".klineage"
        self.state_dir.mkdir()
        self.read_roots = tuple(Path(path).resolve() for path in read_roots)
        self.instances.append(self)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None


class ActionSandboxTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeRunner.instances.clear()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.problem_path = self.root / "problem" / "reference.py"
        self.problem_path.parent.mkdir()
        self.problem_path.write_text("def torch_ref():\n    pass\n", encoding="utf-8")
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / "expert.cu").write_text(CUDA_SOURCE, encoding="utf-8")

    def test_kernel_resumes_the_same_run_in_a_new_sandbox(self) -> None:
        problem = ProblemSpec("case", "Compute the case.")
        abi = KernelABI()
        context = TargetContext("case", "cuda", "sm120")

        with (
            patch("klineage.action._sandbox._project_root", return_value=self.root),
            patch("klineage.action._sandbox.CodexRunner", FakeRunner),
            _new(_Kind.INIT, self.problem_path, self.repo) as sandbox,
        ):
            sandbox._save(problem, abi, context)
            target = sandbox._file("candidate.cu")
            (sandbox._work / target).write_text(CUDA_SOURCE, encoding="utf-8")
            source, artifact = sandbox._take_file("candidate.cu", "candidate")
            first_run = sandbox._run
            first_work = sandbox._work

        kernel = Kernel(
            "candidate",
            source,
            context,
            artifact_path=artifact,
            problem=problem,
            abi=abi,
        )
        restored = Kernel.from_dict(kernel.to_dict())

        with (
            patch("klineage.action._sandbox._project_root", return_value=self.root),
            patch("klineage.action._sandbox.CodexRunner", FakeRunner),
            _next(_Kind.CODE_GEN, restored) as sandbox,
        ):
            self.assertEqual(sandbox._run, first_run)
            self.assertNotEqual(sandbox._work, first_work)
            self.assertEqual(sandbox._problem.read_text(), self.problem_path.read_text())
            self.assertEqual((sandbox._repo / "expert.cu").read_text(), CUDA_SOURCE)
            self.assertEqual(FakeRunner.instances[-1].read_roots, (first_run / "input",))

    def test_tampered_run_contract_is_rejected(self) -> None:
        problem = ProblemSpec("case", "Compute the case.")
        abi = KernelABI()
        with (
            patch("klineage.action._sandbox._project_root", return_value=self.root),
            patch("klineage.action._sandbox.CodexRunner", FakeRunner),
            _new(_Kind.INIT, self.problem_path, self.repo) as sandbox,
        ):
            sandbox._save(
                problem,
                abi,
                TargetContext("case", "cuda", "sm120"),
            )
            target = sandbox._file("candidate.cu")
            (sandbox._work / target).write_text(CUDA_SOURCE, encoding="utf-8")
            source, artifact = sandbox._take_file("candidate.cu", "candidate")
            manifest = sandbox._run / "run.json"

        kernel = Kernel(
            "candidate",
            source,
            TargetContext("case", "cuda", "sm120"),
            artifact_path=artifact,
            problem=ProblemSpec("other", "Wrong contract."),
            abi=abi,
        )

        with (
            patch("klineage.action._sandbox._project_root", return_value=self.root),
            patch("klineage.action._sandbox.CodexRunner", FakeRunner),
            self.assertRaisesRegex(ValueError, "run contract"),
        ):
            self.assertTrue(manifest.is_file())
            with _next(_Kind.APPLY, kernel):
                pass

    def test_tampered_target_context_is_rejected(self) -> None:
        problem = ProblemSpec("case", "Compute the case.")
        abi = KernelABI()
        context = TargetContext("case", "cuda", "sm120")
        with (
            patch("klineage.action._sandbox._project_root", return_value=self.root),
            patch("klineage.action._sandbox.CodexRunner", FakeRunner),
            _new(_Kind.INIT, self.problem_path, self.repo) as sandbox,
        ):
            sandbox._save(problem, abi, context)
            target = sandbox._file("candidate.cu")
            (sandbox._work / target).write_text(CUDA_SOURCE, encoding="utf-8")
            source, artifact = sandbox._take_file("candidate.cu", "candidate")

        kernel = Kernel(
            "candidate",
            source,
            TargetContext("case", "cuda", "sm90"),
            artifact_path=artifact,
            problem=problem,
            abi=abi,
        )

        with (
            patch("klineage.action._sandbox._project_root", return_value=self.root),
            patch("klineage.action._sandbox.CodexRunner", FakeRunner),
            self.assertRaisesRegex(ValueError, "target"),
            _next(_Kind.CODE_GEN, kernel),
        ):
            pass

    def test_extra_bundle_file_is_rejected(self) -> None:
        problem = ProblemSpec("case", "Compute the case.")
        abi = KernelABI()
        interface = EvaluatorInterface()
        files = {
            interface.module: "def load():\n    return lambda x: x\n",
            "kernel.cu": CUDA_SOURCE,
        }
        with (
            patch("klineage.action._sandbox._project_root", return_value=self.root),
            patch("klineage.action._sandbox.CodexRunner", FakeRunner),
            _new(_Kind.INIT, self.problem_path, self.repo) as sandbox,
        ):
            sandbox._save(
                problem,
                abi,
                TargetContext("case", "cuda", "sm120"),
            )
            output = sandbox._dir("submission")
            for relative, source in files.items():
                path = sandbox._work / output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source, encoding="utf-8")
            source_files, artifact = sandbox._take_bundle(
                output,
                "candidate",
                interface,
            )

        (artifact / "injected.py").write_text("raise RuntimeError\n", encoding="utf-8")
        kernel = Kernel(
            "candidate",
            source_files[interface.module],
            TargetContext("case", "cuda", "sm120"),
            artifact_path=artifact,
            problem=problem,
            abi=abi,
            source_files=source_files,
        )

        with (
            patch("klineage.action._sandbox._project_root", return_value=self.root),
            patch("klineage.action._sandbox.CodexRunner", FakeRunner),
            self.assertRaisesRegex(ValueError, "bundle does not match"),
            _next(_Kind.CODE_GEN, kernel),
        ):
            pass

    def test_runs_keep_inputs_separate(self) -> None:
        problem = ProblemSpec("case", "Compute the case.")
        abi = KernelABI()
        context = TargetContext("case", "cuda", "sm120")
        kernels: list[Kernel] = []
        with (
            patch("klineage.action._sandbox._project_root", return_value=self.root),
            patch("klineage.action._sandbox.CodexRunner", FakeRunner),
        ):
            for index in range(2):
                with _new(
                    _Kind.INIT,
                    self.problem_path,
                    self.repo,
                ) as sandbox:
                    sandbox._save(problem, abi, context)
                    name = f"candidate-{index}.cu"
                    target = sandbox._file(name)
                    (sandbox._work / target).write_text(
                        CUDA_SOURCE,
                        encoding="utf-8",
                    )
                    source, artifact = sandbox._take_file(name, "candidate")
                    kernels.append(
                        Kernel(
                            name,
                            source,
                            context,
                            artifact_path=artifact,
                            problem=problem,
                            abi=abi,
                        )
                    )

            with (
                _next(_Kind.DECOMPOSE, kernels[0]) as source_run,
                _next(_Kind.APPLY, kernels[1]) as held_out_run,
            ):
                self.assertNotEqual(source_run._run, held_out_run._run)
                self.assertEqual(
                    FakeRunner.instances[-2].read_roots,
                    (source_run._run / "input",),
                )
                self.assertEqual(
                    FakeRunner.instances[-1].read_roots,
                    (held_out_run._run / "input",),
                )

    def test_run_inputs_cannot_point_at_action_artifacts(self) -> None:
        problem = ProblemSpec("case", "Compute the case.")
        abi = KernelABI()
        context = TargetContext("case", "cuda", "sm120")
        source = self.problem_path.read_text(encoding="utf-8").strip()
        self.problem_path.write_text(source, encoding="utf-8")
        with (
            patch("klineage.action._sandbox._project_root", return_value=self.root),
            patch("klineage.action._sandbox.CodexRunner", FakeRunner),
            _new(_Kind.INIT, self.problem_path, self.repo) as sandbox,
        ):
            sandbox._save(problem, abi, context)
            target = sandbox._file("candidate.cu")
            (sandbox._work / target).write_text(source, encoding="utf-8")
            _, artifact = sandbox._take_file("candidate.cu", "candidate")
            manifest_path = sandbox._run / "run.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["problem_path"] = artifact.relative_to(sandbox._run).as_posix()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        kernel = Kernel(
            "candidate",
            source,
            context,
            artifact_path=artifact,
            problem=problem,
            abi=abi,
        )
        with (
            patch("klineage.action._sandbox._project_root", return_value=self.root),
            patch("klineage.action._sandbox.CodexRunner", FakeRunner),
            self.assertRaisesRegex(ValueError, "outside trusted input"),
            _next(_Kind.CODE_GEN, kernel),
        ):
            pass

    def test_output_targets_must_be_direct_children(self) -> None:
        with (
            patch("klineage.action._sandbox._project_root", return_value=self.root),
            patch("klineage.action._sandbox.CodexRunner", FakeRunner),
            _new(_Kind.INIT, self.problem_path, self.repo) as sandbox,
        ):
            with self.assertRaisesRegex(ValueError, "direct child"):
                sandbox._file("nested/candidate.cu")
            with self.assertRaisesRegex(ValueError, "direct child"):
                sandbox._dir("nested/submission")

    def test_bundle_member_cannot_pose_as_a_committed_artifact(self) -> None:
        problem = ProblemSpec("case", "Compute the case.")
        abi = KernelABI()
        context = TargetContext("case", "cuda", "sm120")
        interface = abi.interface
        files = {
            interface.module: "def load():\n    return lambda x: x\n",
            "kernel.cu": CUDA_SOURCE,
        }
        with (
            patch("klineage.action._sandbox._project_root", return_value=self.root),
            patch("klineage.action._sandbox.CodexRunner", FakeRunner),
            _new(_Kind.INIT, self.problem_path, self.repo) as sandbox,
        ):
            sandbox._save(problem, abi, context)
            output = sandbox._dir("submission")
            for relative, source in files.items():
                path = sandbox._work / output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source, encoding="utf-8")
            _, artifact = sandbox._take_bundle(output, "candidate", interface)

        disguised = Kernel(
            "candidate",
            CUDA_SOURCE,
            context,
            artifact_path=artifact / "kernel.cu",
            problem=problem,
            abi=abi,
        )
        with (
            patch("klineage.action._sandbox._project_root", return_value=self.root),
            patch("klineage.action._sandbox.CodexRunner", FakeRunner),
            self.assertRaisesRegex(ValueError, "outside an action sandbox"),
            _next(_Kind.CODE_GEN, disguised),
        ):
            pass


if __name__ == "__main__":
    unittest.main()

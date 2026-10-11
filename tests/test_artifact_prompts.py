import json
import re
import tempfile
import unittest
from contextlib import chdir
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jinja2 import UndefinedError
from kernel_fixtures import edit_kernel, kernel, write_kernel

from klineage.backend.cuda import CudaKernel
from klineage.artifact import ACTION_TYPES, Problem, save_artifact
from klineage.artifact.kernel import KERNEL, kernel_layouts
from klineage.artifact.tree import ACTION_PREFIX
from klineage.backend import backends, get_backend
from klineage.memory import SkillCard
from klineage.artifact.problem import load_problem
from klineage.prompts import action_spec, render_prompt
from klineage.serialization import raw_text, to_json


class ArtifactPromptTests(unittest.TestCase):
    def test_layouts_are_queried(self):
        current = json.loads(to_json(kernel()))
        contexts = {
            "generation": {"problem": load_problem(kernel())},
            "init": {
                "problem": "/problem",
                "repository": "/repo",
                "expert_kernel": "expert.cu",
            },
            "decompose": {"input_kernel": current},
            "apply": {"current_kernel": current, "memory": None, "exclude_skills": ()},
        }
        for stage, context in contexts.items():
            with self.subTest(stage=stage):
                prompt = render_prompt(stage, **context)
                self.assertIn("action_spec(", prompt)
                self.assertNotIn(raw_text(action_spec(stage)), prompt)
                self.assertNotIn(".klineage/specs", prompt)
                self.assertNotIn(".klineage/message", prompt)
                self.assertNotIn("verifier", prompt.lower())
                self.assertNotIn("verify_", prompt)
                self.assertNotIn("harnessuate", prompt)
                self.assertNotIn("workloads.Read", prompt)

    def test_generation_problem_input(self):
        prompt = render_prompt("generation", problem=load_problem(kernel()))
        self.assertIn("klineage.evaluate(", prompt)
        for unrelated in (
            "Current kernel:", "Memory:", "reference=current_kernel",
            "For edits, preserve", "For an unchanged result",
        ):
            self.assertNotIn(unrelated, prompt)
        self.assertEqual(tuple(child.name for child in action_spec("generation").children), ("kernel",))

    def test_generation_python_entry(self):
        problem = load_problem(kernel()).replace(language="python")
        prompt = render_prompt("generation", problem=problem)
        self.assertIn("Python host entries", prompt)
        self.assertIn("kernel_layouts(backend.kernel_type)", prompt)
        self.assertNotIn("Retain the configured callable", prompt)
        self.assertNotIn("native language", prompt)

    def test_memory_is_optional(self):
        current = json.loads(to_json(kernel()))
        for memory in (None, "", " \t "):
            prompt = render_prompt(
                "apply", current_kernel=current, memory=memory, exclude_skills=()
            )
            self.assertNotIn("\nMemory:", prompt)
        prompt = render_prompt(
            "apply", current_kernel=current, memory="/memory", exclude_skills=("a",)
        )
        self.assertIn('Memory: "/memory"', prompt)
        self.assertIn('Exclude skills: ["a"]', prompt)

    def test_missing_input_fails(self):
        with self.assertRaises((UndefinedError, TypeError)):
            render_prompt("apply", memory=None)

    def test_python_bundle_contract(self):
        config = """[solution]
name = "python"
definition = "gemm"
author = "test"
[build]
language = "python"
entry_point = "main.py::run"
"""
        current = edit_kernel(kernel(), source_files={
                "config.toml": config,
                "solution/main.py": "def run(a, b): return a @ b",
            })
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("klineage.backend.detect_backend", side_effect=AssertionError("local device probe")),
            patch("klineage.artifact.problem.trace_module", side_effect=AssertionError("reference executed")),
        ):
            root = Path(temporary)
            directory = write_kernel(current, root / "versions/0")
            manifest = save_artifact(current, root / "input.json")
            for value in (json.loads(to_json(current)), str(directory), str(root), str(manifest)):
                for stage in ("apply", "decompose"):
                    with self.subTest(stage=stage, input=type(value).__name__):
                        context = (
                            {"current_kernel": value, "memory": None, "exclude_skills": ()}
                            if stage == "apply" else {"input_kernel": value}
                        )
                        prompt = render_prompt(stage, **context)
                        self.assertIn("kernel_layouts(backend.kernel_type)", prompt)

    def test_missing_kernel_path_renders(self):
        with tempfile.TemporaryDirectory() as temporary:
            prompt = render_prompt(
                "apply", current_kernel=str(Path(temporary) / "missing"),
                memory=None, exclude_skills=(),
            )
        self.assertIn("get_backend(problem.language, problem.platform)", prompt)

    def test_backend_example_runs(self):
        prompt = render_prompt("backend")
        examples = re.findall(r"```python\n(.*?)\n```", prompt, re.DOTALL)
        self.assertEqual(len(examples), 1)
        problem = load_problem(kernel())
        scope = {}
        with (
            patch("klineage.problem", return_value=problem),
            patch("klineage.info", return_value={"device": "evaluation target"}) as info,
            patch("klineage.backend.detect_backend", side_effect=AssertionError("local device probe")),
            patch("builtins.print") as display,
        ):
            exec(examples[0], scope)
        backend = get_backend(problem.language, problem.platform)
        self.assertIs(scope["backend"], backend)
        info.assert_called_once_with()
        display.assert_any_call(info.return_value)
        display.assert_any_call(raw_text(backend.kernel_type.kernel_spec()))
        display.assert_any_call(backend.skill_root / "SKILL.md")

    def test_cuda_skill_example(self):
        document = (get_backend("cuda").skill_root / "cuda.md").read_text()
        self.run_service_examples(document.split("## Local checks")[0])

    def test_cuda_local_example(self):
        document = (get_backend("cuda").skill_root / "cuda.md").read_text()
        section = document.split("## Local checks")[1].split("## Performance inspection")[0]
        examples = re.findall(r"```python\n(.*?)\n```", section, re.DOTALL)
        self.assertEqual(len(examples), 1)
        scope = {"inputs": (2, 3)}
        with patch("klineage.artifact.Kernel.load_kernel", return_value=lambda x, y: x + y) as load:
            exec(examples[0], scope)
        load.assert_called_once_with(build_root=Path("current_kernel"))
        self.assertEqual(scope["outputs"], 5)

    def run_service_examples(self, document):
        from kernel_fixtures import accepted
        from test_kernel_build import SOURCES

        from klineage.artifact import load_artifact

        examples = re.findall(r"```python\n(.*?)\n```", document, re.DOTALL)

        def measure(root, **kwargs):
            candidate = load_artifact(root, artifact_type=CudaKernel)
            measured = edit_kernel(candidate, validation=accepted())
            return SimpleNamespace(kernel=measured, accepted=True, diagnostics={}, validation=accepted())

        with tempfile.TemporaryDirectory() as directory, chdir(directory):
            current = kernel()
            problem_dir = Path(json.loads(current.find("metadata.json").content)["problem"]["path"])
            Path("problems").mkdir()
            Path("problems/gemm").symlink_to(problem_dir, target_is_directory=True)
            for name, text in SOURCES.items():
                path = Path("current_kernel/submission") / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            scope = {"reference": None, "problem_id": "gemm"}
            with (
                patch("klineage.problem", return_value=load_problem(current)),
                patch("klineage.evaluate", side_effect=measure) as evaluate,
                patch("klineage.backend.cuda.CudaKernel.build_kernel",
                      side_effect=AssertionError("service example compiled locally")),
            ):
                for example in examples:
                    exec(example, scope)
                evaluate.assert_called_once_with(Path("current_kernel"), reference=None)
            self.assertEqual(scope["reply"].kernel.validation, accepted())
            self.assertIsNone(scope["reply"].kernel.function)
            self.assertFalse(Path(".klineage/kernel.json").exists())

    def test_service_measurement_path(self):
        root = Path(__file__).parents[1] / "src/klineage"
        paths = [root / "prompts/templates" / f"{name}.j4" for name in ("backend", "generation", "apply", "init", "decompose")]
        paths += [root / "harness/resources/bench/SKILL.md"]
        paths += [root / "backend" / name / "skills/SKILL.md" for name in ("cuda", "hip", "ascend")]
        paths += [root / "backend/cuda/skills/cuda.md"]
        paths += list((root / "backend").glob("*/prompts/*.j4"))
        for path in paths:
            with self.subTest(path=path):
                document = path.read_text()
                self.assertNotIn("Client(", document)
                self.assertNotIn("klineage.harness.evaluate", document)
                self.assertNotIn("from klineage.harness import evaluate", document)
                self.assertNotIn("metadata[\"validation\"] = result", document)
                for removed in (
                    "klineage.tools.profile", "ProfileOptions", "ncu_command",
                    "The agent has no GPU", "The agent has no accelerator",
                    "must not compile", "Never build locally", "do not invoke NCU",
                    "Remote profiling is unsupported",
                ):
                    self.assertNotIn(removed, document)
        bench = (root / "harness/resources/bench/SKILL.md").read_text()
        self.assertIn("klineage.evaluate(", bench)
        self.assertIn("klineage.result(", bench)
        self.assertIn("metadata.json", bench)
        self.assertNotIn("evaluations/", bench)
        self.assertIn("workspace/", bench)
        self.assertIn("ncu", bench)
        self.assertIn("counter permissions", bench)

    def test_contract_catalog(self):
        layouts = [
            KERNEL, Problem.problem_spec(), SkillCard.schema,
            *(action_spec(kind.removeprefix(ACTION_PREFIX)) for kind in ACTION_TYPES),
            *(layout for backend in backends() for layout in kernel_layouts(backend.kernel_type)),
        ]
        for layout in layouts:
            described = raw_text(layout)
            self.assertIn(layout.description.splitlines()[0], described)
            for child in layout.children:
                self.assertIn(child.name, described)
        root = Path(__file__).parents[1] / "src/klineage"
        self.assertFalse(list((root / "message").glob("*.md")))
        self.assertFalse(list((root / "prompts").rglob("verify*.j4")))

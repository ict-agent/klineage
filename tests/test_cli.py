import importlib
import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from kernel_fixtures import kernel

from klineage.constants import MAX_APPLY_STEPS

# Required Init inputs shared by CLI cases.
INPUTS = [
    "--problem",
    "problem.py",
    "--repo",
    "https://example.test/expert",
    "--expert-kernel",
    "expert.cu",
]


class CliTests(unittest.TestCase):
    def test_defaults_and_results(self):
        current = kernel()
        cases = (
            ("workflow", [], current, current.to_dict(), True),
            (
                "init_memory",
                ["--memory-dir", "memory"],
                (Path("memory/one/SKILL.md"),),
                ["memory/one/SKILL.md"],
                False,
            ),
        )
        for name, extra, result, expected, verifier in cases:
            module = importlib.import_module(f"klineage.cli.{name}")
            output = io.StringIO()
            with (
                self.subTest(command=name),
                patch.object(module, name, return_value=result) as run,
                redirect_stdout(output),
            ):
                module.main(INPUTS + extra)
            self.assertEqual(json.loads(output.getvalue()), expected)
            run.assert_called_once()
            values = run.call_args.kwargs
            self.assertEqual(str(values["problem"]), "problem.py")
            self.assertEqual(values["repo"], "https://example.test/expert")
            self.assertEqual(values["expert_kernel"], "expert.cu")
            self.assertIs(values["enable_verifier"], verifier)

    def test_option_overrides(self):
        options = [
            "--max-decompose-step",
            "2",
            "--workdir",
            "work",
            "--timeout",
            "20",
            "--max-retries",
            "0",
        ]
        cases = (
            ("workflow", ["--no-verifier", "--max-apply-step", "1"], kernel(), False),
            ("init_memory", ["--verifier", "--memory-dir", "memory"], (), True),
        )
        for name, extra, result, verifier in cases:
            module = importlib.import_module(f"klineage.cli.{name}")
            with (
                self.subTest(command=name),
                patch.object(module, name, return_value=result) as run,
                redirect_stdout(io.StringIO()),
            ):
                module.main(INPUTS + options + extra)
            values = run.call_args.kwargs
            self.assertEqual(values["max_decompose_step"], 2)
            self.assertEqual(values["workdir"], Path("work"))
            self.assertEqual(values["timeout"], 20)
            self.assertEqual(values["max_retries"], 0)
            self.assertIs(values["enable_verifier"], verifier)
            if name == "workflow":
                self.assertEqual(values["max_apply_step"], 1)
            else:
                self.assertEqual(values["memory_dir"], Path("memory"))

    def test_missing_inputs_fail(self):
        for name, arguments in (
            ("workflow", []),
            ("init_memory", INPUTS),
            ("optimize", ["--start_kernel", "start"]),
        ):
            module = importlib.import_module(f"klineage.cli.{name}")
            with (
                self.subTest(command=name),
                patch.object(module, name) as run,
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as failure,
            ):
                module.main(arguments)
            self.assertEqual(failure.exception.code, 2)
            run.assert_not_called()

    def test_optimize_inputs(self):
        module = importlib.import_module("klineage.cli.optimize")
        current = kernel()
        arguments = [
            "--start_kernel",
            "start/kernel.json",
            "--memory_dir",
            "memory",
            "--workdir",
            "work",
        ]
        for flags, verifier, steps in (
            ([], True, MAX_APPLY_STEPS),
            (["--no-verifier", "--max-apply-step", "2"], False, 2),
        ):
            output = io.StringIO()
            with (
                self.subTest(flags=flags),
                patch.object(module, "optimize", return_value=current) as run,
                redirect_stdout(output),
            ):
                module.main(arguments + flags)
            self.assertEqual(json.loads(output.getvalue()), current.to_dict())
            values = run.call_args.kwargs
            self.assertEqual(values["start_kernel"], Path("start/kernel.json"))
            self.assertEqual(values["memory_dir"], "memory")
            self.assertEqual(values["workdir"], Path("work"))
            self.assertIs(values["enable_verifier"], verifier)
            self.assertEqual(values["max_apply_step"], steps)

    def test_baseline_memory_option(self):
        module = importlib.import_module("klineage.cli.optimize")
        arguments = ["--start_kernel", "start", "--workdir", "work"]
        for value in (None, "", " \t "):
            extra = [] if value is None else ["--memory_dir", value]
            with (
                self.subTest(memory=value),
                patch.object(module, "optimize", return_value=kernel()) as run,
                redirect_stdout(io.StringIO()),
            ):
                module.main(arguments + extra)
            self.assertEqual(run.call_args.kwargs["memory_dir"], value)

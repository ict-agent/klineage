from __future__ import annotations

import json
import shutil
import tempfile
import time
import unittest
from dataclasses import fields
from pathlib import Path

from klineage.harness import CodexRunner, CodexRunnerError, CodexRunResult

FAKE_CODEX = '''#!/usr/bin/python3
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

args = sys.argv[1:]
prompt = sys.stdin.read()
output_path = pathlib.Path(args[args.index("--output-last-message") + 1])

print(
    json.dumps(
        {
            "args": args,
            "path": os.environ["PATH"],
            "tmpdir": os.environ["TMPDIR"],
        }
    ),
    flush=True,
)
if prompt.startswith("PROCESS_TREE:"):
    pid_path = prompt.removeprefix("PROCESS_TREE:")
    child_code = """
import os
import pathlib
import signal
import subprocess
import sys
import time

grandchild = subprocess.Popen([
    sys.executable,
    "-c",
    "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
])
pathlib.Path(sys.argv[1]).write_text(
    f"{os.getpid()} {grandchild.pid}",
    encoding="utf-8",
)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(60)
"""
    subprocess.Popen([sys.executable, "-c", child_code, pid_path])
    time.sleep(60)
if prompt == "TIMEOUT":
    time.sleep(60)
if prompt == "FAIL":
    print("simulated failure", file=sys.stderr)
    raise SystemExit(7)
if prompt.startswith("SYMLINK:"):
    output_path.unlink()
    output_path.symlink_to(prompt.removeprefix("SYMLINK:"))
    raise SystemExit(0)

output_path.write_text("answer: " + prompt, encoding="utf-8")
'''

FAKE_BWRAP = """#!/bin/sh
if [ "$1" = "--help" ]; then
    echo "--as-pid-1 --argv0 --perms --ro-bind-fd"
fi
exit 0
"""

PROCESS_WAIT_SECONDS = 2.0


class CodexRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sandbox = self.root / "sandbox"
        self.fake_codex = self._executable("fake-codex", FAKE_CODEX)
        self.fake_bwrap = self._executable("fake-bwrap", FAKE_BWRAP)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _executable(self, name: str, source: str) -> Path:
        path = self.root / name
        path.write_text(source, encoding="utf-8")
        path.chmod(0o755)
        return path

    def runner(self, **kwargs: object) -> CodexRunner:
        return CodexRunner(
            self.sandbox,
            codex_bin=self.fake_codex,
            bwrap_bin=self.fake_bwrap,
            **kwargs,
        )

    def test_success_persists_minimal_trace_bundle(self) -> None:
        result = self.runner()("build the thing", run_id="successful-run")

        self.assertEqual(
            [field.name for field in fields(CodexRunResult)],
            ["final_message", "trace_path"],
        )
        self.assertEqual(result.final_message, "answer: build the thing")
        self.assertEqual(result.trace_path.parent.name, "successful-run")
        self.assertEqual(
            result.trace_path.parent.parent,
            (self.sandbox / ".klineage" / "codex-runs").resolve(),
        )
        self.assertEqual(
            (result.trace_path.parent / "prompt.txt").read_text(),
            "build the thing",
        )
        self.assertTrue((result.trace_path.parent / "final_message.txt").is_file())
        self.assertTrue((result.trace_path.parent / "stderr.log").is_file())
        self.assertFalse((result.trace_path.parent / "metadata.json").exists())

    def test_nonzero_exit_raises_with_trace_and_stderr(self) -> None:
        with self.assertRaises(CodexRunnerError) as caught:
            self.runner()("FAIL", run_id="failed-run")

        error = caught.exception
        self.assertIsNotNone(error.trace_path)
        assert error.trace_path is not None
        self.assertTrue(error.trace_path.exists())
        self.assertIn("exit code 7", str(error))
        self.assertIn("simulated failure", str(error))

    def test_rejects_a_replaced_final_message_file(self) -> None:
        outside = self.root / "outside.txt"
        outside.write_text("outside", encoding="utf-8")

        with self.assertRaisesRegex(CodexRunnerError, "replaced"):
            self.runner()(f"SYMLINK:{outside}", run_id="replaced-output")

    def test_constructor_timeout_raises_with_trace(self) -> None:
        with self.assertRaises(CodexRunnerError) as caught:
            self.runner(timeout=0.05)("TIMEOUT", run_id="timed-out-run")

        error = caught.exception
        self.assertIsNotNone(error.trace_path)
        self.assertIn("timed out", str(error))

    def test_rejects_empty_prompt_and_unsafe_run_id(self) -> None:
        runner = self.runner()

        with self.assertRaises(ValueError):
            runner("   ")
        with self.assertRaises(ValueError):
            runner("prompt", run_id="../escape")

    def test_state_dir_stays_inside_sandbox(self) -> None:
        result = self.runner(state_dir="runner-state")(
            "inspect",
            run_id="custom-state",
        )

        self.assertEqual(
            result.trace_path.parent.parent,
            (self.sandbox / "runner-state" / "codex-runs").resolve(),
        )
        with self.assertRaises(ValueError):
            self.runner(state_dir=self.root / "outside")

    def test_command_is_strict_offline_and_sets_reasoning(self) -> None:
        low = self.runner(reasoning_effort="low")(
            "inspect",
            run_id="low-reasoning",
        )
        inherited = self.runner(reasoning_effort=None)(
            "inspect",
            run_id="inherited-reasoning",
        )
        low_args = self._trace_args(low.trace_path)
        inherited_args = self._trace_args(inherited.trace_path)

        self.assertIn("never", low_args)
        self.assertIn("--strict-config", low_args)
        self.assertIn("--ignore-user-config", low_args)
        self.assertIn("--ignore-rules", low_args)
        self.assertIn("--ephemeral", low_args)
        self.assertIn('model_reasoning_effort="low"', low_args)
        self.assertFalse(
            any(value.startswith("model_reasoning_effort=") for value in inherited_args)
        )
        self.assertTrue(
            any("network = { enabled = false }" in value for value in low_args)
        )
        self.assertNotIn("--yolo", low_args)

    def test_command_grants_only_declared_read_roots(self) -> None:
        source = self.root / "input"
        source.mkdir()
        result = self.runner(read_roots=(source,))(
            "inspect",
            run_id="read-roots",
        )
        args = self._trace_args(result.trace_path)
        profile = next(value for value in args if str(source.resolve()) in value)

        self.assertIn(f'"{source.resolve()}" = "read"', profile)

        with self.assertRaises(ValueError):
            self.runner(read_roots=(self.root,))

    def test_rejects_invalid_reasoning_effort(self) -> None:
        with self.assertRaises(ValueError):
            self.runner(reasoning_effort="extreme")

    def test_launcher_contains_only_required_tools(self) -> None:
        runner = self.runner()
        launcher = runner._launcher_dir
        result = runner("inspect", run_id="launcher-path")
        event = self._trace_event(result.trace_path)

        self.assertFalse(launcher.is_relative_to(self.sandbox))
        self.assertEqual(
            {path.name for path in launcher.iterdir()},
            {"apply_patch", "applypatch", "bwrap"},
        )
        self.assertEqual((launcher / "bwrap").resolve(), self.fake_bwrap.resolve())
        self.assertEqual(
            (launcher / "apply_patch").resolve(),
            self.fake_codex.resolve(),
        )
        self.assertEqual(event["path"].split(":")[0], str(launcher))
        self.assertEqual(
            Path(event["tmpdir"]),
            (self.sandbox / ".klineage" / "tmp").resolve(),
        )

    def test_resolves_native_executable_from_npm_launcher(self) -> None:
        package = self.root / "npm"
        launcher = package / "bin" / "codex.js"
        native = (
            package
            / "node_modules"
            / "@openai"
            / "codex-linux-x64"
            / "vendor"
            / "x86_64-unknown-linux-musl"
            / "bin"
            / "codex"
        )
        launcher.parent.mkdir(parents=True)
        native.parent.mkdir(parents=True)
        launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        launcher.chmod(0o755)
        shutil.copyfile("/bin/true", native)
        native.chmod(0o755)

        runner = CodexRunner(
            self.sandbox,
            codex_bin=launcher,
            bwrap_bin=self.fake_bwrap,
        )

        self.assertEqual(runner._codex_program, native.resolve())

    def test_bwrap_must_expose_required_features(self) -> None:
        incompatible = self._executable(
            "old-bwrap",
            "#!/bin/sh\necho --as-pid-1\n",
        )

        with self.assertRaises(CodexRunnerError):
            CodexRunner(
                self.sandbox,
                codex_bin=self.fake_codex,
                bwrap_bin=incompatible,
            )

    def test_codex_must_be_executable(self) -> None:
        self.fake_codex.chmod(0o644)

        with self.assertRaises(CodexRunnerError):
            CodexRunner(
                self.sandbox,
                codex_bin=self.fake_codex,
                bwrap_bin=self.fake_bwrap,
            )

    def test_timeout_kills_descendant_processes(self) -> None:
        pid_path = self.root / "process-tree.pid"
        prompt = f"PROCESS_TREE:{pid_path}"

        with self.assertRaises(CodexRunnerError):
            self.runner(timeout=0.2)(prompt, run_id="process-tree")

        child_pid, grandchild_pid = map(int, pid_path.read_text().split())
        deadline = time.monotonic() + PROCESS_WAIT_SECONDS
        while time.monotonic() < deadline:
            if not self._alive(child_pid) and not self._alive(grandchild_pid):
                break
            time.sleep(0.02)

        self.assertFalse(self._alive(child_pid))
        self.assertFalse(self._alive(grandchild_pid))

    def test_close_removes_launcher(self) -> None:
        runner = self.runner()
        launcher = runner._launcher_dir

        with runner:
            self.assertTrue(launcher.exists())

        self.assertFalse(launcher.exists())
        runner.close()

    @staticmethod
    def _trace_args(trace_path: Path) -> list[str]:
        return CodexRunnerTests._trace_event(trace_path)["args"]

    @staticmethod
    def _trace_event(trace_path: Path) -> dict[str, object]:
        return json.loads(trace_path.read_text().splitlines()[0])

    @staticmethod
    def _alive(pid: int) -> bool:
        stat_path = Path("/proc") / str(pid) / "stat"
        try:
            state = stat_path.read_text().split()[2]
        except FileNotFoundError:
            return False
        return state != "Z"


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from klineage.harness import CodexRunner, CodexRunnerError
from klineage.harness.codex_runner import PACKAGE_ROOT, SKILL_ROOT

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
            "tmpdir": os.environ.get("TMPDIR"),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "skills": {
                name: (pathlib.Path(".agents/skills") / name / "SKILL.md").read_text()
                for name in ("bench", "cuda")
            },
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
if prompt == "RUNTIME":
    subprocess.run([sys.executable, "-c", "import klineage"], check=True)
    assert pathlib.Path(".klineage/message/init.md").is_file()
if prompt == "NO_FINAL":
    output_path.unlink()
    raise SystemExit(0)
if prompt == "FAIL":
    print("simulated failure", file=sys.stderr)
    raise SystemExit(7)

output_path.write_text("answer: " + prompt, encoding="utf-8")
'''


PROCESS_WAIT_SECONDS = 2.0


class CodexRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.work = self.root / "work"
        self.fake_codex = self.executable("fake-codex", FAKE_CODEX)

    def tearDown(self):
        self.temp.cleanup()

    def executable(self, name: str, source: str) -> Path:
        path = self.root / name
        path.write_text(source, encoding="utf-8")
        path.chmod(0o755)
        return path

    def runner(self, **kwargs: object) -> CodexRunner:
        return CodexRunner(
            self.work,
            codex_bin=self.fake_codex,
            **kwargs,
        )

    def test_success_persists_minimal_trace_bundle(self):
        result = self.runner()("build the thing", run_id="successful-run")

        self.assertEqual(result.final_message, "answer: build the thing")
        self.assertEqual(result.trace_path.parent.name, "successful-run")
        self.assertEqual(
            result.trace_path.parent.parent,
            (self.work / ".klineage" / "codex-runs").resolve(),
        )
        self.assertEqual(
            (result.trace_path.parent / "prompt.txt").read_text(),
            "build the thing",
        )
        self.assertTrue((result.trace_path.parent / "final_message.txt").is_file())
        self.assertTrue((result.trace_path.parent / "stderr.log").is_file())
        self.assertFalse((result.trace_path.parent / "metadata.json").exists())

    def test_nonzero_exit_raises_with_trace_and_stderr(self):
        with self.assertRaises(CodexRunnerError) as caught:
            self.runner()("FAIL", run_id="failed-run")

        error = caught.exception
        self.assertIsNotNone(error.trace_path)
        assert error.trace_path is not None
        self.assertTrue(error.trace_path.exists())
        self.assertIn("exit code 7", str(error))
        self.assertIn("simulated failure", str(error))

    def test_constructor_timeout_raises_with_trace(self):
        with self.assertRaises(CodexRunnerError) as caught:
            self.runner(timeout=0.05)("TIMEOUT", run_id="timed-out-run")

        error = caught.exception
        self.assertIsNotNone(error.trace_path)
        self.assertIn("timed out", str(error))

    def test_rejects_empty_prompt_and_unsafe_run_id(self):
        runner = self.runner()

        with self.assertRaises(ValueError):
            runner("   ")
        with self.assertRaises(ValueError):
            runner("prompt", run_id="../escape")

    def test_rejects_invalid_reasoning_effort(self):
        with self.assertRaises(ValueError):
            self.runner(reasoning_effort="extreme")

    def test_codex_must_be_executable(self):
        self.fake_codex.chmod(0o644)

        with self.assertRaises(CodexRunnerError):
            CodexRunner(
                self.work,
                codex_bin=self.fake_codex,
            )

    def test_timeout_kills_descendant_processes(self):
        pid_path = self.root / "process-tree.pid"
        prompt = f"PROCESS_TREE:{pid_path}"

        with self.assertRaises(CodexRunnerError):
            self.runner(timeout=0.2)(prompt, run_id="process-tree")

        child_pid, grandchild_pid = map(int, pid_path.read_text().split())
        deadline = time.monotonic() + PROCESS_WAIT_SECONDS
        while time.monotonic() < deadline:
            if not self.alive(child_pid) and not self.alive(grandchild_pid):
                break
            time.sleep(0.02)

        self.assertFalse(self.alive(child_pid))
        self.assertFalse(self.alive(grandchild_pid))

    def test_missing_reply_has_trace(self):
        with self.assertRaises(CodexRunnerError) as caught:
            self.runner()("NO_FINAL", run_id="missing-reply")
        self.assertTrue(caught.exception.trace_path.is_file())

    def test_interrupt_stops_process(self):
        runner = self.runner()
        for error in (KeyboardInterrupt(), SystemExit()):
            process = Mock()
            process.communicate.side_effect = (error, None)
            with (
                self.subTest(error=type(error).__name__),
                patch(
                    "klineage.harness.codex_runner.subprocess.Popen",
                    return_value=process,
                ),
                patch("klineage.harness.codex_runner.stop_process") as stop,
                self.assertRaises(type(error)),
            ):
                runner("interrupt")
            stop.assert_called_once_with(process)

    def test_local_command(self):
        result = self.runner(reasoning_effort="low")("inspect", run_id="local")
        args = self.trace_args(result.trace_path)
        self.assertEqual(args[args.index("-s") + 1], "danger-full-access")
        self.assertIn('model_reasoning_effort="low"', args)
        self.assertIn("--ignore-user-config", args)
        self.assertFalse(any("permissions." in arg for arg in args))

    def test_external_trace_directory(self):
        directory = self.root / "traces"
        result = self.runner(state_dir=directory)("inspect", run_id="external")
        self.assertEqual(result.trace_path.parent.parent, directory / "codex-runs")

    def test_process_skill_access(self):
        result = self.runner()("inspect", run_id="skills")
        event = self.trace_event(result.trace_path)
        for name in ("bench", "cuda"):
            self.assertEqual(
                event["skills"][name], (SKILL_ROOT / name / "SKILL.md").read_text()
            )
        source = self.work / ".agents/skills/cuda/assets/add_one/solution/kernel.cu"
        self.assertTrue(source.is_file())

    def test_native_skill_discovery(self):
        self.runner()
        for name in ("cuda", "hip", "ascendc"):
            with self.subTest(name=name):
                resource = self.work / ".agents/skills" / name
                self.assertTrue(resource.is_symlink())
                self.assertTrue((resource / "SKILL.md").is_file())
                self.assertEqual(resource.resolve(), SKILL_ROOT / name)

    def test_reuses_skills_across_runs(self):
        self.runner()("first", run_id="first")
        result = self.runner()("retry", run_id="retry")
        self.assertEqual(result.final_message, "answer: retry")
        self.assertTrue((self.work / ".agents/skills/bench").is_symlink())

    def test_process_runtime_access(self):
        with patch.dict(os.environ, {"PYTHONPATH": str(self.root / "existing")}):
            result = self.runner()("RUNTIME", run_id="runtime")
            self.assertEqual(os.environ["PYTHONPATH"], str(self.root / "existing"))
        paths = self.trace_event(result.trace_path)["pythonpath"].split(os.pathsep)
        self.assertEqual(paths, [str(PACKAGE_ROOT.parent), str(self.root / "existing")])

    def test_message_conflict_preserved(self):
        message = self.work / ".klineage/message"
        message.mkdir(parents=True)
        (message / "keep.md").write_text("User contract")
        with self.assertRaisesRegex(
            CodexRunnerError, "workspace resource already exists"
        ):
            self.runner()
        self.assertEqual((message / "keep.md").read_text(), "User contract")

    def test_skill_conflict_preserved(self):
        skill = self.work / ".agents/skills/cuda"
        skill.mkdir(parents=True)
        contents = "User skill"
        (skill / "SKILL.md").write_text(contents)
        with self.assertRaisesRegex(
            CodexRunnerError, "workspace resource already exists"
        ):
            self.runner()
        self.assertEqual((skill / "SKILL.md").read_text(), contents)

    @staticmethod
    def trace_args(trace_path: Path) -> list[str]:
        return CodexRunnerTests.trace_event(trace_path)["args"]

    @staticmethod
    def trace_event(trace_path: Path) -> dict[str, object]:
        return json.loads(trace_path.read_text().splitlines()[0])

    @staticmethod
    def alive(pid: int) -> bool:
        stat_path = Path("/proc") / str(pid) / "stat"
        try:
            state = stat_path.read_text().split()[2]
        except FileNotFoundError:
            return False
        return state != "Z"


if __name__ == "__main__":
    unittest.main()

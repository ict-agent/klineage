from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import yaml
from kernel_fixtures import skill

from klineage.constants import MEMORY_DIRECTORY
from klineage.harness import CodexRunner, RunnerError
from klineage.memory import save_skill

FAKE_CODEX = '''#!/usr/bin/python3
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

args = sys.argv[1:]
received_prompt = sys.stdin.read()
prompt = received_prompt.splitlines()[2]
output_path = pathlib.Path(args[args.index("--output-last-message") + 1])
memory = pathlib.Path("workspace/.agents/skills/memory")

print(
    json.dumps(
        {
            "args": args,
            "prompt": received_prompt,
            "path": os.environ["PATH"],
            "tmpdir": os.environ.get("TMPDIR"),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "home": os.environ.get("HOME"),
            "codex_home": os.environ.get("CODEX_HOME"),
            "agents": pathlib.Path("workspace/AGENTS.md").read_text(),
            "memory": {
                path.relative_to(memory).as_posix(): path.read_text()
                for path in memory.rglob("SKILL.md")
            },
            "skills": {
                name: (pathlib.Path("workspace/.agents/skills") / name / "SKILL.md").read_text()
                for name in ("bench", "cuda", "hip", "ascendc")
            },
        }
    ),
    flush=True,
)
if prompt.startswith(("DETACHED_TREE:", "ORPHAN_EXIT:")):
    pid_path = prompt.partition(":")[2]
    child_code = """
import pathlib
import subprocess
import sys

grandchild = subprocess.Popen(
    [sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)", sys.argv[1]],
    start_new_session=True,
)
pathlib.Path(sys.argv[1]).write_text(str(grandchild.pid))
"""
    child = subprocess.Popen(
        [sys.executable, "-c", child_code, pid_path], start_new_session=True,
    )
    child.wait()
    if prompt.startswith("DETACHED_TREE:"):
        time.sleep(60)
    deadline = time.monotonic() + 5
    while not pathlib.Path(pid_path + ".observed").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
if prompt == "TIMEOUT":
    time.sleep(60)
if prompt.startswith("ISOLATION:"):
    secret = pathlib.Path(prompt.removeprefix("ISOLATION:"))
    output_path.write_text(json.dumps({
        "host_secret": secret.exists(),
        "host_key": os.environ.get("KLINEAGE_TEST_HOST_KEY"),
        "devices": [str(path) for path in pathlib.Path("/dev").glob("nvidia*")],
    }))
    raise SystemExit(0)
if prompt.startswith("STATS:"):
    from klineage.harness.logging import record
    record("probe", marker=prompt.removeprefix("STATS:"))
if prompt == "FAIL":
    print("simulated failure", file=sys.stderr)
    raise SystemExit(7)

output_path.write_text("answer: " + prompt, encoding="utf-8")
'''.replace("#!/usr/bin/python3", f"#!{sys.executable}")


PROCESS_WAIT_SECONDS = 2.0


class CodexRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.codex_home = self.root / "codex-home"
        self.codex_home.mkdir()
        self.enterContext(patch.dict(os.environ, {"CODEX_HOME": str(self.codex_home)}))
        for name in ("KLINEAGE_MODEL", "KLINEAGE_API_BASE_URL", "KLINEAGE_API_KEY_ENV"):
            os.environ.pop(name, None)
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

    def test_persists_trace_bundle(self):
        result = self.runner()("build the thing", run_id="successful-run")

        self.assertEqual(result.final_message, "answer: build the thing")
        self.assertEqual(result.trace_path.parent.name, "successful-run")
        self.assertEqual(
            result.trace_path.parent.parent,
            (self.work / "workspace/.klineage" / "agent-runs").resolve(),
        )
        prompt = (result.trace_path.parent / "prompt.txt").read_text()
        self.assertTrue(prompt.startswith("Read `workspace/AGENTS.md`"))
        self.assertIn("`workspace/.agents/skills/`", prompt)
        self.assertIn("non-artifact files under `workspace/`", prompt)
        self.assertTrue(prompt.endswith("build the thing"))
        self.assertTrue((result.trace_path.parent / "final_message.txt").is_file())
        self.assertTrue((result.trace_path.parent / "stderr.log").is_file())
        self.assertEqual((result.trace_path.parent / "stats.jsonl").read_text(), "")

    def test_attempt_stats(self):
        runner = self.runner()
        first = runner("STATS:first", run_id="stats-first")
        runner.resume_id = "recorded-session"
        second = runner("STATS:retry", run_id="stats-retry")
        for result, marker in ((first, "first"), (second, "retry")):
            events = (result.trace_path.parent / "stats.jsonl").read_text().splitlines()
            self.assertEqual([json.loads(line)["marker"] for line in events], [marker])
        self.assertFalse((runner.state_dir / "stats").exists())

    def test_error_retains_trace(self):
        with self.assertRaises(RunnerError) as caught:
            self.runner()("FAIL", run_id="failed-run")

        error = caught.exception
        self.assertIsNotNone(error.trace_path)
        assert error.trace_path is not None
        self.assertTrue(error.trace_path.exists())
        self.assertIn("exit code 7", str(error))
        self.assertIn("simulated failure", str(error))

    def test_timeout_retains_trace(self):
        with self.assertRaises(RunnerError) as caught:
            self.runner(timeout=0.05)("TIMEOUT", run_id="timed-out-run")

        error = caught.exception
        self.assertIsNotNone(error.trace_path)
        self.assertIn("timed out", str(error))

    def test_agent_isolation(self):
        secret = self.root / "host-secret"
        secret.write_text("outside the agent workspace")
        with patch.dict(os.environ, {"KLINEAGE_TEST_HOST_KEY": "private"}):
            result = self.runner()(f"ISOLATION:{secret}")
        self.assertEqual(json.loads(result.final_message), {
            "host_secret": False, "host_key": None, "devices": [],
        })

    def test_timeout_kills_orphans(self):
        self.check_tree_cleanup("DETACHED_TREE", 1)

    def check_tree_cleanup(self, mode, count):
        runner = self.runner(timeout=PROCESS_WAIT_SECONDS)
        pid_path = self.work / "process-tree.pid"
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        pids = set()
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(runner, f"{mode}:{pid_path}")
                deadline = time.monotonic() + PROCESS_WAIT_SECONDS
                while time.monotonic() < deadline:
                    pids = self.host_pids(str(pid_path))
                    if pid_path.exists() and len(pids) >= count:
                        break
                    time.sleep(0.01)
                self.assertTrue(pid_path.exists(), "children did not initialize")
                self.assertGreaterEqual(len(pids), count, "children were never observed on host")
                Path(str(pid_path) + ".observed").touch()
                if mode == "ORPHAN_EXIT":
                    future.result(timeout=PROCESS_WAIT_SECONDS + 1)
                else:
                    with self.assertRaisesRegex(RunnerError, "timed out"):
                        future.result(timeout=PROCESS_WAIT_SECONDS + 1)

            # Namespace PIDs in the file cannot be used with host kill()/proc.
            self.assertEqual(len(pid_path.read_text().split()), count)
            deadline = time.monotonic() + PROCESS_WAIT_SECONDS
            while any(self.alive(pid) for pid in pids) and time.monotonic() < deadline:
                time.sleep(0.02)

            self.assertFalse(self.host_pids(str(pid_path)))
            self.assertFalse(any(self.alive(pid) for pid in pids))
            self.assertIsNone(unrelated.poll())
        finally:
            for pid in self.host_pids(str(pid_path)):
                os.kill(pid, signal.SIGKILL)
            unrelated.terminate()
            unrelated.wait()

    @staticmethod
    def host_pids(marker):
        found = set()
        for directory in Path("/proc").iterdir():
            if not directory.name.isdecimal():
                continue
            try:
                args = (directory / "cmdline").read_bytes().split(b"\0")
            except (OSError, ProcessLookupError):
                continue
            if marker.encode() in args:
                found.add(int(directory.name))
        return found

    def test_local_command(self):
        result = self.runner(reasoning_effort="low")("inspect", run_id="local")
        args = self.trace_args(result.trace_path)
        self.assertEqual(args[args.index("-s") + 1], "danger-full-access")
        self.assertIn('model_reasoning_effort="low"', args)
        self.assertIn("--ignore-user-config", args)
        self.assertFalse(any("permissions." in arg for arg in args))

    def test_codex_config_new_resume(self):
        (self.codex_home / "config.toml").write_text(
            'model = "configured-model"\n'
            'model_reasoning_effort = "high"\n'
            'model_context_window = 200000\n'
            'model_provider = "test"\n'
            '[model_providers.test]\n'
            'base_url = "https://provider.invalid/v1"\n'
            'env_key = "TEST_PROVIDER_KEY"\n'
        )
        with patch.dict(os.environ, {"TEST_PROVIDER_KEY": "host-secret"}):
            for session in (None, "recorded-session"):
                with self.subTest(session=session):
                    runner = self.runner(resume_id=session)
                    args = runner.command(self.work / "reply.txt")
                    self.assertIn("configured-model", args)
                    self.assertIn('model_reasoning_effort="high"', args)
                    self.assertIn("model_context_window=200000", args)
                    self.assertIn("https://provider.invalid/v1", runner.api_base_url)
                    self.assertNotIn("host-secret", " ".join(args))

    def test_process_memory_mount(self):
        memory = self.root / "memory"
        card = memory / "gemm/staging/SKILL.md"
        card.parent.mkdir(parents=True)
        save_skill(skill("staging"), card)
        contents = card.read_text()
        runner = self.runner()
        mounted = runner.mount_memory(memory)
        self.assertEqual(mounted, self.work / MEMORY_DIRECTORY)
        self.assertFalse(mounted.is_symlink())
        self.assertEqual(runner.mount_memory(memory), mounted)

        for index, process in enumerate((runner, self.runner())):
            result = process("inspect memory", run_id=f"memory-{index}")
            files = self.trace_event(result.trace_path)["memory"]
            self.assertEqual(len(files), 1)
            header = yaml.safe_load(next(iter(files.values())).split("---", 2)[1])
            self.assertEqual(header["metadata"]["skill_id"], "staging")
        self.assertEqual(card.read_text(), contents)
        self.assertEqual(
            set(memory.rglob("*")), {card.parent.parent, card.parent, card}
        )

    def test_memory_conflict_preserved(self):
        first, second = self.root / "first", self.root / "second"
        first.mkdir()
        second.mkdir()
        card = first / "SKILL.md"
        save_skill(skill("existing"), card)
        contents = card.read_text()
        runner = self.runner()
        mounted = runner.mount_memory(first)
        with self.assertRaisesRegex(RunnerError, "already exists"):
            runner.mount_memory(second)
        self.assertTrue(mounted.is_dir())
        self.assertEqual(card.read_text(), contents)

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

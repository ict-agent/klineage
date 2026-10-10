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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import yaml
from kernel_fixtures import skill
from klineage.serialization import raw_text
from klineage.memory import save_skill
from klineage.artifact import ACTION_TYPES, ArtifactKind, Problem
from klineage.artifact.kernel import KERNEL, kernel_layouts
from klineage.artifact.tree import ACTION_PREFIX
from klineage.prompts import action_spec
from klineage.memory import SkillCard
from klineage.backend import get_backend

from klineage.harness import CodexRunner, CodexRunnerError
from klineage.harness.codex_runner import MEMORY_DIRECTORY, PACKAGE_ROOT, BENCH_SKILL

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
prompt = received_prompt.splitlines()[0]
output_path = pathlib.Path(args[args.index("--output-last-message") + 1])
memory = pathlib.Path(".agents/skills/memory")

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
            "agents": pathlib.Path("AGENTS.md").read_text(),
            "memory": {
                path.relative_to(memory).as_posix(): path.read_text()
                for path in memory.rglob("SKILL.md")
            },
            "skills": {
                name: (pathlib.Path(".agents/skills") / name / "SKILL.md").read_text()
                for name in ("bench", "cuda", "hip", "ascendc")
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
    sys.argv[1],
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
if prompt.startswith("SYMLINK_REPLY:"):
    output_path.symlink_to(prompt.removeprefix("SYMLINK_REPLY:"))
    raise SystemExit(0)
if prompt == "FIFO_REPLY":
    os.mkfifo(output_path)
    raise SystemExit(0)
if prompt.startswith("REDIRECT_STATE:"):
    import shutil
    state = pathlib.Path(".klineage/codex-runs")
    shutil.rmtree(state)
    state.symlink_to(prompt.removeprefix("REDIRECT_STATE:"), target_is_directory=True)
    output_path.write_text("replaced state")
    raise SystemExit(0)
if prompt.startswith("REDIRECT_FINAL:"):
    target = pathlib.Path(".klineage/codex-runs") / output_path.stem / "final_message.txt"
    target.unlink()
    target.symlink_to(prompt.removeprefix("REDIRECT_FINAL:"))
    output_path.write_text("redirected write")
    raise SystemExit(0)
if prompt.startswith("REDIRECT_SKILLS:"):
    import shutil
    shutil.rmtree(".agents")
    pathlib.Path(".agents").symlink_to(prompt.removeprefix("REDIRECT_SKILLS:"), target_is_directory=True)
    output_path.write_text("replaced skills")
    raise SystemExit(0)
if prompt.startswith("READONLY:"):
    reference = pathlib.Path(prompt.removeprefix("READONLY:"))
    value = reference.read_text()
    try:
        reference.write_text("overwrite")
        writable = True
    except OSError:
        writable = False
    output_path.write_text(json.dumps({"value": value, "writable": writable}))
    raise SystemExit(0)
if prompt == "STATE":
    state = pathlib.Path(os.environ["CODEX_HOME"]) / "resume-marker"
    count = int(state.read_text()) + 1 if state.exists() else 1
    state.write_text(str(count))
    output_path.write_text(json.dumps({
        "count": count, "home": os.environ["HOME"],
        "host_key": os.environ.get("KLINEAGE_TEST_HOST_KEY"),
        "devices": [str(path) for path in pathlib.Path("/dev").glob("nvidia*")],
    }))
    raise SystemExit(0)
if prompt == "RUNTIME":
    subprocess.run([sys.executable, "-c", "import klineage"], check=True)
    assert pathlib.Path(".klineage/specs/action.init.json").is_file()
if prompt == "NO_FINAL":
    output_path.unlink(missing_ok=True)
    raise SystemExit(0)
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
            (self.work / ".klineage" / "codex-runs").resolve(),
        )
        self.assertEqual(
            (result.trace_path.parent / "prompt.txt").read_text(),
            "build the thing",
        )
        self.assertTrue((result.trace_path.parent / "final_message.txt").is_file())
        self.assertTrue((result.trace_path.parent / "stderr.log").is_file())

    def test_error_retains_trace(self):
        with self.assertRaises(CodexRunnerError) as caught:
            self.runner()("FAIL", run_id="failed-run")

        error = caught.exception
        self.assertIsNotNone(error.trace_path)
        assert error.trace_path is not None
        self.assertTrue(error.trace_path.exists())
        self.assertIn("exit code 7", str(error))
        self.assertIn("simulated failure", str(error))

    def test_prompt_has_fresh_deadline(self):
        budget = 60
        runner = self.runner(timeout=budget)
        deadlines = []
        for run_id in ("first", "second"):
            started = datetime.now(UTC)
            result = runner("Build a kernel", run_id=run_id)
            prompt = (result.trace_path.parent / "prompt.txt").read_text()
            received = json.loads(result.trace_path.read_text().splitlines()[0])["prompt"]
            self.assertEqual(received, prompt)
            line = next(line for line in prompt.splitlines() if line.startswith("Deadline (UTC): "))
            deadline = datetime.fromisoformat(line.removeprefix("Deadline (UTC): "))
            self.assertGreaterEqual(deadline, started + timedelta(seconds=budget))
            self.assertLessEqual(deadline, datetime.now(UTC) + timedelta(seconds=budget))
            deadlines.append(deadline)
        self.assertGreater(deadlines[1], deadlines[0])

    def test_timeout_retains_trace(self):
        with self.assertRaises(CodexRunnerError) as caught:
            self.runner(timeout=0.05)("TIMEOUT", run_id="timed-out-run")

        error = caught.exception
        self.assertIsNotNone(error.trace_path)
        self.assertIn("timed out", str(error))

    def test_rejects_invalid_input(self):
        runner = self.runner()

        with self.assertRaises(ValueError):
            runner("   ")
        with self.assertRaises(ValueError):
            runner("prompt", run_id="../escape")

    def test_agent_isolation(self):
        secret = self.root / "host-secret"
        secret.write_text("outside the agent workspace")
        with patch.dict(os.environ, {"KLINEAGE_TEST_HOST_KEY": "private"}):
            result = self.runner()(f"ISOLATION:{secret}")
        self.assertEqual(json.loads(result.final_message), {
            "host_secret": False, "host_key": None, "devices": [],
        })

    def test_reply_symlink_rejected(self):
        secret = self.root / "host-secret"
        secret.write_text("must not leave the host")
        with self.assertRaises(CodexRunnerError):
            self.runner()(f"SYMLINK_REPLY:{secret}")
        self.assertEqual(secret.read_text(), "must not leave the host")

    def test_reply_fifo_rejected(self):
        with self.assertRaises(CodexRunnerError):
            self.runner()("FIFO_REPLY")

    def test_trace_tampering_rejected(self):
        secret = self.root / "host-file"
        secret.write_text("private")
        with self.assertRaises(CodexRunnerError):
            self.runner()(f"REDIRECT_FINAL:{secret}")

    def test_final_symlink_no_write(self):
        secret = self.root / "host-file"
        secret.write_text("must not be overwritten")
        try:
            self.runner()(f"REDIRECT_FINAL:{secret}")
        except CodexRunnerError:
            pass
        self.assertEqual(secret.read_text(), "must not be overwritten")

    def test_retry_symlink_no_write(self):
        outside = self.root / "host-directory"
        outside.mkdir()
        runner = self.runner()
        try:
            runner(f"REDIRECT_STATE:{outside}", run_id="first")
        except (CodexRunnerError, OSError):
            pass
        try:
            runner("second attempt", run_id="second")
        except (CodexRunnerError, OSError):
            pass
        self.assertEqual(list(outside.iterdir()), [])

    def test_memory_parent_no_write(self):
        outside = self.root / "outside"
        (outside / "skills").mkdir(parents=True)
        memory = self.root / "memory"
        memory.mkdir()
        save_skill(skill("source"), memory / "SKILL.md")
        runner = self.runner()
        runner(f"REDIRECT_SKILLS:{outside}")
        with self.assertRaises(OSError):
            runner.mount_memory(memory)
        self.assertEqual(list((outside / "skills").iterdir()), [])

    def test_rejects_invalid_effort(self):
        with self.assertRaises(ValueError):
            self.runner(reasoning_effort="extreme")

    def test_codex_must_be_executable(self):
        self.fake_codex.chmod(0o644)

        with self.assertRaises(CodexRunnerError):
            CodexRunner(
                self.work,
                codex_bin=self.fake_codex,
            )

    def test_timeout_kills_children(self):
        self.check_tree_cleanup("PROCESS_TREE", 2)

    def test_missing_reply_has_trace(self):
        with self.assertRaises(CodexRunnerError) as caught:
            self.runner()("NO_FINAL", run_id="missing-reply")
        self.assertTrue(caught.exception.trace_path.is_file())

    def test_timeout_kills_orphans(self):
        self.check_tree_cleanup("DETACHED_TREE", 1)

    def test_exit_kills_orphans(self):
        self.check_tree_cleanup("ORPHAN_EXIT", 1)

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
                    with self.assertRaisesRegex(CodexRunnerError, "timed out"):
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
            stop.assert_called_once()
            self.assertEqual(stop.call_args.args, (process,))
            self.assertEqual(Path(stop.call_args.kwargs["scope"]).parent, runner.trace_root)

    def test_local_command(self):
        result = self.runner(reasoning_effort="low")("inspect", run_id="local")
        args = self.trace_args(result.trace_path)
        self.assertEqual(args[args.index("-s") + 1], "danger-full-access")
        self.assertIn('model_reasoning_effort="low"', args)
        self.assertIn("--ignore-user-config", args)
        self.assertFalse(any("permissions." in arg for arg in args))

    def test_model_env_in_sandbox(self):
        with patch.dict(os.environ, {"KLINEAGE_MODEL": "provider/model-v1"}):
            for resume_id in (None, "recorded-session"):
                with self.subTest(resume_id=resume_id):
                    result = self.runner(resume_id=resume_id)("inspect")
                    args = self.trace_args(result.trace_path)
                    self.assertEqual(args.count("--model"), 1)
                    self.assertEqual(args[args.index("--model") + 1], "provider/model-v1")
                    subcommand = "resume" if resume_id else "exec"
                    self.assertGreater(args.index("--model"), args.index(subcommand))

    def test_model_arg_overrides_env(self):
        with patch.dict(os.environ, {"KLINEAGE_MODEL": "environment-model"}):
            runner = self.runner(model=" explicit-model ")
        args = runner.command(self.work / "reply.txt")
        self.assertEqual(args[args.index("--model") + 1], "explicit-model")

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

    def test_codex_config_overrides(self):
        (self.codex_home / "config.toml").write_text(
            'model = "configured-model"\nmodel_reasoning_effort = "high"\n'
        )
        with patch.dict(os.environ, {"KLINEAGE_MODEL": "environment-model"}):
            runner = self.runner(reasoning_effort="low")
            self.assertEqual(runner.model, "environment-model")
            self.assertEqual(runner.reasoning_effort, "low")
            self.assertEqual(self.runner(model="explicit-model").model, "explicit-model")

    def test_codex_config_is_host_only(self):
        config = self.codex_home / "config.toml"
        config.write_text(
            'model = "configured-model"\n'
            'model_provider = "test"\n'
            '[model_providers.test]\n'
            'base_url = "https://provider.invalid/v1"\n'
            'env_key = "KLINEAGE_TEST_HOST_KEY"\n'
        )
        with patch.dict(os.environ, {"KLINEAGE_TEST_HOST_KEY": "host-secret"}):
            result = self.runner()(f"ISOLATION:{config}")
        reply = json.loads(result.final_message)
        self.assertFalse(reply["host_secret"])
        self.assertEqual(reply["host_key"], "sandbox-provider")
        self.assertIn("configured-model", self.trace_args(result.trace_path))
        self.assertNotIn("host-secret", result.trace_path.read_text())
        self.assertFalse((self.work / ".klineage/codex-home/config.toml").exists())

    def test_missing_provider_key(self):
        with patch.dict(os.environ):
            os.environ.pop("UNSET_HOST_KEY", None)
            runner = self.runner(
                api_base_url="https://provider.invalid/v1", api_key_env="UNSET_HOST_KEY",
            )
        self.assertIn("exec", runner.command(self.work / "reply.txt"))
        with self.assertRaisesRegex(ValueError, "UNSET_HOST_KEY"):
            runner("inspect")

    def test_model_default(self):
        with patch.dict(os.environ):
            os.environ.pop("KLINEAGE_MODEL", None)
            runner = self.runner()
        self.assertNotIn("--model", runner.command(self.work / "reply.txt"))

    def test_empty_model_rejected(self):
        for model in ("", " "):
            with self.subTest(model=model):
                with self.assertRaisesRegex(ValueError, "model"):
                    self.runner(model=model)
                with patch.dict(os.environ, {"KLINEAGE_MODEL": model}):
                    with self.assertRaisesRegex(ValueError, "model"):
                        self.runner()

    def test_external_trace_directory(self):
        directory = self.root / "traces"
        result = self.runner(state_dir=directory)("inspect", run_id="external")
        self.assertEqual(result.trace_path.parent.parent, directory / "codex-runs")
        self.assertEqual((result.trace_path.parent / "final_message.txt").read_text(), result.final_message)
        args = self.trace_args(result.trace_path)
        self.assertEqual(Path(args[args.index("--output-last-message") + 1]).parent, self.work / ".klineage/replies")

    def test_relative_external_state(self):
        result = self.runner(state_dir=Path("../traces"))("external", run_id="relative")
        self.assertEqual(result.trace_path, self.root / "traces/codex-runs/relative/trace.jsonl")

    def test_external_traces_hidden(self):
        directory = self.root / "shared-traces"
        directory.mkdir()
        other_run = directory / "another-run-secret"
        other_run.write_text("another run's private trace")
        result = self.runner(state_dir=directory)(f"ISOLATION:{other_run}")
        self.assertFalse(json.loads(result.final_message)["host_secret"])

    def test_external_catalog_scope(self):
        directory = self.root / "shared-state"
        runner = self.runner(state_dir=directory)
        catalog = directory / "model.json"
        catalog.write_text("model catalog")
        runner.model_catalog = catalog
        result = runner(f"READONLY:{catalog}")
        self.assertEqual(json.loads(result.final_message), {"value": "model catalog", "writable": False})
        sibling = directory / "another-session"
        sibling.write_text("private")
        result = runner(f"ISOLATION:{sibling}")
        self.assertFalse(json.loads(result.final_message)["host_secret"])

    def test_declared_input_readonly(self):
        reference = self.root / "reference.txt"
        reference.write_text("mounted input")
        result = self.runner(read_only=(reference,))(f"READONLY:{reference}")
        self.assertEqual(json.loads(result.final_message), {"value": "mounted input", "writable": False})
        self.assertEqual(reference.read_text(), "mounted input")

    def test_resume_isolation(self):
        with patch.dict(os.environ, {"KLINEAGE_TEST_HOST_KEY": "private"}):
            first = self.runner()("STATE")
            second = self.runner(resume_id="recorded-session")("STATE")
        self.assertEqual(json.loads(first.final_message)["count"], 1)
        self.assertEqual(json.loads(second.final_message), {
            "count": 2, "home": "/home/agent", "host_key": None, "devices": [],
        })
        args = self.trace_args(second.trace_path)
        self.assertIn("resume", args)
        self.assertIn("recorded-session", args)
        self.assertEqual(self.trace_event(first.trace_path)["codex_home"], self.trace_event(second.trace_path)["codex_home"])

    def test_process_skill_access(self):
        result = self.runner()("inspect", run_id="skills")
        event = self.trace_event(result.trace_path)
        for name in ("bench", "cuda", "hip", "ascendc"):
            self.assertEqual(
                event["skills"][name], ((BENCH_SKILL if name == "bench" else get_backend(name).skill_root) / "SKILL.md").read_text()
            )
        source = self.work / ".agents/skills/cuda/assets/add_one/solution/kernel.cu"
        self.assertTrue(source.is_file())

    def test_process_function_docs(self):
        runner = self.runner()
        path = self.work / "AGENTS.md"
        path.write_text("# Local instructions\nKeep user guidance.\n")
        first = runner("inspect", run_id="first")
        docs = self.trace_event(first.trace_path)["agents"]
        self.assertIn("Keep user guidance.", docs)
        self.assertIn("from klineage.harness import evaluate", docs)
        self.assertIn("Signature: `evaluate(kernel:", docs)
        self.assertNotIn("### klineage.tools.retrieve\n", docs)

        path.write_text(docs.replace("klineage.harness.evaluate", "obsolete.evaluate"))
        second = runner("inspect", run_id="second")
        self.assertEqual(self.trace_event(second.trace_path)["agents"], docs)

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
        with self.assertRaisesRegex(CodexRunnerError, "already exists"):
            runner.mount_memory(second)
        self.assertTrue(mounted.is_dir())
        self.assertEqual(card.read_text(), contents)

    def test_memory_requires_directory(self):
        runner = self.runner()
        card = self.root / "SKILL.md"
        card.write_text("A file is not a memory directory")
        with self.assertRaises(NotADirectoryError):
            runner.mount_memory(card)
        with self.assertRaises(FileNotFoundError):
            runner.mount_memory(self.root / "missing")
        self.assertFalse((self.work / MEMORY_DIRECTORY).exists())

    def test_memory_none(self):
        runner = self.runner()
        self.assertIsNone(runner.mount_memory(None))
        self.assertFalse((self.work / MEMORY_DIRECTORY).exists())
        result = runner("inspect", run_id="no-memory")
        self.assertEqual(self.trace_event(result.trace_path)["memory"], {})

    def test_rejects_stale_memory(self):
        memory = self.root / "memory"
        memory.mkdir()
        card = memory / "SKILL.md"
        save_skill(skill("source"), card)
        contents = card.read_text()
        runner = self.runner()
        mounted = runner.mount_memory(memory)
        with self.assertRaisesRegex(ValueError, "fresh workdir"):
            runner.mount_memory(None)
        self.assertFalse(mounted.is_symlink())
        self.assertEqual(card.read_text(), contents)
        local_card = mounted / "SKILL.md"
        local_card.write_text("Keep local skill")
        with self.assertRaisesRegex(ValueError, "fresh workdir"):
            runner.mount_memory(None)
        self.assertEqual(local_card.read_text(), "Keep local skill")

    def test_rejects_recursive_memory(self):
        runner = self.runner()
        mounted = self.work / MEMORY_DIRECTORY
        mounted.mkdir()
        for source in (self.root, self.work, mounted.parent, mounted):
            with (
                self.subTest(source=source),
                self.assertRaisesRegex(ValueError, "contain its workspace mount"),
            ):
                runner.mount_memory(source)
        self.assertTrue(mounted.is_dir())
        self.assertFalse(mounted.is_symlink())

    def test_process_runtime_access(self):
        with patch.dict(os.environ, {"PYTHONPATH": str(self.root / "existing")}):
            result = self.runner()("RUNTIME", run_id="runtime")
            self.assertEqual(os.environ["PYTHONPATH"], str(self.root / "existing"))
        paths = self.trace_event(result.trace_path)["pythonpath"].split(os.pathsep)
        self.assertEqual(paths, [str(PACKAGE_ROOT.parent)])

    def test_schema_exports(self):
        self.runner()
        schemas = [
            (Problem.artifact_kind, (Problem.problem_spec(),)),
            (SkillCard.artifact_kind, (SkillCard.schema,)),
            (ArtifactKind.KERNEL, (KERNEL,)),
            *((kind, (action_spec(kind.removeprefix(ACTION_PREFIX)),)) for kind in ACTION_TYPES),
            *((get_backend(language).kernel_type.artifact_kind, kernel_layouts(get_backend(language).kernel_type)) for language in ("cuda", "hip", "ascendc")),
        ]
        for kind, layouts in schemas:
            path = self.work / ".klineage/specs" / f"{kind}.json"
            self.assertEqual(json.loads(path.read_text()), [json.loads(raw_text(layout)) for layout in layouts])
        self.assertFalse((self.work / ".klineage/artifacts").exists())
        self.assertFalse((self.work / ".klineage/message").exists())

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

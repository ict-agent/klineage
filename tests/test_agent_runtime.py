"""Operator runtime settings survive sandbox launches and action retries."""

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from kernel_fixtures import kernel

from klineage.action import Apply, Decompose, Init
from klineage.action.action import input_paths
from klineage.artifact import load_problem
from klineage.harness import AgentRuntime, CodexRunner
from klineage.harness.bridge import RELAY_PORT


RUN_TIMEOUT = 15


class AgentRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.work = self.root / "work"
        self.work.mkdir()
        self.venv = self.root / "agent-python"
        self.venv.symlink_to(sys.prefix, target_is_directory=True)
        self.python = self.venv / "bin" / "python"
        self.toolchain = self.root / "toolchain"
        self.toolchain.mkdir()
        (self.toolchain / "runtime_marker.py").write_text("VALUE = 'operator supplied'\n")
        self.context = self.root / "context.json"
        self.context.write_text("trusted context")
        self.secret = self.root / "host-secret"
        self.secret.write_text("host only")
        self.enterContext(patch.dict(os.environ, {
            "CODEX_HOME": str(self.root / "codex-home"),
            "KLINEAGE_SERVER": "", "KLINEAGE_API_BASE_URL": "",
            "KLINEAGE_API_KEY_ENV": "", "AIPING_API_KEY": "host-provider-key",
            "HOST_ONLY_VALUE": "host only",
        }))

    def runner(self, **kwargs):
        return CodexRunner(self.work, codex_bin="/usr/bin/true", **kwargs)

    def execute(self, runner, source):
        program = [str(runner.agent.python), "-c", source]
        with patch.object(runner, "command", return_value=program):
            with runner.execution(self.work / "reply.txt", self.work / "run") as (command, environment):
                result = subprocess.run(
                    command, env=environment, capture_output=True, text=True,
                    timeout=RUN_TIMEOUT,
                )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout), command

    def test_default_is_device_free(self):
        runner = self.runner()
        self.assertEqual(runner.agent.device_nodes, ())
        result, command = self.execute(runner, """
import json,os,pathlib
assert not list(pathlib.Path('/dev').glob('nvidia*'))
assert not pathlib.Path('/dev/dri').exists()
assert not pathlib.Path('/dev/davinci0').exists()
assert 'AIPING_API_KEY' not in os.environ
assert 'HOST_ONLY_VALUE' not in os.environ
print(json.dumps({'home': os.environ['HOME']}))
""")
        self.assertEqual(result["home"], "/home/agent")
        self.assertNotIn("--dev-bind", command)

    def test_explicit_runtime_launch(self):
        agent = AgentRuntime(
            python=self.python, device_nodes=(Path("/dev/null"),),
            read_only=(self.toolchain,),
            environment={
                "PYTHONPATH": str(self.toolchain), "PATH": "/usr/local/cuda/bin:/usr/bin:/bin",
                "LD_LIBRARY_PATH": str(self.toolchain), "CUDA_VISIBLE_DEVICES": "0",
                "RUNTIME_SETTING": "explicit",
            },
        )
        runner = self.runner(agent=agent, read_only=(self.context,))
        result, command = self.execute(runner, f"""
import errno,json,os,pathlib,sys
import runtime_marker
assert runtime_marker.VALUE == 'operator supplied'
assert pathlib.Path({str(self.context)!r}).read_text() == 'trusted context'
assert not pathlib.Path({str(self.secret)!r}).exists()
assert 'AIPING_API_KEY' not in os.environ
assert 'HOST_ONLY_VALUE' not in os.environ
assert os.environ['RUNTIME_SETTING'] == 'explicit'
assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'
assert os.environ['LD_LIBRARY_PATH'] == {str(self.toolchain)!r}
for target in (pathlib.Path({str(self.context)!r}), pathlib.Path(runtime_marker.__file__)):
    try:
        target.write_text('changed')
    except OSError as error:
        assert error.errno == errno.EROFS
    else:
        raise AssertionError('runtime dependency is writable')
print(json.dumps({{'python':sys.executable,'path':os.environ['PATH']}}))
""")
        self.assertEqual(result["python"], str(self.python))
        self.assertTrue(result["path"].startswith(str(self.python.parent) + ":"))
        self.assertIn("/usr/local/cuda/bin", result["path"])
        device = command.index("--dev-bind")
        self.assertEqual(command[device:device + 3], ["--dev-bind", "/dev/null", "/dev/null"])

    def test_relay_uses_agent_python(self):
        authorizations = []

        class Provider(BaseHTTPRequestHandler):
            def do_POST(self):
                authorizations.append(self.headers.get("Authorization"))
                self.rfile.read(int(self.headers["Content-Length"]))
                data = b'{"ok":true}'
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            runner = self.runner(
                agent=AgentRuntime(python=self.python),
                api_base_url=f"http://127.0.0.1:{server.server_port}",
            )
            result, command = self.execute(runner, """
import json,os,sys,urllib.request
assert os.environ['AIPING_API_KEY'] == 'sandbox-provider'
request=urllib.request.Request('http://127.0.0.1:RELAY_PORT/responses', data=b'{}')
with urllib.request.urlopen(request, timeout=5) as response:
    assert json.load(response) == {'ok':True}
print(json.dumps({'python':sys.executable}))
""".replace("RELAY_PORT", str(RELAY_PORT)))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        entry = command.index("--") + 1
        self.assertEqual(command[entry:entry + 3], [str(self.python), "-m", "klineage.harness.bridge"])
        self.assertEqual(result["python"], str(self.python))
        self.assertEqual(authorizations, ["Bearer host-provider-key"])
        self.assertNotIn("host-provider-key", " ".join(command))

    def test_private_env_rejected(self):
        for name in (
            "HOME", "TMPDIR", "CODEX_HOME", "KLINEAGE_SERVER", "KLINEAGE_PROBLEM",
            "KLINEAGE_PROCESS_SCOPE", "AIPING_API_KEY",
            "LD_PRELOAD", "LD_AUDIT", "PYTHONHOME", "PYTHONSTARTUP",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                AgentRuntime(environment={name: "not allowed"})

    def test_invalid_environment(self):
        for value in ({"INVALID-NAME": "x"}, {"SETTING": "x\0y"}, {"SETTING": 1}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                AgentRuntime(environment=value)

    def test_environment_is_frozen(self):
        supplied = {"RUNTIME_SETTING": "original"}
        agent = AgentRuntime(environment=supplied)
        supplied["RUNTIME_SETTING"] = "changed"
        self.assertEqual(agent.environment["RUNTIME_SETTING"], "original")
        with self.assertRaises(TypeError):
            agent.environment["RUNTIME_SETTING"] = "changed"
        self.assertNotIn("original", repr(agent))

    def test_venv_path_is_preserved(self):
        agent = AgentRuntime(python=self.python)
        self.assertEqual(agent.python, self.python)
        self.assertNotEqual(agent.python, self.python.resolve())
        with self.assertRaisesRegex(ValueError, "executable"):
            AgentRuntime(python=self.root / "missing-python")

    def test_devices_are_not_inferred(self):
        agent = AgentRuntime(environment={"CUDA_VISIBLE_DEVICES": "2"})
        runner = self.runner(agent=agent)
        result, command = self.execute(runner, """
import json,os,pathlib
assert os.environ['CUDA_VISIBLE_DEVICES'] == '2'
assert not list(pathlib.Path('/dev').glob('nvidia*'))
print(json.dumps({'devices':[]}))
""")
        self.assertEqual(result, {"devices": []})
        self.assertNotIn("--dev-bind", command)

    def test_action_resume_keeps_agent(self):
        agent = AgentRuntime(
            python=self.python, device_nodes=(Path("/dev/null"),),
            read_only=(self.toolchain,), environment={"RUNTIME_SETTING": "explicit"},
        )
        current = kernel()
        trace = self.work / "trace.jsonl"
        trace.write_text(json.dumps({"type": "thread.started", "thread_id": "same-session"}) + "\n")
        with patch("klineage.action.action.CodexRunner") as factory:
            actions = (
                Apply(current, workdir=self.work, agent=agent),
                Decompose(current, workdir=self.work, agent=agent),
                Init(load_problem(current), self.root, "expert.cu", workdir=self.work, agent=agent),
            )
            for action in actions:
                with self.subTest(action=type(action).__name__):
                    self.assertIs(action.agent, agent)
                    action.read_only = input_paths(current)
                    original = action.runner
                    original.read_only = action.read_only
                    action.resume_from(trace)
                    self.assertIs(action.runner, original)
                    self.assertIs(factory.call_args.kwargs["agent"], agent)
                    self.assertEqual(action.runner.read_only, input_paths(current))
                    self.assertEqual(action.runner.resume_id, "same-session")
            self.assertEqual(factory.call_count, len(actions))

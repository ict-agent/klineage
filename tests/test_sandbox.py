from __future__ import annotations

import importlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from kernel_fixtures import make_kernel
from problem_fixtures import problem_spec

from klineage.artifact import load_problem
from klineage.backend import get_backend
from klineage.harness import AgentRuntime, CodexRunner
from klineage.harness.eval import WorkerError
from klineage.harness.sandbox import Sandbox
from klineage.serialization import to_json
from klineage.server.worker import WorkerRuntime

PROCESS_WAIT_SECONDS = 5
POLL_SECONDS = 0.02


class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "work"
        self.workspace.mkdir()
        self.secret = self.root / "host-secret"
        self.secret.write_text("private")
        self.reference = self.root / "reference"
        self.reference.write_text("readable")

    def sandbox(self, **kwargs):
        return Sandbox(self.workspace, **kwargs)

    def run_code(self, source, sandbox=None):
        sandbox = sandbox or self.sandbox()
        return subprocess.run(
            sandbox.command(["/usr/bin/python3", "-c", source]),
            env=sandbox.environment,
            capture_output=True,
            text=True,
            timeout=PROCESS_WAIT_SECONDS,
            check=True,
        )

    def test_no_bwrap_no_fallback(self):
        with patch("klineage.harness.sandbox.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "bubblewrap"):
                self.sandbox()

    def test_reject_broad_mounts(self):
        for path in (Path("/"), self.root, Path.home(), Path("/proc"), Path("/dev")):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    self.sandbox(read_only=(path,))

    def test_scoped_writable_mount(self):
        state = self.root / "state"
        state.mkdir()
        self.run_code(f"""
from pathlib import Path
Path({str(state / 'session')!r}).write_text('resumable')
assert not Path({str(self.secret)!r}).exists()
""", self.sandbox(writable=(state,)))
        self.assertEqual((state / "session").read_text(), "resumable")

    def test_workspace_read_only(self):
        source = f"""
from pathlib import Path
Path('written').write_text('yes')
reference = Path({str(self.reference)!r})
assert reference.read_text() == 'readable'
try:
    reference.write_text('changed')
except PermissionError:
    pass
except OSError as error:
    import errno
    assert error.errno == errno.EROFS
else:
    raise AssertionError('read-only mount was writable')
"""
        self.run_code(source, self.sandbox(read_only=(self.reference,)))
        self.assertEqual((self.workspace / "written").read_text(), "yes")
        self.assertEqual(self.reference.read_text(), "readable")

    def test_host_paths_hidden(self):
        sibling = self.root / "another-task"
        sibling.mkdir()
        (self.workspace / "escape").symlink_to(self.secret)
        source = f"""
from pathlib import Path
assert not Path({str(self.secret)!r}).exists()
assert not Path({str(sibling)!r}).exists()
assert not Path('escape').exists()
assert not Path({str(Path.home() / '.bashrc')!r}).exists()
assert not list(Path('/dev').glob('nvidia*'))
assert not Path('/dev/dri').exists()
assert not Path('/dev/davinci0').exists()
"""
        self.run_code(source)

    def test_private_home_and_tmp(self):
        source = """
import os
from pathlib import Path
assert os.environ['HOME'] == '/home/agent'
assert os.environ['TMPDIR'] == '/tmp'
for path in (Path.home() / 'private', Path('/tmp/private')):
    assert not path.exists()
    path.write_text('sandbox only')
"""
        sandbox = self.sandbox()
        self.run_code(source, sandbox)
        self.run_code(source, sandbox)

    def test_private_username(self):
        # TorchInductor names its cache with getpass without a mounted passwd file.
        self.run_code("import getpass; assert getpass.getuser() == 'agent'")

    def test_private_pid_namespace(self):
        source = f"""
import os
from pathlib import Path
assert os.getpid() < 10
assert not Path('/proc/{os.getpid()}').exists()
assert Path('/proc/1/comm').read_text().strip() == 'bwrap'
"""
        self.run_code(source)

    def test_no_environment_secrets(self):
        with patch.dict(os.environ, {"AIPING_API_KEY": "host-secret"}):
            sandbox = self.sandbox(environment={"KLINEAGE_TEST": "explicit"})
        self.assertNotIn("AIPING_API_KEY", sandbox.environment)
        self.assertNotIn("explicit", sandbox.command(["/usr/bin/true"]))
        self.run_code(
            "import os; assert 'AIPING_API_KEY' not in os.environ; "
            "assert os.environ['KLINEAGE_TEST'] == 'explicit'",
            sandbox,
        )

    def test_network_isolated(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            self.run_code(f"""
import socket
sock = socket.socket()
sock.settimeout(0.5)
assert sock.connect_ex(('127.0.0.1', {port})) != 0
assert socket.if_nameindex() == [(1, 'lo')]
""")

    def test_socket_allowlist(self):
        socket_path = self.root / "gateway.sock"
        with socket.socket(socket.AF_UNIX) as listener:
            listener.bind(str(socket_path))
            listener.listen()
            listener.settimeout(PROCESS_WAIT_SECONDS)
            messages = []

            def respond():
                connection, _ = listener.accept()
                with connection:
                    messages.append(connection.recv(32))
                    connection.sendall(b"reply")

            thread = threading.Thread(target=respond)
            thread.start()
            try:
                self.run_code(f"""
import socket
from pathlib import Path
sock = socket.socket(socket.AF_UNIX)
sock.connect({str(socket_path)!r})
sock.sendall(b'hello')
assert sock.recv(32) == b'reply'
assert not Path({str(self.secret)!r}).exists()
""", self.sandbox(sockets=(socket_path,)))
            finally:
                thread.join(PROCESS_WAIT_SECONDS)
            self.assertFalse(thread.is_alive())
            self.assertEqual(messages, [b"hello"])

    def wait_for(self, predicate):
        deadline = time.monotonic() + PROCESS_WAIT_SECONDS
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(POLL_SECONDS)
        self.fail("process did not reach the expected state")

    def child_pid(self, marker):
        for path in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                if marker.encode() in path.read_bytes().split(b"\0"):
                    return int(path.parent.name)
            except (OSError, ValueError):
                continue
        return None

    def process_exited(self, pid):
        try:
            state = Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0]
            return state == "Z"
        except (FileNotFoundError, ProcessLookupError):
            return True

    def detached_child(self):
        marker = uuid.uuid4().hex
        source = f"""
import subprocess, pathlib, time
subprocess.Popen(
    ['/usr/bin/python3', '-c', 'import time; time.sleep(60)', {marker!r}],
    start_new_session=True,
)
pathlib.Path('ready').write_text('yes')
while not pathlib.Path('release').exists():
    time.sleep(0.01)
"""
        sandbox = self.sandbox()
        process = subprocess.Popen(
            sandbox.command(["/usr/bin/python3", "-c", source]),
            env=sandbox.environment, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        self.addCleanup(lambda: process.poll() is None and process.kill())
        self.wait_for(lambda: (self.workspace / "ready").exists())
        return process, self.wait_for(lambda: self.child_pid(marker))

    def test_exit_cleans_children(self):
        process, child = self.detached_child()
        (self.workspace / "release").touch()
        process.communicate(timeout=PROCESS_WAIT_SECONDS)
        self.assertEqual(process.returncode, 0)
        self.wait_for(lambda: self.process_exited(child))

    def test_kill_cleans_children(self):
        process, child = self.detached_child()
        process.kill()
        process.communicate(timeout=PROCESS_WAIT_SECONDS)
        self.wait_for(lambda: self.process_exited(child))


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

    def test_private_env_rejected(self):
        for name in (
            "HOME", "TMPDIR", "CODEX_HOME", "KLINEAGE_SERVER", "KLINEAGE_PROBLEM",
            "KLINEAGE_PROCESS_SCOPE", "AIPING_API_KEY",
            "LD_PRELOAD", "LD_AUDIT", "PYTHONHOME", "PYTHONSTARTUP",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                AgentRuntime(environment={name: "not allowed"})


WORKER_TIMEOUT = 10


class WorkerSandboxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "job" / "work"
        self.build = self.workspace / "build"
        self.build.mkdir(parents=True)
        self.logs = self.root / "job" / "evaluations"
        self.logs.mkdir()
        self.secret = self.root / "job" / "result.json"
        self.secret.write_text("service-owned")
        self.fixed = self.root / "trusted" / "fixed.safetensors"
        self.fixed.parent.mkdir()
        self.fixed.write_bytes(b"fixed-input")
        problem = problem_spec()
        workload = problem.workloads[0]
        workload["inputs"]["a"] = {
            "type": "safetensors", "path": str(self.fixed), "tensor_key": "a",
        }
        self.kernel = make_kernel("isolated", problem.replace(workloads=(workload,)))
        self.request = {
            "operation": "evaluate", "kernel": json.loads(to_json(self.kernel)), "reference": None,
            "config": {"build_root": str(self.build), "include_paths": []},
        }

    def invoke(self, code, runtime=None):
        runtime = runtime or WorkerRuntime(timeout=WORKER_TIMEOUT)
        with patch("klineage.harness.eval.WORKER_ENTRY", code):
            return runtime.invoke(self.request, self.logs, "sandbox")

    def test_service_files_hidden(self):
        code = f"""
import errno,json,pathlib,sys
from klineage.artifact import Kernel,load_problem
from klineage.serialization import restore
request=json.load(sys.stdin)
problem=load_problem(restore(request['kernel'], Kernel))
assert not pathlib.Path({str(self.secret)!r}).exists()
assert not pathlib.Path({str(self.logs)!r}).exists()
assert not pathlib.Path({str(self.root / 'job' / 'candidate.json')!r}).exists()
pathlib.Path('local.txt').write_text('writable')
source=pathlib.Path(problem.workloads[0]['inputs']['a']['path'])
assert source.read_bytes() == b'fixed-input'
try:
    source.write_text('changed')
except OSError as error:
    assert error.errno == errno.EROFS
else:
    raise AssertionError('worker could modify trusted data')
try:
    (pathlib.Path.cwd().parent/'result.json').write_text('forged')
except OSError as error:
    assert error.errno == errno.EROFS
print(json.dumps({{'isolated': True}}))
"""
        self.assertEqual(self.invoke(code), {"isolated": True})
        self.assertEqual((self.workspace / "local.txt").read_text(), "writable")
        self.assertEqual(self.secret.read_text(), "service-owned")
        self.assertEqual(self.fixed.read_bytes(), b"fixed-input")
        record = json.loads((self.logs / "sandbox-process.json").read_text())
        self.assertEqual(Path(record["command"][0]).name, "bwrap")

    def test_credentials_not_inherited(self):
        with patch.dict(os.environ, {"AIPING_API_KEY": "sensitive", "LD_PRELOAD": "/host/hook.so"}):
            runtime = WorkerRuntime(environment={"WORKER_TEST_SETTING": "explicit"})
        result = self.invoke("""
import json,os
assert 'AIPING_API_KEY' not in os.environ
assert 'LD_PRELOAD' not in os.environ
assert os.environ['WORKER_TEST_SETTING'] == 'explicit'
print(json.dumps({'environment': 'clean'}))
""", runtime)
        self.assertEqual(result, {"environment": "clean"})
        for name in ("AIPING_API_KEY", "SERVER_TOKEN", "LD_PRELOAD"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                WorkerRuntime(environment={name: "disallowed"})

    def test_no_implicit_devices(self):
        result = self.invoke("""
import json,pathlib
assert not list(pathlib.Path('/dev').glob('nvidia*'))
assert not pathlib.Path('/dev/dri').exists()
assert not pathlib.Path('/dev/davinci0').exists()
print(json.dumps({'devices': 'none'}))
""")
        self.assertEqual(result, {"devices": "none"})

    def test_include_headers_readonly(self):
        headers = self.root / "headers"
        headers.mkdir()
        header = headers / "values.h"
        header.write_text("#define VALUE 7")
        self.request["config"]["include_paths"] = [str(headers)]
        self.assertEqual(self.invoke(f"""
import errno,json,pathlib
header=pathlib.Path({str(header)!r})
assert header.read_text() == '#define VALUE 7'
try:
    header.write_text('changed')
except OSError as error:
    assert error.errno == errno.EROFS
else:
    raise AssertionError('include path is writable')
print(json.dumps({{'headers': 'readonly'}}))
"""), {"headers": "readonly"})

    def test_plugin_in_host_and_worker(self):
        plugin_name = "worker_plugin_" + uuid.uuid4().hex
        directory = self.root / "plugins"
        directory.mkdir()
        plugin = directory / f"{plugin_name}.py"
        plugin.write_text(f"""
from klineage.backend import register_backend
from klineage.backend.cuda.kernel import CudaBackend,CudaKernel
class TestKernel(CudaKernel):
    artifact_kind={plugin_name!r}
class TestBackend(CudaBackend):
    kind={plugin_name!r}
    language={plugin_name!r}
    platform_prefixes=({plugin_name!r},)
    kernel_type=TestKernel
register_backend(TestBackend())
""")
        with patch.object(sys, "path", [str(directory), *sys.path]):
            importlib.invalidate_caches()
            runtime = WorkerRuntime(worker_plugins=(plugin_name,))
            from klineage.backend import get_backend
            self.assertEqual(get_backend(plugin_name).kind, plugin_name)
            result = self.invoke(f"""
import json,pathlib
from klineage.backend import get_backend
backend=get_backend({plugin_name!r})
assert not pathlib.Path({str(self.secret)!r}).exists()
print(json.dumps({{'backend': backend.kind}}))
""", runtime)
        self.assertEqual(result, {"backend": plugin_name})
        from klineage.backend.registry import REGISTERED_BACKENDS
        REGISTERED_BACKENDS[:] = [item for item in REGISTERED_BACKENDS if item.kind != plugin_name]
        sys.modules.pop(plugin_name, None)

    def test_timeout_cleans_children(self):
        marker = uuid.uuid4().hex
        code = f"""
import pathlib,subprocess,time
subprocess.Popen(['/usr/bin/python3','-c','import time; time.sleep(60)',{marker!r}], start_new_session=True)
pathlib.Path('started').touch()
time.sleep(60)
"""
        with self.assertRaisesRegex(WorkerError, "timed out"):
            self.invoke(code, WorkerRuntime(timeout=1))
        self.assertTrue((self.workspace / "started").exists())
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            alive = []
            for path in Path("/proc").glob("[0-9]*/cmdline"):
                try:
                    if marker.encode() in path.read_bytes().split(b"\0"):
                        alive.append(path)
                except OSError:
                    continue
            if not alive:
                break
            time.sleep(0.02)
        self.assertEqual(alive, [])
        record = json.loads((self.logs / "sandbox-process.json").read_text())
        self.assertTrue(record["timed_out"])
        self.assertEqual(Path(record["command"][0]).name, "bwrap")
        self.assertEqual(load_problem(self.kernel).name, "gemm")


class LocalRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.devices = self.root / "dev"
        self.devices.mkdir()
        self.toolkit = self.root / "vendor"
        (self.toolkit / "bin").mkdir(parents=True)
        self.compiler = self.toolkit / "bin" / "compiler"
        self.compiler.touch()
        self.compiler.chmod(0o755)
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.enterContext(
            patch("klineage.backend.runtime.DEVICE_ROOT", self.devices, create=True)
        )

    def node(self, name):
        path = self.devices / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to("/dev/null")
        return path

    def local(self, language="cuda", **info):
        backend = get_backend(language)
        metadata = {"available": True, "compiler": str(self.compiler)} | info
        with patch.object(type(backend), "info", return_value=metadata):
            return AgentRuntime.local(backend)

    def test_device_families(self):
        names = {
            "cuda": (
                "nvidia0",
                "nvidia2",
                "nvidiactl",
                "nvidia-uvm",
                "nvidia-uvm-tools",
                "nvidia-caps/nvidia-cap2",
            ),
            "hip": ("kfd", "dri/renderD128", "dri/renderD130"),
        }
        for group in names.values():
            for name in group:
                self.node(name)
        (self.devices / "nvidia9").touch()
        (self.devices / "davinci7").mkdir()
        self.node("unrelated")

        for language, expected in names.items():
            with self.subTest(language=language):
                runtime = self.local(language)
                self.assertEqual(
                    set(runtime.device_nodes),
                    {self.devices / name for name in expected},
                )
                self.assertEqual(runtime.python, Path(sys.executable).absolute())

    def test_visibility_and_secrets(self):
        self.node("nvidia2")
        with patch.dict(
            os.environ,
            {
                "CUDA_VISIBLE_DEVICES": "2",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "KLINEAGE_BACKEND": "ascend",
                "AIPING_API_KEY": "secret",
                "LD_PRELOAD": "/host/hook.so",
                "PYTHONPATH": "/host/private",
                "UNRELATED_SETTING": "private",
                "HOME": str(self.root),
            },
        ):
            runtime = self.local()
        self.assertEqual(
            dict(runtime.environment),
            {
                "CUDA_VISIBLE_DEVICES": "2",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "KLINEAGE_BACKEND": "cuda",
                "PATH": f"{self.compiler.parent}:/usr/bin:/bin",
            },
        )

    def test_compiler_is_on_path(self):
        self.node("nvidia0")
        for path in ("/usr/bin:/bin", f"{self.compiler.parent}:/usr/bin:/bin"):
            with self.subTest(path=path), patch.dict(os.environ, {"PATH": path}):
                runtime = self.local()
                search = runtime.environment["PATH"]
                self.assertEqual(shutil.which(self.compiler.name, path=search), str(self.compiler))
                self.assertEqual(search.split(os.pathsep).count(str(self.compiler.parent)), 1)
                self.assertTrue(search.endswith("/usr/bin:/bin"))

    def test_external_toolkit_paths(self):
        self.node("nvidia0")
        libraries = self.root / "vendor-libraries"
        libraries.mkdir()
        with patch.dict(
            os.environ,
            {
                "PATH": f"{self.toolkit / 'bin'}:/usr/bin:/bin",
                "LD_LIBRARY_PATH": str(libraries),
            },
        ):
            runtime = self.local()
        self.assertEqual(set(runtime.read_only), {self.toolkit, libraries})

    def test_skips_private_and_broad_paths(self):
        self.node("nvidia0")
        unsafe = (
            str(Path.home()),
            str(Path.cwd()),
            str(Path.cwd().parent),
            "/dev",
            "/proc",
            "/sys",
            "/",
            "/tmp",
            ".",
        )
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": ":".join(unsafe)}):
            runtime = self.local()
        self.assertEqual(runtime.read_only, (self.toolkit,))

    def test_unavailable_backend(self):
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            self.local(available=False)

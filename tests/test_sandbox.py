from __future__ import annotations

import errno
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

from klineage.harness.sandbox import Sandbox, runtime_paths


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

    def test_reject_broad_writable(self):
        for path in (Path("/"), self.root, Path.home(), Path("/proc"), Path("/dev")):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    self.sandbox(writable=(path,))

    def test_scoped_writable_mount(self):
        state = self.root / "state"
        state.mkdir()
        self.run_code(f"""
from pathlib import Path
Path({str(state / 'session')!r}).write_text('resumable')
assert not Path({str(self.secret)!r}).exists()
""", self.sandbox(writable=(state,)))
        self.assertEqual((state / "session").read_text(), "resumable")

    def test_reject_non_devices(self):
        with self.assertRaisesRegex(ValueError, "device"):
            self.sandbox(devices=(self.secret,))

    def test_reject_non_sockets(self):
        with self.assertRaisesRegex(ValueError, "socket"):
            self.sandbox(sockets=(self.secret,))

    def test_reject_host_home(self):
        with self.assertRaisesRegex(ValueError, "HOME"):
            self.sandbox(environment={"HOME": str(Path.home())})

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

    def test_workspace_read_only_child(self):
        protected = self.workspace / "reference"
        protected.write_text("trusted")
        sandbox = self.sandbox(read_only=(protected,))
        self.run_code("""
import errno
from pathlib import Path
try:
    Path('reference').write_text('changed')
except OSError as error:
    assert error.errno == errno.EROFS
else:
    raise AssertionError('workspace shadowed a read-only mount')
""", sandbox)

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

    def test_python_runtime_paths(self):
        sandbox = self.sandbox(read_only=runtime_paths(sys.executable))
        result = subprocess.run(
            sandbox.command([sys.executable, "-c", "import ssl,json; print('ready')"]),
            env=sandbox.environment,
            capture_output=True,
            text=True,
            timeout=PROCESS_WAIT_SECONDS,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "ready")

    def test_python_symlink_chain(self):
        executable = Path(sys.executable).resolve(strict=True)
        alias = self.root / "base-alias"
        alias.symlink_to(executable.parent.parent, target_is_directory=True)
        environment = self.root / "python-env"
        (environment / "bin").mkdir(parents=True)
        python = environment / "bin/python"
        python.symlink_to("python-middle")
        python.with_name("python-middle").symlink_to(alias / "bin" / executable.name)
        sandbox = self.sandbox(read_only=runtime_paths(python))
        result = subprocess.run(
            sandbox.command([str(python), "-c", "print('ready')"]),
            env=sandbox.environment, capture_output=True, text=True,
            timeout=PROCESS_WAIT_SECONDS, check=True,
        )
        self.assertEqual(result.stdout.strip(), "ready")

    def test_executable_symlink(self):
        binary = self.root / "binary"
        binary.symlink_to(shutil.which("true"))
        sandbox = self.sandbox(read_only=runtime_paths(sys.executable, binary))
        subprocess.run(
            sandbox.command([str(binary)]), env=sandbox.environment,
            check=True, timeout=PROCESS_WAIT_SECONDS,
        )

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

    def test_exit_observer_race(self):
        with patch.object(Path, "read_text", side_effect=ProcessLookupError(errno.ESRCH, "No such process")):
            self.assertTrue(self.process_exited(os.getpid()))
        with patch.object(Path, "read_text", side_effect=PermissionError(errno.EACCES, "Permission denied")):
            with self.assertRaises(PermissionError):
                self.process_exited(os.getpid())

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

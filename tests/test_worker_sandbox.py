from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from kernel_fixtures import make_kernel
from problem_fixtures import problem_spec

from klineage.artifact import load_problem
from klineage.harness.eval import WorkerError
from klineage.serialization import to_json
from klineage.server.worker import WorkerRuntime

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

    def test_relative_worker_python(self):
        runtime = WorkerRuntime(python=os.path.relpath(sys.executable))
        result = self.invoke("import json,sys; print(json.dumps({'prefix': sys.prefix}))", runtime)
        self.assertEqual(result, {"prefix": sys.prefix})

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

    def test_missing_inputs_rejected(self):
        self.fixed.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.invoke("print('{}')")

    def test_other_operations_rejected(self):
        self.request["operation"] = "inspect"
        with self.assertRaisesRegex(WorkerError, "evaluation and info requests only"):
            self.invoke("print('{}')")

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

"""Device descriptions come from the configured worker process and namespace."""

import importlib
import json
import sys
import sysconfig
import tempfile
import unittest
import uuid
import venv
from functools import partial
from pathlib import Path
from unittest.mock import patch

from problem_fixtures import problem_spec

from klineage.backend import get_backend
from klineage.server import Client, Service
from klineage.server.worker import WorkerRuntime


class ServerInfoTests(unittest.TestCase):
    def test_configured_worker_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = root / "python-env"
            venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
            python = environment / "bin/python"
            secret = root / "host-secret"
            secret.write_text("private")
            device = root / "selected-device"
            device.symlink_to("/dev/null")
            name = "info_backend_" + uuid.uuid4().hex
            (root / (name + ".py")).write_text(f"""
import os,sys,uuid
from pathlib import Path
from klineage.backend import register_backend
from klineage.backend.cuda import CudaBackend,CudaKernel
class InfoKernel(CudaKernel):
    artifact_kind = 'kernel.' + {name!r}
class InfoBackend(CudaBackend):
    kind = {name!r}
    language = {name!r}
    platform_prefixes = ({name!r},)
    kernel_type = InfoKernel
    def info(self):
        return {{'available': True, 'prefix': sys.prefix,
                'setting': os.environ.get('DEVICE_SETTING'),
                'device_visible': Path({str(device)!r}).is_char_device(),
                'host_secret_visible': Path({str(secret)!r}).exists(),
                'probe_id': uuid.uuid4().hex}}
register_backend(InfoBackend())
""")
            try:
                with patch.object(sys, "path", [str(root), *sys.path]):
                    importlib.invalidate_caches()
                    importlib.import_module(name)
                    factory = partial(
                        WorkerRuntime, python=python, worker_plugins=(name,),
                        device_nodes=(device,), read_only=(Path(sys.prefix),),
                        environment={"DEVICE_SETTING": "assigned", "PYTHONPATH": sysconfig.get_path("purelib")},
                    )
                    service = Service(
                        root / "service", get_backend(name),
                        {"task": problem_spec(language=name, platform=name)},
                        runtime_factory=factory,
                    )
                    with service.start() as running:
                        client = Client(running.endpoint, "task")
                        first = client.info()["device"]
                        second = client.info()["device"]
                    self.assertEqual(first["prefix"], str(environment))
                    self.assertEqual(first["setting"], "assigned")
                    self.assertTrue(first["device_visible"])
                    self.assertFalse(first["host_secret_visible"])
                    self.assertEqual(first["probe_id"], second["probe_id"])
                    record = json.loads((root / "service/info/evaluations/info-process.json").read_text())
                    self.assertEqual(Path(record["command"][0]).name, "bwrap")
                    self.assertIn(str(python), record["command"])
            finally:
                from klineage.backend.registry import REGISTERED_BACKENDS

                REGISTERED_BACKENDS[:] = [item for item in REGISTERED_BACKENDS if item.kind != name]
                sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main()

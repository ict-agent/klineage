from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from kernel_fixtures import make_kernel, measured
from problem_fixtures import problem_spec

from klineage.backend.cuda.kernel import compile_native
from klineage.backend.runtime import BundleLoader


class FakeRuntime:
    calls = []
    active = 0
    peak = 0
    delay = 0

    def __init__(self, *, timeout):
        self.timeout = timeout

    def info(self, backend, **options):
        return backend.info()

    def evaluate(self, candidate, **options):
        type(self).active += 1
        type(self).peak = max(type(self).peak, type(self).active)
        try:
            time.sleep(self.delay)
            self.calls.append((candidate, self.timeout, options))
            logs = options["log_dir"]
            logs.mkdir(parents=True)
            (logs / "eval-0001-stderr.log").write_text("measured timing evidence\n")
            return measured(candidate)
        finally:
            type(self).active -= 1


class RecordingRuntime:
    calls = []

    def __init__(self, *, timeout):
        self.timeout = timeout

    def info(self, backend, **options):
        return backend.info()

    def evaluate(self, candidate, **options):
        self.calls.append((os.getpid(), candidate))
        options["log_dir"].mkdir(parents=True)
        (options["log_dir"] / "worker.log").write_text("test runtime evidence")
        return measured(candidate)


_CONFIG = """[solution]
name = "example"
definition = "example"
author = "klineage"
[build]
language = "{language}"
entry_point = "{entry}"
destination_passing_style = {dps}
"""


class BundleCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.loader = BundleLoader(self.root / "build", compiler=compile_native)
        self.counter = 0

    def kernel(self, sources, entry="binding.py::run", dps="false", language=None):
        self.counter += 1
        root = self.root / str(self.counter)
        language = language or ("python" if entry.endswith(".py::run") else "cuda")
        files = {
            "config.toml": _CONFIG.format(entry=entry, dps=dps, language=language),
            **{"solution/" + name: value for name, value in sources.items()},
        }
        for name, source in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        return make_kernel(
            name="example",
            source_files=files,
            problem=problem_spec("example", "cuda", "sm120"),
        )

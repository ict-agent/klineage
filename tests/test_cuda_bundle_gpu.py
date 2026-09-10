"""Opt-in CUDA execution checks: KLINEAGE_CUDA_TESTS=1."""

import os
import unittest
from dataclasses import replace

from problem_fixtures import io_problem
from test_cuda_bundle import BundleCase

from klineage.artifact.kernel import Kernel
from klineage.harness.artifacts import read_source_tree
from klineage.harness.codex_runner import SKILL_ROOT

RUN_CUDA = os.environ.get("KLINEAGE_CUDA_TESTS") == "1"
SIZE = 1027


@unittest.skipUnless(RUN_CUDA, "set KLINEAGE_CUDA_TESTS=1 on an available CUDA device")
class CudaBundleExecutionTests(BundleCase):
    def setUp(self):
        super().setUp()
        import torch

        from klineage.contract import ABIValue

        self.torch = torch
        value = ABIValue("x", "float32", (SIZE,), description="Contiguous CUDA tensor.")
        self.problem = io_problem(
            inputs=(value,), outputs=(replace(value, name="output"),)
        )

    def test_cuda_skill_example(self):
        # Compile the skill's shipped files, including its actual ABI checks.
        files = read_source_tree(SKILL_ROOT / "cuda/assets/add_one", "example")
        kernel = Kernel.from_sources(
            files, self.problem, build_root=self.root / "build"
        )
        self.check(kernel)

        # Call pybind11 directly to exercise the example's own ABI checks.
        run = self.loader.load(kernel)
        torch = self.torch
        x = torch.arange(SIZE, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        for inputs in (
            (x.cpu(), y),
            (x.half(), y),
            (x.reshape(1, -1), y),
            (x.repeat_interleave(2)[::2], y),
            (x, y[:-1]),
            (x, y.half()),
            (x, y.cpu()),
        ):
            with self.assertRaises((ValueError, TypeError)):
                run(*inputs)

        # Empty, single-element, and offset views remain valid.
        for size in (0, 1, SIZE):
            x = torch.arange(size + 1, device="cuda", dtype=torch.float32)[1:]
            y = torch.empty_like(x)
            run(x, y)
            torch.testing.assert_close(y, x + 1)

    def test_python_return(self):
        kernel = self.kernel(
            {"kernel.py": "def run(x): return x + 1"}, entry="kernel.py::run"
        )
        current = Kernel.from_sources(
            kernel.source_files, self.problem, build_root=self.root / "build"
        )
        self.check(current)

    def check(self, function):
        torch = self.torch
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x = torch.arange(SIZE, dtype=torch.float32, device="cuda")
            first = function(x)
            second = function(x + 7)
        stream.synchronize()
        torch.testing.assert_close(first, x + 1)
        torch.testing.assert_close(second, x + 8)

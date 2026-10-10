"""Opt-in CUDA execution checks: KLINEAGE_CUDA_TESTS=1."""

import os
import unittest

from kernel_fixtures import load_entry, make_kernel, write_kernel
from problem_fixtures import io_problem
from test_cuda_bundle import BundleCase

from klineage.artifact import source_files
from klineage.artifact.tree import read_source_tree
from klineage.backend import get_backend
from klineage.backend.cuda import CudaKernel

RUN_CUDA = os.environ.get("KLINEAGE_CUDA_TESTS") == "1"
SIZE = 1027


@unittest.skipUnless(RUN_CUDA, "set KLINEAGE_CUDA_TESTS=1 on an available CUDA device")
class CudaBundleExecutionTests(BundleCase):
    def setUp(self):
        super().setUp()
        import torch

        self.torch = torch
        value = {"dtype": "float32", "shape": [SIZE], "description": "Contiguous CUDA tensor."}
        self.problem = io_problem(
            inputs={"x": value}, outputs={"output": value}
        )

    def test_cuda_skill_example(self):
        # Compile the skill's shipped files, including its actual ABI checks.
        files = read_source_tree(get_backend("cuda").skill_root / "assets/add_one", "example")
        kernel = CudaKernel.load_kernel(write_kernel(make_kernel("add-one", self.problem, files), self.root / "kernel"))
        self.check(kernel)

        # Call pybind11 directly to exercise the example's own ABI checks.
        run = load_entry(self.loader, kernel)
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
        current = CudaKernel.load_kernel(
            write_kernel(
                make_kernel(
                    "python", self.problem, source_files(kernel.node("submission"))
                ),
                self.root / "kernel",
            )
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

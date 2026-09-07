"""Opt-in CUDA execution checks: KLINEAGE_CUDA_TESTS=1."""

import hashlib
import os
import unittest
from dataclasses import replace

from test_cuda_bundle import _BundleCase

_RUN_CUDA = os.environ.get("KLINEAGE_CUDA_TESTS") == "1"
_SIZE = 1027
_CUDA = '''#include <ATen/cuda/CUDAContext.h>
#include <stdexcept>
__global__ void add_one(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + 1;
}
void launch(const float* x, float* y, int n) {
  auto stream = at::cuda::getCurrentCUDAStream();
  add_one<<<(n + 255) / 256, 256, 0, stream>>>(x, y, n);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
'''


@unittest.skipUnless(_RUN_CUDA, "set KLINEAGE_CUDA_TESTS=1 on an available CUDA device")
class CudaBundleExecutionTests(_BundleCase):
    def setUp(self):
        super().setUp()
        import torch
        from klineage.contract import ABIValue, KernelABI
        from klineage.harness import _cuda_worker as worker
        from klineage.harness.timing import TimingPolicy

        self.torch = torch
        value = ABIValue("x", "float32", (_SIZE,), constraints={"stride": [1]})
        self.abi = KernelABI(inputs=(value,), outputs=(replace(value, name="output"),))
        self.worker = worker._RawLoader(worker._Config(
            self.root / "reference.py", self.root / "build", (), TimingPolicy(), 0,
        ))

    def test_direct_cuda_destination(self):
        source = _CUDA + '''#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
void run(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  launch(static_cast<const float*>(x.data_ptr()),
         static_cast<float*>(y.data_ptr()), x.size(0));
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
'''
        kernel = self.kernel({"kernel.cu": source}, entry="kernel.cu::run", dps="true")
        self._check(self.worker.load(replace(kernel, abi=self.abi)))

    def test_python_binding_return(self):
        namespace = "kl_test_" + hashlib.sha256(str(self.root).encode()).hexdigest()[:12]
        source = _CUDA + '''#include <ATen/ATen.h>
#include <torch/library.h>
at::Tensor run(at::Tensor x) {
  auto y = at::empty_like(x);
  launch(x.data_ptr<float>(), y.data_ptr<float>(), x.numel());
  return y;
}
TORCH_LIBRARY(NAMESPACE, m) { m.def("run(Tensor x) -> Tensor", &run); }
'''.replace("NAMESPACE", namespace)
        binding = f"import torch\n_native = torch.ops.{namespace}.run.default\ndef run(x): return _native(x)\n"
        kernel = self.kernel({"kernel.cu": source, "binding.py": binding}, language="cuda")
        self._check(self.worker.load(replace(kernel, abi=self.abi)))

    def _check(self, function):
        torch = self.torch
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x = torch.arange(_SIZE, dtype=torch.float32, device="cuda")
            first = function(x)
            second = function(x + 7)
        stream.synchronize()
        torch.testing.assert_close(first, x + 1)
        torch.testing.assert_close(second, x + 8)

        # Changed inputs expose kernels skipped by graph capture on the wrong stream.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = function(x)
        for value in (5, 8):
            x.fill_(value)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph.replay()
            stream.synchronize()
            torch.testing.assert_close(output, torch.full_like(x, value + 1))

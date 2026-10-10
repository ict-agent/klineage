#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {
constexpr int kThreads = 256;

__global__ void add_one(const float* x, float* y, int64_t n) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  y[i] = x[i] + 1.0f;
}

void check_tensor(const torch::Tensor& tensor) {
  TORCH_CHECK_VALUE(tensor.is_cuda(), "Expected CUDA tensor");
  TORCH_CHECK_TYPE(tensor.scalar_type() == torch::kFloat32, "Expected float32");
  TORCH_CHECK_VALUE(tensor.dim() == 1, "Expected rank 1");
  TORCH_CHECK_VALUE(tensor.stride(0) == 1, "Expected stride 1");
}

void kernel(const torch::Tensor& x, const torch::Tensor& y) {
  // Validate the contract before touching device memory.
  check_tensor(x);
  check_tensor(y);
  TORCH_CHECK_VALUE(x.size(0) == y.size(0), "Shape mismatch");
  TORCH_CHECK_VALUE(x.device() == y.device(), "Device mismatch");
  const int64_t n = x.size(0);
  if (n == 0) return;

  // data_ptr includes storage offset; preserve the caller's device and stream.
  c10::cuda::CUDAGuard guard(x.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  add_one<<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), n);
  const auto error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("kernel", &kernel);
}

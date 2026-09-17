// Host wrapper: validates the ordered ABI, sets cluster/shared attributes and
// launches the single-pass radix Top-K kernel on the caller's stream.
#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "topk.cuh"

namespace {

constexpr int kSharedCarveoutPercent = 25;

void check_input(const torch::Tensor& values) {
  TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
  TORCH_CHECK(values.scalar_type() == torch::kFloat32, "values must be float32");
  TORCH_CHECK(values.dim() == 2, "values must be rank 2");
  TORCH_CHECK(values.size(0) == 1 && values.size(1) == topk::kLength,
      "unsupported workload shape");
  TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(values.data_ptr<float>()) % 16 == 0,
      "values must be 16-byte aligned");
}

void check_output(const torch::Tensor& tensor, torch::ScalarType dtype) {
  TORCH_CHECK(tensor.is_cuda(), "output must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == dtype, "unexpected output dtype");
  TORCH_CHECK(tensor.sizes() == torch::IntArrayRef({1, topk::kSelected}),
      "unsupported output shape");
  TORCH_CHECK(tensor.is_contiguous(), "output must be contiguous");
}

void prepare() {
  static bool configured = false;
  if (configured) return;
  auto status = cudaFuncSetAttribute(topk::select,
      cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
  status = cudaFuncSetAttribute(topk::select,
      cudaFuncAttributePreferredSharedMemoryCarveout, kSharedCarveoutPercent);
  TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
  configured = true;
}

void kernel(const torch::Tensor& values, const torch::Tensor& top_values,
    const torch::Tensor& indices) {
  check_input(values);
  check_output(top_values, torch::kFloat32);
  check_output(indices, torch::kInt64);
  TORCH_CHECK(top_values.device() == values.device(), "device mismatch");
  TORCH_CHECK(indices.device() == values.device(), "device mismatch");

  c10::cuda::CUDAGuard guard(values.device());
  prepare();

  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim = {topk::kBlocks, 1, 1};

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(topk::kBlocks, 1, 1);
  config.blockDim = dim3(topk::kThreads, 1, 1);
  config.dynamicSmemBytes = 0;
  config.stream = c10::cuda::getCurrentCUDAStream(values.get_device()).stream();
  config.attrs = &attribute;
  config.numAttrs = 1;

  auto status = cudaLaunchKernelEx(&config, topk::select, values.data_ptr<float>(),
      top_values.data_ptr<float>(), indices.data_ptr<int64_t>());
  TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("kernel", &kernel);
}

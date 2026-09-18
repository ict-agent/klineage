// Host wrapper: validate the ordered ABI, then launch one 16-CTA cluster per row.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include "topk.cuh"

namespace {
constexpr int kClusterCarveoutPercent = 50;

void check_input(const torch::Tensor& tensor) {
  TORCH_CHECK_VALUE(tensor.is_cuda(), "values must be a CUDA tensor");
  TORCH_CHECK_TYPE(tensor.scalar_type() == torch::kFloat32, "values must be float32");
  TORCH_CHECK_VALUE(tensor.dim() == 2, "values must have shape [batch, sequence_length]");
  TORCH_CHECK_VALUE(tensor.stride(0) == tensor.size(1) && tensor.stride(1) == 1,
      "values must be contiguous");
}

void check_output(const torch::Tensor& tensor, const torch::Tensor& input,
    torch::ScalarType dtype, const char* name) {
  TORCH_CHECK_VALUE(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK_TYPE(tensor.scalar_type() == dtype, name, " has the wrong dtype");
  TORCH_CHECK_VALUE(tensor.dim() == 2 && tensor.size(0) == input.size(0)
      && tensor.size(1) == topk::kSelected, name, " must have shape [batch, 2048]");
  TORCH_CHECK_VALUE(tensor.stride(0) == tensor.size(1) && tensor.stride(1) == 1,
      name, " must be contiguous");
}

// Every CTA caches one chunk per rendezvous, so reject rows that need more.
void check_capacity(const torch::Tensor& input) {
  const int length = int(input.size(1));
  const uintptr_t row = reinterpret_cast<uintptr_t>(input.data_ptr<float>());
  const int head =
      int((topk::kAlignBytes - (row & (topk::kAlignBytes - 1))) & (topk::kAlignBytes - 1))
      / int(sizeof(float));
  const int chunks = (length - head) / topk::kChunkItems;
  const int per_cta = (chunks + topk::kBlocks - 1) / topk::kBlocks;
  TORCH_CHECK_VALUE(per_cta <= topk::kMaxChunks, "sequence length exceeds the chunk cache");
}

void set_attributes() {
  auto status = cudaFuncSetAttribute(topk::select,
      cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
  status = cudaFuncSetAttribute(topk::select, cudaFuncAttributeMaxDynamicSharedMemorySize,
      topk::kDynamicBytes);
  TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
  status = cudaFuncSetAttribute(topk::select,
      cudaFuncAttributePreferredSharedMemoryCarveout, kClusterCarveoutPercent);
  TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}

void launch(const torch::Tensor& input, torch::Tensor& top_values, torch::Tensor& indices) {
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim = {1, topk::kBlocks, 1};

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(unsigned(input.size(0)), topk::kBlocks, 1);
  config.blockDim = dim3(topk::kThreads, 1, 1);
  config.dynamicSmemBytes = topk::kDynamicBytes;
  config.stream = c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
  config.attrs = &attr;
  config.numAttrs = 1;

  const auto status = cudaLaunchKernelEx(&config, topk::select, input.data_ptr<float>(),
      top_values.data_ptr<float>(), indices.data_ptr<int64_t>(), int(input.size(1)));
  TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}

void kernel(const torch::Tensor& values, torch::Tensor& top_values, torch::Tensor& indices) {
  check_input(values);
  check_output(top_values, values, torch::kFloat32, "top_values");
  check_output(indices, values, torch::kInt64, "indices");
  check_capacity(values);
  TORCH_CHECK_VALUE(values.size(1) >= topk::kSelected, "sequence_length must be >= 2048");

  c10::cuda::CUDAGuard guard(values.device());
  set_attributes();
  launch(values, top_values, indices);
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("kernel", &kernel);
}

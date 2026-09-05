#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

#include <vector>

extern "C" cudaError_t klineage_launch(
    const void* const* inputs,
    void* const* outputs,
    cudaStream_t stream);

namespace {

void launch(
    const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& outputs) {
  TORCH_CHECK(
      !inputs.empty() || !outputs.empty(),
      "the raw CUDA ABI requires at least one tensor");

  const torch::Tensor& anchor = inputs.empty() ? outputs.front() : inputs.front();
  TORCH_CHECK(anchor.is_cuda(), "all ABI values must be CUDA tensors");
  c10::cuda::CUDAGuard device_guard(anchor.device());

  std::vector<const void*> input_pointers;
  input_pointers.reserve(inputs.size());
  for (const torch::Tensor& tensor : inputs) {
    TORCH_CHECK(tensor.is_cuda(), "all ABI inputs must be CUDA tensors");
    TORCH_CHECK(
        tensor.get_device() == anchor.get_device(),
        "all ABI values must be on the same CUDA device");
    input_pointers.push_back(tensor.data_ptr());
  }

  std::vector<void*> output_pointers;
  output_pointers.reserve(outputs.size());
  for (const torch::Tensor& tensor : outputs) {
    TORCH_CHECK(tensor.is_cuda(), "all ABI outputs must be CUDA tensors");
    TORCH_CHECK(
        tensor.get_device() == anchor.get_device(),
        "all ABI values must be on the same CUDA device");
    output_pointers.push_back(tensor.data_ptr());
  }

  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(anchor.get_device()).stream();
  cudaError_t status = klineage_launch(
      input_pointers.data(), output_pointers.data(), stream);
  TORCH_CHECK(
      status == cudaSuccess,
      "klineage_launch failed: ",
      cudaGetErrorString(status));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("launch", &launch, "Launch a raw KLineage CUDA kernel");
}

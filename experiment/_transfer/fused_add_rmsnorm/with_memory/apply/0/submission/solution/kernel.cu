// Fused residual add + RMSNorm, BF16 in/out with FP32 accumulation.
//
//   residual_out = (hidden + residual)                                   BF16
//   output       = (hidden + residual) * rsqrt(mean(square) + eps) * w   BF16
//
// One CTA owns one row: 896 threads * 8 columns == HIDDEN_SIZE. Every thread
// keeps its eight unrounded FP32 sums in registers, so the normalization pass
// neither re-reads the inputs nor needs a shared-memory row cache.

#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

// HIDDEN_SIZE is fixed by the problem definition; TOKENS is the grid dimension.
constexpr uint32_t kHidden = 7168;
constexpr uint32_t kVector = 8;  // BF16 lanes per 16-byte int4 transfer.
constexpr uint32_t kWarp = 32;
constexpr uint32_t kWarps = 28;
constexpr uint32_t kThreads = kWarps * kWarp;  // == kHidden / kVector.
constexpr uint32_t kAlignment = 16;            // int4 transfer alignment.
constexpr int kPdlEnabled = 1;
constexpr float kEpsilon = 1.0e-6f;

// Butterfly exchange of one FP32 register across a full warp.
__device__ __forceinline__ float shuffle_xor(float value, uint32_t mask) {
  float result;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
               : "=f"(result)
               : "f"(value), "r"(mask));
  return result;
}

__device__ __forceinline__ float rsqrt_approx(float value) {
  float result;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

__global__ void __launch_bounds__(kThreads) fused_norm(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    __nv_bfloat16* __restrict__ residual_out) {
  __shared__ float warp_totals[kWarps];

  const uint32_t lane = threadIdx.x;
  const uint32_t warp = threadIdx.y;
  const uint32_t column = (warp * kWarp + lane) * kVector;
  const size_t row_offset = size_t(blockIdx.x) * kHidden + column;

  // Wait for the upstream grid's memory before reading its results.
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;");
#endif

  // One int4 per tensor: eight adjacent BF16 values.
  const int4 input_pack = *reinterpret_cast<const int4*>(input + row_offset);
  const int4 residual_pack = *reinterpret_cast<const int4*>(residual + row_offset);
  const auto* input_values = reinterpret_cast<const __nv_bfloat16*>(&input_pack);
  const auto* residual_values = reinterpret_cast<const __nv_bfloat16*>(&residual_pack);

  // First pass: residual add, residual output, unrounded sum of squares.
  int4 residual_pack_out;
  auto* residual_out_values = reinterpret_cast<__nv_bfloat16*>(&residual_pack_out);
  float sums[kVector];
  float sum_squares = 0.0f;

#pragma unroll
  for (uint32_t j = 0; j < kVector; ++j) {
    const float value = __bfloat162float(input_values[j]) + __bfloat162float(residual_values[j]);
    sums[j] = value;
    sum_squares = fmaf(value, value, sum_squares);
    residual_out_values[j] = __float2bfloat16_rn(value);
  }
  *reinterpret_cast<int4*>(residual_out + row_offset) = residual_pack_out;

  // Two-level XOR reduction: local warps, then warp zero over the warp totals.
#pragma unroll
  for (uint32_t offset = kWarp / 2; offset > 0; offset /= 2) {
    sum_squares += shuffle_xor(sum_squares, offset);
  }
  if (lane == 0) warp_totals[warp] = sum_squares;
  __syncthreads();

  if (warp == 0) {
    sum_squares = lane < kWarps ? warp_totals[lane] : 0.0f;
#pragma unroll
    for (uint32_t offset = kWarp / 2; offset > 0; offset /= 2) {
      sum_squares += shuffle_xor(sum_squares, offset);
    }
    if (lane == 0) warp_totals[0] = sum_squares;
  }
  __syncthreads();

  const float inverse_rms = rsqrt_approx(warp_totals[0] / float(kHidden) + kEpsilon);
  const int4 weight_pack = *reinterpret_cast<const int4*>(weight + column);
  const auto* weight_values = reinterpret_cast<const __nv_bfloat16*>(&weight_pack);

  // Second pass: normalize the retained unrounded sums.
  int4 output_pack;
  auto* output_values = reinterpret_cast<__nv_bfloat16*>(&output_pack);

#pragma unroll
  for (uint32_t j = 0; j < kVector; ++j) {
    output_values[j] = __float2bfloat16_rn(sums[j] * inverse_rms * __bfloat162float(weight_values[j]));
  }
  *reinterpret_cast<int4*>(output + row_offset) = output_pack;

  // Allow the next grid in the stream to start while this one drains.
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

void check_tensor(const torch::Tensor& tensor) {
  TORCH_CHECK_VALUE(tensor.is_cuda(), "Expected CUDA tensor");
  TORCH_CHECK_TYPE(tensor.scalar_type() == torch::kBFloat16, "Expected bfloat16");
  TORCH_CHECK_VALUE(tensor.is_contiguous(), "Expected contiguous tensor");
  TORCH_CHECK_VALUE(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % kAlignment == 0,
                    "Expected 16-byte aligned tensor");
}

void kernel(const torch::Tensor& hidden_states, const torch::Tensor& residual,
            const torch::Tensor& weight, const torch::Tensor& output,
            const torch::Tensor& residual_out) {
  // Validate the contract before touching device memory.
  check_tensor(hidden_states);
  check_tensor(residual);
  check_tensor(weight);
  check_tensor(output);
  check_tensor(residual_out);
  TORCH_CHECK_VALUE(hidden_states.dim() == 2, "Expected rank 2 activations");
  TORCH_CHECK_VALUE(hidden_states.size(1) == kHidden, "Expected HIDDEN_SIZE columns");
  TORCH_CHECK_VALUE(residual.sizes() == hidden_states.sizes(), "Residual shape mismatch");
  TORCH_CHECK_VALUE(weight.dim() == 1 && weight.size(0) == kHidden, "Weight shape mismatch");
  TORCH_CHECK_VALUE(output.sizes() == hidden_states.sizes(), "Output shape mismatch");
  TORCH_CHECK_VALUE(residual_out.sizes() == hidden_states.sizes(), "Residual output shape mismatch");
  TORCH_CHECK_VALUE(hidden_states.device() == weight.device() && hidden_states.device() == output.device() &&
                        hidden_states.device() == residual.device() &&
                        hidden_states.device() == residual_out.device(),
                    "Device mismatch");

  const int64_t tokens = hidden_states.size(0);
  if (tokens == 0) return;

  // Preserve the caller's device and stream.
  c10::cuda::CUDAGuard guard(hidden_states.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(hidden_states.get_device()).stream();

  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = kPdlEnabled;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(uint32_t(tokens));
  config.blockDim = dim3(kWarp, kWarps);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  config.attrs = &attribute;
  config.numAttrs = 1;

  const auto* input_ptr = reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr());
  const auto* residual_ptr = reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr());
  const auto* weight_ptr = reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr());
  auto* output_ptr = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
  auto* residual_out_ptr = reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr());

  const cudaError_t status = cudaLaunchKernelEx(&config, fused_norm, input_ptr, residual_ptr,
                                                weight_ptr, output_ptr, residual_out_ptr);
  TORCH_CHECK(status == cudaSuccess, "Launch failed: ", cudaGetErrorString(status));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("kernel", &kernel);
}

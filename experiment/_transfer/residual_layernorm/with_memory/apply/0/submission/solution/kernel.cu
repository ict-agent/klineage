// Fused residual add + LayerNorm over the last (hidden) axis.
//
//   added  = fp16(x + residual)                    -> materialized output
//   output = fp16((added - mean) * rsqrt(var + eps) * gamma + beta)
//
// Statistics are taken on the *rounded* fp16 values, exactly like the
// reference. The op is bandwidth bound: six B*S*H tensors cross memory, so the
// kernel is built around one 16-byte access per thread per tensor and around
// keeping the row in registers between the two reduction phases.
//
// Worker mapping (hidden == kGroup), one warp owns one row:
//
//   warp w of CTA b -> row b*kWarps + w
//   round r, lane l -> columns [r*256 + l*8, +8)   (512 contiguous bytes/warp)
//
//   registers: x_pack/r_pack[4] --16B loads--> fp32 add --fp16 round--> values[32]
//                                                                       |
//                        warp butterfly sum --> mean --------------------+
//                        warp butterfly sq  --> var  --> rsqrt ---------+
//                                                                       v
//                        gamma/beta 16B loads --> fma --> output[32]
//
// The row never leaves registers, so each tensor is touched exactly once.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 128;              // four warps per CTA
constexpr int kWarps = kThreads / 32;
constexpr int kVec = 8;                    // fp16 lanes per 16-byte access
constexpr int kLanes = 32;
constexpr int kGroup = kThreads * kVec;    // columns covered by one sweep
constexpr int kRounds = kGroup / (kLanes * kVec);
constexpr int kMinBlocks = 9;              // 56 registers keeps two full waves
constexpr int kFallbackThreads = 256;
constexpr float kEpsilon = 1e-12f;
constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ float warp_sum(float value) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_xor_sync(kFullMask, value, offset);
  return value;
}

// CTA-wide sum: warp butterflies, one shared round, broadcast to every thread.
__device__ __forceinline__ float block_sum(float value, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_sum(value);
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < kWarps ? scratch[lane] : 0.0f;
    value = warp_sum(value);
    if (lane == 0) scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}

// Main path: hidden == kGroup, one warp per row, no CTA barrier.
__global__ __launch_bounds__(kThreads, kMinBlocks) void norm_warp_kernel(
    const __half* __restrict__ x, const __half* __restrict__ residual,
    const __half* __restrict__ gamma, const __half* __restrict__ beta,
    __half* __restrict__ output, __half* __restrict__ added, int hidden) {
  const int lane = threadIdx.x & 31;
  const int64_t base =
      (static_cast<int64_t>(blockIdx.x) * kWarps + (threadIdx.x >> 5)) * hidden;

  // Pass 1: residual add at the fp16 boundary, fp32 sum of the rounded values.
  // All four rounds are loaded up front so eight 16-byte requests are in flight.
  uint4 x_pack[kRounds];
  uint4 r_pack[kRounds];
  #pragma unroll
  for (int r = 0; r < kRounds; ++r) {
    const int64_t offset = base + r * (kLanes * kVec) + lane * kVec;
    x_pack[r] = *reinterpret_cast<const uint4*>(x + offset);
    r_pack[r] = *reinterpret_cast<const uint4*>(residual + offset);
  }

  float values[kRounds * kVec];
  float sum = 0.0f;
  #pragma unroll
  for (int r = 0; r < kRounds; ++r) {
    const __half* x_vals = reinterpret_cast<const __half*>(&x_pack[r]);
    const __half* r_vals = reinterpret_cast<const __half*>(&r_pack[r]);
    __half group[kVec];
    #pragma unroll
    for (int j = 0; j < kVec; ++j) {
      float* value = &values[r * kVec + j];
      *value = __half2float(__float2half_rn(__half2float(x_vals[j]) +
                                            __half2float(r_vals[j])));
      group[j] = __float2half_rn(*value);
      sum += *value;
    }
    *reinterpret_cast<uint4*>(added + base + r * (kLanes * kVec) + lane * kVec) =
        *reinterpret_cast<const uint4*>(group);
  }

  // Pass 2: mean, then the centered second moment of the rounded row.
  const float mean = warp_sum(sum) * (1.0f / hidden);
  float square_sum = 0.0f;
  #pragma unroll
  for (int j = 0; j < kRounds * kVec; ++j) {
    const float centered = values[j] - mean;
    square_sum = fmaf(centered, centered, square_sum);
  }
  const float inverse = rsqrtf(warp_sum(square_sum) * (1.0f / hidden) + kEpsilon);

  // Pass 3: affine normalization from the cached row.
  #pragma unroll
  for (int r = 0; r < kRounds; ++r) {
    const int column = r * (kLanes * kVec) + lane * kVec;
    const uint4 g_pack = *reinterpret_cast<const uint4*>(gamma + column);
    const uint4 b_pack = *reinterpret_cast<const uint4*>(beta + column);
    const __half* g_vals = reinterpret_cast<const __half*>(&g_pack);
    const __half* b_vals = reinterpret_cast<const __half*>(&b_pack);

    __half result[kVec];
    #pragma unroll
    for (int j = 0; j < kVec; ++j) {
      const float scaled = (values[r * kVec + j] - mean) * inverse;
      result[j] = __float2half_rn(
          fmaf(scaled, __half2float(g_vals[j]), __half2float(b_vals[j])));
    }
    *reinterpret_cast<uint4*>(output + base + column) =
        *reinterpret_cast<const uint4*>(result);
  }
}

// Generic path: any hidden width, one CTA per row, three streaming passes.
__global__ __launch_bounds__(kFallbackThreads) void norm_scalar_kernel(
    const __half* __restrict__ x, const __half* __restrict__ residual,
    const __half* __restrict__ gamma, const __half* __restrict__ beta,
    __half* __restrict__ output, __half* __restrict__ added, int64_t rows,
    int hidden) {
  __shared__ float scratch[kFallbackThreads / 32];

  const int64_t row = blockIdx.x;
  if (row >= rows) return;
  const int64_t base = row * hidden;

  float sum = 0.0f;
  for (int col = threadIdx.x; col < hidden; col += kFallbackThreads) {
    const float value = __half2float(__float2half_rn(
        __half2float(x[base + col]) + __half2float(residual[base + col])));
    added[base + col] = __float2half_rn(value);
    sum += value;
  }
  const float mean = block_sum(sum, scratch) * (1.0f / hidden);

  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < hidden; col += kFallbackThreads) {
    const float centered = __half2float(added[base + col]) - mean;
    square_sum = fmaf(centered, centered, square_sum);
  }
  const float inverse =
      rsqrtf(block_sum(square_sum, scratch) * (1.0f / hidden) + kEpsilon);

  for (int col = threadIdx.x; col < hidden; col += kFallbackThreads) {
    const float scaled = (__half2float(added[base + col]) - mean) * inverse;
    output[base + col] = __float2half_rn(
        fmaf(scaled, __half2float(gamma[col]), __half2float(beta[col])));
  }
}

void check_row_tensor(const torch::Tensor& tensor, int64_t rows, int hidden) {
  TORCH_CHECK(tensor.is_cuda(), "expected CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16, "expected float16");
  TORCH_CHECK(tensor.dim() == 3, "expected rank 3");
  TORCH_CHECK(tensor.is_contiguous(), "expected contiguous tensor");
  TORCH_CHECK(tensor.size(0) * tensor.size(1) == rows, "row count mismatch");
  TORCH_CHECK(tensor.size(2) == hidden, "hidden size mismatch");
}

void check_vector(const torch::Tensor& tensor, int64_t hidden) {
  TORCH_CHECK(tensor.is_cuda(), "expected CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16, "expected float16");
  TORCH_CHECK(tensor.dim() == 1, "expected rank 1");
  TORCH_CHECK(tensor.is_contiguous(), "expected contiguous tensor");
  TORCH_CHECK(tensor.size(0) == hidden, "hidden size mismatch");
}

void kernel(const torch::Tensor& x, const torch::Tensor& residual,
            const torch::Tensor& gamma, const torch::Tensor& beta,
            const torch::Tensor& output, const torch::Tensor& added) {
  const int64_t rows = x.size(0) * x.size(1);
  const int64_t hidden = x.size(2);
  check_row_tensor(x, rows, hidden);
  check_row_tensor(residual, rows, hidden);
  check_row_tensor(output, rows, hidden);
  check_row_tensor(added, rows, hidden);
  check_vector(gamma, hidden);
  check_vector(beta, hidden);
  TORCH_CHECK(x.device() == residual.device() && x.device() == output.device() &&
                  x.device() == added.device(),
              "device mismatch");
  if (rows == 0 || hidden == 0) return;

  c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(x.get_device()).stream();

  const __half* x_ptr = reinterpret_cast<const __half*>(x.data_ptr<at::Half>());
  const __half* residual_ptr =
      reinterpret_cast<const __half*>(residual.data_ptr<at::Half>());
  const __half* gamma_ptr =
      reinterpret_cast<const __half*>(gamma.data_ptr<at::Half>());
  const __half* beta_ptr = reinterpret_cast<const __half*>(beta.data_ptr<at::Half>());
  __half* output_ptr = reinterpret_cast<__half*>(output.data_ptr<at::Half>());
  __half* added_ptr = reinterpret_cast<__half*>(added.data_ptr<at::Half>());
  const int width = static_cast<int>(hidden);

  if (hidden == kGroup && rows % kWarps == 0) {
    norm_warp_kernel<<<static_cast<unsigned>(rows / kWarps), kThreads, 0, stream>>>(
        x_ptr, residual_ptr, gamma_ptr, beta_ptr, output_ptr, added_ptr, width);
  } else {
    norm_scalar_kernel<<<static_cast<unsigned>(rows), kFallbackThreads, 0, stream>>>(
        x_ptr, residual_ptr, gamma_ptr, beta_ptr, output_ptr, added_ptr, rows, width);
  }
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("kernel", &kernel);
}

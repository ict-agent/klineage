// Fused residual-add + LayerNorm for contiguous FP16 BSH tensors.
//
// One warp owns one row. A row of `hidden` FP16 columns is processed in steps
// of 256 columns (32 lanes x 8 columns); every lane moves 128-bit packets and
// keeps its 8 FP16 results in registers for the normalization pass.
//
//   columns ->  0                                                               1023
//   step 0      |lane0|8||lane1|8| ... |lane31|8|                (cols   0..255)
//   step 1      |lane0|8||lane1|8| ... |lane31|8|                (cols 256..511)
//   step 2      ...                                              (cols 512..767)
//   step 3      ...                                              (cols 768..1023)
//
// Row statistics (sum, sum of squares) are reduced with warp shuffles, so the
// kernel uses no shared memory and no CTA barrier. The FP16 residual rounding
// boundary is materialized exactly once, and the statistics read that rounded
// value, matching the reference semantics.

#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kLanes = 32;              // lanes per warp
constexpr int kVec = 8;                 // FP16 values per 128-bit transfer
constexpr int kChunk = kLanes * kVec;   // columns covered by one warp step
constexpr int kThreads = 256;           // threads per CTA
constexpr int kRowsPerCta = kThreads / kLanes;
constexpr int kMaxChunks = 4;           // column steps per row (hidden <= 1024)
constexpr float kEpsilon = 1e-12f;
constexpr unsigned kFullMask = 0xffffffffu;

// 128-bit packet of 8 FP16 values, addressed as four packed pairs.
struct __align__(16) Half8 {
  __half2 h[4];
};

__device__ __forceinline__ void accumulate(const __half2 value, float& sum, float& sumsq) {
  const float2 f = __half22float2(value);
  sum += f.x + f.y;
  sumsq = fmaf(f.y, f.y, fmaf(f.x, f.x, sumsq));
}

__device__ __forceinline__ __half2 normalize(const __half2 value, const float mean,
                                             const float rstd, const __half2 gamma,
                                             const __half2 beta) {
  const float2 f = __half22float2(value);
  const float2 g = __half22float2(gamma);
  const float2 b = __half22float2(beta);
  return __floats2half2_rn(fmaf((f.x - mean) * rstd, g.x, b.x),
                           fmaf((f.y - mean) * rstd, g.y, b.y));
}

__global__ void __launch_bounds__(kThreads)
residual_ln_kernel(const __half* __restrict__ x, const __half* __restrict__ residual,
                   const __half* __restrict__ gamma, const __half* __restrict__ beta,
                   __half* __restrict__ output, __half* __restrict__ added,
                   const int64_t rows, const int hidden) {
  const int lane = threadIdx.x & (kLanes - 1);
  const int row_in_cta = threadIdx.x >> 5;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kRowsPerCta + row_in_cta;
  if (row >= rows) return;

  const int64_t row_base = row * hidden;
  const int column0 = lane * kVec;
  const int chunks = hidden / kChunk;

  Half8 values[kMaxChunks];
  float sum = 0.0f;
  float sumsq = 0.0f;

  // Pass 1: FP16 residual add, materialize `added`, accumulate row statistics.
#pragma unroll
  for (int c = 0; c < kMaxChunks; ++c) {
    if (c >= chunks) break;
    const int64_t offset = row_base + c * kChunk + column0;
    const Half8 xv = *reinterpret_cast<const Half8*>(x + offset);
    Half8 av;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      av.h[i] = __hadd2(xv.h[i], reinterpret_cast<const Half8*>(residual + offset)->h[i]);
    *reinterpret_cast<Half8*>(added + offset) = av;
    values[c] = av;
#pragma unroll
    for (int i = 0; i < 4; ++i) accumulate(av.h[i], sum, sumsq);
  }

  // Butterfly reduction: every lane ends with the full row total.
#pragma unroll
  for (int delta = kLanes / 2; delta > 0; delta >>= 1) {
    sum += __shfl_xor_sync(kFullMask, sum, delta);
    sumsq += __shfl_xor_sync(kFullMask, sumsq, delta);
  }

  const float inv_hidden = 1.0f / static_cast<float>(hidden);
  const float mean = sum * inv_hidden;
  const float rstd = rsqrtf(fmaf(-mean, mean, sumsq * inv_hidden) + kEpsilon);

  // Pass 2: affine normalization of the retained values, FP16 rounded output.
#pragma unroll
  for (int c = 0; c < kMaxChunks; ++c) {
    if (c >= chunks) break;
    const int column = c * kChunk + column0;
    const Half8 gv = *reinterpret_cast<const Half8*>(gamma + column);
    const Half8 bv = *reinterpret_cast<const Half8*>(beta + column);
    Half8 out;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      out.h[i] = normalize(values[c].h[i], mean, rstd, gv.h[i], bv.h[i]);
    *reinterpret_cast<Half8*>(output + row_base + column) = out;
  }
}

void check_tensor(const torch::Tensor& tensor, const int64_t rows, const int64_t hidden,
                  const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kHalf, name, " must be FP16");
  TORCH_CHECK(tensor.dim() == 3, name, " must be rank 3");
  TORCH_CHECK(tensor.size(0) == rows && tensor.size(1) * tensor.size(0) / tensor.size(0) > 0,
              name, " batch mismatch");
  TORCH_CHECK(tensor.size(1) > 0 && tensor.size(2) == hidden, name, " shape mismatch");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void kernel(const torch::Tensor& x, const torch::Tensor& residual, const torch::Tensor& gamma,
            const torch::Tensor& beta, const torch::Tensor& output, const torch::Tensor& added) {
  const int64_t hidden = x.size(-1);
  const int64_t rows = x.size(0) * x.size(1);

  // Validate the contract before touching device memory.
  TORCH_CHECK(x.dim() == 3, "x must be rank 3");
  TORCH_CHECK(hidden % kVec == 0, "hidden must be a multiple of ", kVec);
  TORCH_CHECK(hidden <= kMaxChunks * kChunk, "hidden must not exceed ", kMaxChunks * kChunk);
  check_tensor(x, x.size(0), hidden, "x");
  check_tensor(residual, x.size(0), hidden, "residual");
  check_tensor(output, x.size(0), hidden, "output");
  check_tensor(added, x.size(0), hidden, "added");
  TORCH_CHECK(gamma.is_cuda() && gamma.scalar_type() == torch::kHalf && gamma.is_contiguous(),
              "gamma must be contiguous FP16");
  TORCH_CHECK(beta.is_cuda() && beta.scalar_type() == torch::kHalf && beta.is_contiguous(),
              "beta must be contiguous FP16");
  TORCH_CHECK(gamma.numel() == hidden && beta.numel() == hidden, "gamma/beta shape mismatch");
  TORCH_CHECK(x.device() == residual.device() && x.device() == output.device() &&
                  x.device() == added.device() && x.device() == gamma.device(),
              "device mismatch");
  if (rows == 0) return;

  c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const int64_t blocks = (rows + kRowsPerCta - 1) / kRowsPerCta;
  residual_ln_kernel<<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(residual.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(gamma.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(beta.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(added.data_ptr<at::Half>()), rows,
      static_cast<int>(hidden));

  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "residual_layernorm launch failed: ", cudaGetErrorString(error));
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("kernel", &kernel);
}

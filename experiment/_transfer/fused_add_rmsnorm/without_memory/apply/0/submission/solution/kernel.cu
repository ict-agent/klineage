// Fused residual add + RMSNorm (DeepSeek-V3.2), bf16 tensors, fp32 math.
//
//   sums    = fp32(hidden) + fp32(residual)          elementwise
//   res_out = bf16(sums)
//   inv_rms = rsqrt(mean(sums^2) + eps)              per token row
//   out     = bf16(sums * inv_rms * fp32(weight))
//
// One thread block per token row, each thread holding its slice of `sums` in
// registers, so the row is read once from DRAM. Both stores are issued after
// the reduction, which keeps the read and the write phases of a block apart:
//
//   pass 1 : load hidden, residual -> sums, square-sum
//   reduce : warp shuffle + shared -> row square-sum -> inv_rms
//   pass 2 : store res_out = bf16(sums); store out = bf16(sums * inv_rms * w)
//
// DRAM traffic is the minimum possible: read hidden + residual, write output +
// res_out (4 x TOKENS x HIDDEN x 2 bytes); weight is 14 KiB, read from L2.

#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr float kEps = 1e-6f;
constexpr int kVecLanes = 8;       // bf16 lanes per 16-byte vector
constexpr int kThreads = 256;      // threads per block on the vector path
constexpr int kVecsPerThread = 4;  // ceil(7168 / 8 / kThreads)
constexpr int kMaxVecs = kThreads * kVecsPerThread;

using bf16 = __nv_bfloat16;
using bf16x2 = __nv_bfloat162;

__device__ __forceinline__ float warp_sum(float value) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}

// sums = fp32(h) + fp32(r) for 8 lanes; folds their squares into `square_sum`.
__device__ __forceinline__ void add_vector(const uint4& h, const uint4& r,
                                           float (&sums)[kVecLanes],
                                           float& square_sum) {
  const bf16x2* h2 = reinterpret_cast<const bf16x2*>(&h);
  const bf16x2* r2 = reinterpret_cast<const bf16x2*>(&r);

  #pragma unroll
  for (int k = 0; k < kVecLanes / 2; ++k) {
    const float2 hf = __bfloat1622float2(h2[k]);
    const float2 rf = __bfloat1622float2(r2[k]);
    const float2 sf = make_float2(hf.x + rf.x, hf.y + rf.y);
    sums[2 * k] = sf.x;
    sums[2 * k + 1] = sf.y;
    square_sum = fmaf(sf.x, sf.x, square_sum);
    square_sum = fmaf(sf.y, sf.y, square_sum);
  }
}

__device__ __forceinline__ uint4 pack_bf16(const float (&sums)[kVecLanes]) {
  uint4 packed;
  bf16x2* pairs = reinterpret_cast<bf16x2*>(&packed);

  #pragma unroll
  for (int k = 0; k < kVecLanes / 2; ++k)
    pairs[k] = __floats2bfloat162_rn(sums[2 * k], sums[2 * k + 1]);
  return packed;
}

// Vector path: requires HIDDEN % 8 == 0 and HIDDEN / 8 <= kMaxVecs.
__global__ void __launch_bounds__(kThreads)
fused_add_rmsnorm_vec(const bf16* __restrict__ hidden,
                      const bf16* __restrict__ residual,
                      const bf16* __restrict__ weight,
                      bf16* __restrict__ out,
                      bf16* __restrict__ residual_out,
                      int vecs_per_row,
                      float inv_hidden) {
  constexpr int kWarps = kThreads / 32;

  const int lane = threadIdx.x;
  const int64_t row_base = int64_t(blockIdx.x) * vecs_per_row * kVecLanes;

  const uint4* hidden_row = reinterpret_cast<const uint4*>(hidden + row_base);
  const uint4* residual_row = reinterpret_cast<const uint4*>(residual + row_base);
  const uint4* weight_row = reinterpret_cast<const uint4*>(weight);
  uint4* out_row = reinterpret_cast<uint4*>(out + row_base);
  uint4* residual_row_out = reinterpret_cast<uint4*>(residual_out + row_base);

  float sums[kVecsPerThread][kVecLanes];
  float square_sum = 0.f;

  #pragma unroll
  for (int v = 0; v < kVecsPerThread; ++v) {
    const int index = lane + v * kThreads;

    #pragma unroll
    for (int k = 0; k < kVecLanes; ++k)
      sums[v][k] = 0.f;

    if (index >= vecs_per_row)
      continue;

    const uint4 hv = hidden_row[index];
    const uint4 rv = residual_row[index];
    add_vector(hv, rv, sums[v], square_sum);
  }

  // Row square-sum: intra-warp shuffle, then one broadcast read of warp sums.
  __shared__ float warp_sums[kWarps];
  square_sum = warp_sum(square_sum);

  if ((lane & 31) == 0)
    warp_sums[lane >> 5] = square_sum;
  __syncthreads();

  square_sum = 0.f;
  #pragma unroll
  for (int w = 0; w < kWarps; ++w)
    square_sum += warp_sums[w];

  const float inv_rms = rsqrtf(square_sum * inv_hidden + kEps);

  #pragma unroll
  for (int v = 0; v < kVecsPerThread; ++v) {
    const int index = lane + v * kThreads;

    if (index >= vecs_per_row)
      continue;

    residual_row_out[index] = pack_bf16(sums[v]);

    const uint4 wv = weight_row[index];
    const bf16x2* w2 = reinterpret_cast<const bf16x2*>(&wv);
    uint4 ov;
    bf16x2* o2 = reinterpret_cast<bf16x2*>(&ov);

    #pragma unroll
    for (int k = 0; k < kVecLanes / 2; ++k) {
      const float2 wf = __bfloat1622float2(w2[k]);
      o2[k] = __floats2bfloat162_rn(sums[v][2 * k] * inv_rms * wf.x,
                                    sums[v][2 * k + 1] * inv_rms * wf.y);
    }
    out_row[index] = ov;
  }
}

// Scalar fallback for shapes the vector path rejects; re-reads res_out instead
// of keeping the row in registers.
__global__ void __launch_bounds__(kThreads)
fused_add_rmsnorm_scalar(const bf16* __restrict__ hidden,
                         const bf16* __restrict__ residual,
                         const bf16* __restrict__ weight,
                         bf16* __restrict__ out,
                         bf16* __restrict__ residual_out,
                         int hidden_size,
                         float inv_hidden) {
  constexpr int kWarps = kThreads / 32;

  const int lane = threadIdx.x;
  const int64_t row_base = int64_t(blockIdx.x) * hidden_size;
  const bf16* hidden_row = hidden + row_base;
  const bf16* residual_row = residual + row_base;
  bf16* out_row = out + row_base;
  bf16* residual_row_out = residual_out + row_base;

  float square_sum = 0.f;
  for (int i = lane; i < hidden_size; i += kThreads) {
    const float sum = __bfloat162float(hidden_row[i]) + __bfloat162float(residual_row[i]);
    residual_row_out[i] = __float2bfloat16_rn(sum);
    square_sum = fmaf(sum, sum, square_sum);
  }

  __shared__ float warp_sums[kWarps];
  square_sum = warp_sum(square_sum);

  if ((lane & 31) == 0)
    warp_sums[lane >> 5] = square_sum;
  __syncthreads();

  square_sum = 0.f;
  #pragma unroll
  for (int w = 0; w < kWarps; ++w)
    square_sum += warp_sums[w];

  const float inv_rms = rsqrtf(square_sum * inv_hidden + kEps);

  for (int i = lane; i < hidden_size; i += kThreads) {
    const float sum = __bfloat162float(residual_row_out[i]);
    const float scale = inv_rms * __bfloat162float(weight[i]);
    out_row[i] = __float2bfloat16_rn(sum * scale);
  }
}

void check_tensor(const torch::Tensor& tensor, int64_t dims) {
  TORCH_CHECK_VALUE(tensor.is_cuda(), "expected a CUDA tensor");
  TORCH_CHECK_TYPE(tensor.scalar_type() == at::kBFloat16, "expected bfloat16");
  TORCH_CHECK_VALUE(tensor.dim() == dims, "unexpected tensor rank");
  TORCH_CHECK_VALUE(tensor.is_contiguous(), "expected contiguous tensor");
}

void kernel(const torch::Tensor& hidden_states,
            const torch::Tensor& residual,
            const torch::Tensor& weight,
            const torch::Tensor& output,
            const torch::Tensor& residual_out) {
  check_tensor(hidden_states, 2);
  check_tensor(residual, 2);
  check_tensor(weight, 1);
  check_tensor(output, 2);
  check_tensor(residual_out, 2);

  const int64_t tokens = hidden_states.size(0);
  const int64_t hidden_size = hidden_states.size(1);
  TORCH_CHECK_VALUE(hidden_size > 0, "empty hidden size");
  TORCH_CHECK_VALUE(hidden_size == weight.size(0), "weight shape mismatch");
  TORCH_CHECK_VALUE(residual.size(0) == tokens && residual.size(1) == hidden_size,
                    "residual shape mismatch");
  TORCH_CHECK_VALUE(output.sizes() == hidden_states.sizes(), "output shape mismatch");
  TORCH_CHECK_VALUE(residual_out.sizes() == hidden_states.sizes(), "residual_out shape mismatch");
  TORCH_CHECK_VALUE(output.device() == hidden_states.device(), "output device mismatch");
  TORCH_CHECK_VALUE(residual_out.device() == hidden_states.device(), "residual_out device mismatch");
  TORCH_CHECK_VALUE(residual.device() == hidden_states.device(), "residual device mismatch");
  TORCH_CHECK_VALUE(weight.device() == hidden_states.device(), "weight device mismatch");

  if (tokens == 0)
    return;

  c10::cuda::CUDAGuard device_guard(hidden_states.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(hidden_states.get_device()).stream();

  const bf16* hidden_ptr = reinterpret_cast<const bf16*>(hidden_states.data_ptr());
  const bf16* residual_ptr = reinterpret_cast<const bf16*>(residual.data_ptr());
  const bf16* weight_ptr = reinterpret_cast<const bf16*>(weight.data_ptr());
  bf16* output_ptr = reinterpret_cast<bf16*>(output.data_ptr());
  bf16* residual_out_ptr = reinterpret_cast<bf16*>(residual_out.data_ptr());
  const float inv_hidden = 1.0f / static_cast<float>(hidden_size);

  const int vecs_per_row = static_cast<int>(hidden_size / kVecLanes);
  const bool vectorizable = (hidden_size % kVecLanes == 0) && (vecs_per_row <= kMaxVecs);

  if (vectorizable) {
    fused_add_rmsnorm_vec<<<static_cast<unsigned>(tokens), kThreads, 0, stream>>>(
        hidden_ptr, residual_ptr, weight_ptr, output_ptr, residual_out_ptr,
        vecs_per_row, inv_hidden);
  } else {
    fused_add_rmsnorm_scalar<<<static_cast<unsigned>(tokens), kThreads, 0, stream>>>(
        hidden_ptr, residual_ptr, weight_ptr, output_ptr, residual_out_ptr,
        static_cast<int>(hidden_size), inv_hidden);
  }

  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "kernel launch failed: ", cudaGetErrorString(error));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("kernel", &kernel);
}

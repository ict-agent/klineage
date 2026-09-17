// Fused residual-add + LayerNorm for fp16 activations, one row per warp.
//
// Reference semantics (row of H hidden elements):
//   added  = fp16(x + residual)                  // fp32 add, fp16 round-trip
//   mean   = mean(fp32(added))
//   var    = mean((fp32(added) - mean)^2)
//   output = fp16((added - mean) * rsqrt(var + eps) * gamma + beta)
//
//   x ─┐
//      ├─ add ─ round ─┬─ store added ──────────────────► global
//   r ─┘               └─ mean/var (registers) ─ norm ─ affine ─► global
//
// One warp owns a row: 32 threads x 4 vectors of 8 halves = 1024 elements.
// Every thread issues eight independent 16-byte loads before the first
// reduction, so the memory pipeline stays filled; the statistics need only
// warp shuffles, so no barrier ever stalls the block.

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

constexpr int kHidden = 1024;  // H, fixed by the definition
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 2;
constexpr int kThreads = kWarpSize * kWarpsPerBlock;
constexpr int kHalvesPerVec = 8;  // 16 B per access
constexpr int kHalvesPerWarp = kWarpSize * kHalvesPerVec;      // one round
constexpr int kVectorsPerThread = kHidden / kHalvesPerWarp;   // rounds per row
constexpr int kPairsPerVec = kHalvesPerVec / 2;
constexpr float kEps = 1e-12f;
constexpr unsigned kFullMask = 0xffffffffu;

// 8 halves moved by a single 16-byte transaction.
struct alignas(16) Vec8 {
  unsigned int word[kPairsPerVec];
};

__device__ __forceinline__ Vec8 load_vec8(const half* __restrict__ src) {
  const uint4 raw = *reinterpret_cast<const uint4*>(src);
  Vec8 value;
  value.word[0] = raw.x;
  value.word[1] = raw.y;
  value.word[2] = raw.z;
  value.word[3] = raw.w;
  return value;
}

__device__ __forceinline__ void store_vec8(half* __restrict__ dst, Vec8 value) {
  *reinterpret_cast<uint4*>(dst) = make_uint4(value.word[0], value.word[1],
                                              value.word[2], value.word[3]);
}

__device__ __forceinline__ half2 pair_at(const Vec8& value, int index) {
  half2 pair;
  memcpy(&pair, &value.word[index], sizeof(pair));
  return pair;
}

__device__ __forceinline__ void set_pair(Vec8& value, int index, half2 pair) {
  memcpy(&value.word[index], &pair, sizeof(pair));
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(kFullMask, value, offset);
  }
  return value;
}

__global__ __launch_bounds__(kThreads) void residual_layernorm_warp(
    const half* __restrict__ x, const half* __restrict__ residual,
    const half* __restrict__ gamma, const half* __restrict__ beta,
    half* __restrict__ output, half* __restrict__ added, int rows) {
  const int lane = threadIdx.x % kWarpSize;
  const int row = blockIdx.x * kWarpsPerBlock + threadIdx.x / kWarpSize;
  if (row >= rows) return;

  const int col = lane * kHalvesPerVec;
  const size_t rowoffset = size_t(row) * kHidden + col;
  const half* xrow = x + rowoffset;
  const half* rrow = residual + rowoffset;

  // Pass 1: residual add with the fp16 boundary materialized, plus row sum.
  Vec8 av[kVectorsPerThread];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kVectorsPerThread; ++i) {
    const Vec8 xv = load_vec8(xrow + i * kHalvesPerWarp);
    const Vec8 rv = load_vec8(rrow + i * kHalvesPerWarp);
#pragma unroll
    for (int j = 0; j < kPairsPerVec; ++j) {
      const float2 xf = __half22float2(pair_at(xv, j));
      const float2 rf = __half22float2(pair_at(rv, j));
      const half2 pair = __floats2half2_rn(xf.x + rf.x, xf.y + rf.y);
      set_pair(av[i], j, pair);
      const float2 af = __half22float2(pair);
      sum += af.x + af.y;
    }
  }
#pragma unroll
  for (int i = 0; i < kVectorsPerThread; ++i) {
    store_vec8(added + rowoffset + i * kHalvesPerWarp, av[i]);
  }

  const float mean = warp_sum(sum) / float(kHidden);

  // Pass 2: centered variance from the register-resident row.
  float sqsum = 0.0f;
#pragma unroll
  for (int i = 0; i < kVectorsPerThread; ++i) {
#pragma unroll
    for (int j = 0; j < kPairsPerVec; ++j) {
      const float2 af = __half22float2(pair_at(av[i], j));
      const float dx = af.x - mean;
      const float dy = af.y - mean;
      sqsum += dx * dx + dy * dy;
    }
  }
  const float rstd = rsqrtf(warp_sum(sqsum) / float(kHidden) + kEps);

  // Pass 3: affine transform with per-hidden gamma/beta.
#pragma unroll
  for (int i = 0; i < kVectorsPerThread; ++i) {
    const Vec8 gv = load_vec8(gamma + col + i * kHalvesPerWarp);
    const Vec8 bv = load_vec8(beta + col + i * kHalvesPerWarp);
    Vec8 ov;
#pragma unroll
    for (int j = 0; j < kPairsPerVec; ++j) {
      const float2 af = __half22float2(pair_at(av[i], j));
      const float2 gf = __half22float2(pair_at(gv, j));
      const float2 bf = __half22float2(pair_at(bv, j));
      const float nx = (af.x - mean) * rstd;
      const float ny = (af.y - mean) * rstd;
      set_pair(ov, j, __floats2half2_rn(fmaf(nx, gf.x, bf.x), fmaf(ny, gf.y, bf.y)));
    }
    store_vec8(output + rowoffset + i * kHalvesPerWarp, ov);
  }
}

void check_activation(const torch::Tensor& tensor, const char* name,
                      int64_t rows) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16, name, " must be float16");
  TORCH_CHECK(tensor.dim() == 3, name, " must be rank 3");
  TORCH_CHECK(tensor.size(0) * tensor.size(1) == rows, name, " row count mismatch");
  TORCH_CHECK(tensor.size(2) == kHidden, name, " hidden size must be ", kHidden);
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_affine(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16, name, " must be float16");
  TORCH_CHECK(tensor.numel() == kHidden, name, " must hold ", kHidden, " elements");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void kernel(const torch::Tensor& x, const torch::Tensor& residual,
            const torch::Tensor& gamma, const torch::Tensor& beta,
            const torch::Tensor& output, const torch::Tensor& added) {
  const int64_t rows = x.size(0) * x.size(1);
  check_activation(x, "x", rows);
  check_activation(residual, "residual", rows);
  check_affine(gamma, "gamma");
  check_affine(beta, "beta");
  check_activation(output, "output", rows);
  check_activation(added, "added", rows);
  TORCH_CHECK(x.device() == residual.device() && x.device() == output.device() &&
                  x.device() == added.device(),
              "tensors must share a device");
  if (rows == 0) return;

  c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device());
  const int blocks = (int(rows) + kWarpsPerBlock - 1) / kWarpsPerBlock;
  residual_layernorm_warp<<<blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(gamma.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(beta.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      reinterpret_cast<half*>(added.data_ptr<at::Half>()), int(rows));
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("kernel", &kernel);
}

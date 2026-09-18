// Residual + LayerNorm over the last axis (H = 1024), fp16 in/out.
//
//   added  = fp16(x + residual)                      (materialized boundary)
//   output = (added - mean(added)) * rsqrt(var + eps) * gamma + beta
//
// One warp owns one row: 32 lanes x 4 16-byte vectors cover H exactly, the
// values stay in registers, and the statistics need no shared memory or
// block barrier -- only warp shuffles.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kHidden = 1024;                 // problem constant axis H
constexpr int kVecHalves = 8;                 // halves per 16-byte access
constexpr int kVecCount = kHidden / kVecHalves;
constexpr int kWarpSize = 32;
constexpr int kVpt = kVecCount / kWarpSize;   // vectors per lane = 4
constexpr int kWarpsPerBlock = 8;
constexpr int kBlockThreads = kWarpsPerBlock * kWarpSize;
constexpr int kLanes = kVecHalves / 2;        // half2 pairs inside one vector
constexpr int kRowLoop = 0;            // rows handled per warp in total
constexpr float kEpsilon = 1e-12f;
constexpr float kInvHidden = 1.0f / kHidden;

constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ float2 warpSum(float2 v) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v.x += __shfl_down_sync(kFullMask, v.x, offset);
    v.y += __shfl_down_sync(kFullMask, v.y, offset);
  }
  return v;
}

// Block-wide sum of {sum, sum of squares}; valid in warp 0 only.
__device__ __forceinline__ float2 blockSumGeneric(float2 v, float2* scratch) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  const int warps = blockDim.x / kWarpSize;
  v = warpSum(v);
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  if (warp != 0) return make_float2(0.f, 0.f);
  v = (lane < warps) ? scratch[lane] : make_float2(0.f, 0.f);
  return warpSum(v);
}

__device__ __forceinline__ uint4 loadVec(const uint4* p) {
  return *p;
}
__device__ __forceinline__ void storeVec(uint4* p, uint4 v) {
  *p = v;
}

// One row per warp: 32 lanes x 4 vectors = 1024 halves.
__device__ __forceinline__ void lnRow(const half* __restrict__ x,
                                      const half* __restrict__ residual,
                                      const half* __restrict__ gamma,
                                      const half* __restrict__ beta,
                                      half* __restrict__ out,
                                      half* __restrict__ added, int lane,
                                      int64_t base) {
  uint4 av[kVpt];
  half2 ah[kVpt * kLanes];
  float2 acc = make_float2(0.f, 0.f);
#pragma unroll
  for (int k = 0; k < kVpt; ++k) {
    const int64_t off = base + (lane + k * kWarpSize) * kVecHalves;
    const uint4 xv = loadVec(reinterpret_cast<const uint4*>(x + off));
    const uint4 rv = loadVec(reinterpret_cast<const uint4*>(residual + off));
    const half2* xh = reinterpret_cast<const half2*>(&xv);
    const half2* rh = reinterpret_cast<const half2*>(&rv);
    half2* avh = reinterpret_cast<half2*>(&av[k]);
#pragma unroll
    for (int j = 0; j < kLanes; ++j) {
      ah[k * kLanes + j] = __hadd2(xh[j], rh[j]);
      avh[j] = ah[k * kLanes + j];
      const float2 f = __half22float2(avh[j]);
      acc.x += f.x + f.y;
      acc.y = fmaf(f.x, f.x, acc.y);
      acc.y = fmaf(f.y, f.y, acc.y);
    }
  }
#pragma unroll
  for (int k = 0; k < kVpt; ++k) {
    storeVec(reinterpret_cast<uint4*>(added + base) + lane + k * kWarpSize,
             av[k]);
  }

  const float2 total = warpSum(acc);
  const float mean = __shfl_sync(kFullMask, total.x, 0) * kInvHidden;
  const float sq = __shfl_sync(kFullMask, total.y, 0) * kInvHidden;
  const float rstd = rsqrtf(fmaf(-mean, mean, sq) + kEpsilon);

#pragma unroll
  for (int k = 0; k < kVpt; ++k) {
    const int vec = lane + k * kWarpSize;
    const uint4 gv = loadVec(reinterpret_cast<const uint4*>(gamma) + vec);
    const uint4 bv = loadVec(reinterpret_cast<const uint4*>(beta) + vec);
    const half2* gh = reinterpret_cast<const half2*>(&gv);
    const half2* bh = reinterpret_cast<const half2*>(&bv);
    uint4 ov;
    half2* oh = reinterpret_cast<half2*>(&ov);
#pragma unroll
    for (int j = 0; j < kLanes; ++j) {
      const float2 f = __half22float2(ah[k * kLanes + j]);
      const float2 g = __half22float2(gh[j]);
      const float2 b = __half22float2(bh[j]);
      oh[j] = __floats2half2_rn(fmaf((f.x - mean) * rstd, g.x, b.x),
                                fmaf((f.y - mean) * rstd, g.y, b.y));
    }
    storeVec(reinterpret_cast<uint4*>(out + base) + vec, ov);
  }
}

__global__ void __launch_bounds__(kBlockThreads) ln_1024_kernel(
    const half* __restrict__ x, const half* __restrict__ residual,
    const half* __restrict__ gamma, const half* __restrict__ beta,
    half* __restrict__ out, half* __restrict__ added, int64_t rows) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  const int64_t step = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;
#if defined(ROW_LOOP)
  // Persistent warps: each warp walks a strided set of rows so that the loaded
  // vectors of the next row overlap the reduction of the current one.
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
       row < rows; row += step) {
    lnRow(x, residual, gamma, beta, out, added, lane, row * kHidden);
  }
#else
  (void)rows;
  (void)step;
  lnRow(x, residual, gamma, beta, out, added, lane,
        (static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp) * kHidden);
#endif
}

// Fallback for rows that cannot use 16-byte vectors.
__global__ void ln_generic_kernel(const half* __restrict__ x,
                                  const half* __restrict__ residual,
                                  const half* __restrict__ gamma,
                                  const half* __restrict__ beta,
                                  half* __restrict__ out,
                                  half* __restrict__ added, int64_t hidden) {
  __shared__ float2 scratch[32];
  __shared__ float2 stats;

  const int64_t base = static_cast<int64_t>(blockIdx.x) * hidden;
  float2 acc = make_float2(0.f, 0.f);
  for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    const half a = __hadd(x[base + i], residual[base + i]);
    added[base + i] = a;
    const float f = __half2float(a);
    acc.x += f;
    acc.y = fmaf(f, f, acc.y);
  }
  const float2 total = blockSumGeneric(acc, scratch);
  if (threadIdx.x == 0) {
    const float mean = total.x / static_cast<float>(hidden);
    const float var = fmaf(-mean, mean, total.y / static_cast<float>(hidden));
    stats = make_float2(mean, rsqrtf(var + kEpsilon));
  }
  __syncthreads();
  for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = __half2float(added[base + i]);
    const float g = __half2float(gamma[i]);
    const float b = __half2float(beta[i]);
    out[base + i] = __float2half(fmaf((v - stats.x) * stats.y, g, b));
  }
}

void check_input(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK_VALUE(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK_TYPE(tensor.scalar_type() == torch::kFloat16, name, " must be fp16");
  TORCH_CHECK_VALUE(tensor.is_contiguous(), name, " must be contiguous");
}

void check_row(const torch::Tensor& tensor, const char* name, const torch::Tensor& ref) {
  TORCH_CHECK_VALUE(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK_TYPE(tensor.scalar_type() == torch::kFloat16, name, " must be fp16");
  TORCH_CHECK_VALUE(tensor.sizes() == ref.sizes(), name, " shape mismatch");
  TORCH_CHECK_VALUE(tensor.is_contiguous(), name, " must be contiguous");
}

bool is_vec_ready(const torch::Tensor& tensor) {
  const auto ptr = reinterpret_cast<uintptr_t>(tensor.data_ptr<at::Half>());
  return ptr % 16 == 0;
}

bool is_launchable(const torch::Tensor& x, const torch::Tensor& residual,
                   const torch::Tensor& gamma, const torch::Tensor& beta,
                   const torch::Tensor& output, const torch::Tensor& added,
                   int64_t hidden, int64_t rows) {
  if (hidden != kHidden) return false;
  if (rows % kWarpsPerBlock != 0) return false;
  return is_vec_ready(x) && is_vec_ready(residual) && is_vec_ready(output) &&
         is_vec_ready(added) && is_vec_ready(gamma) && is_vec_ready(beta);
}

void kernel(const torch::Tensor& x, const torch::Tensor& residual,
            const torch::Tensor& gamma, const torch::Tensor& beta,
            const torch::Tensor& output, const torch::Tensor& added) {
  check_input(x, "x");
  check_input(residual, "residual");
  check_input(gamma, "gamma");
  check_input(beta, "beta");
  check_row(output, "output", x);
  check_row(added, "added", x);
  TORCH_CHECK_VALUE(x.dim() == 3, "x must be rank 3");
  TORCH_CHECK_VALUE(gamma.dim() == 1 && gamma.size(0) == x.size(2), "gamma mismatch");
  TORCH_CHECK_VALUE(beta.dim() == 1 && beta.size(0) == x.size(2), "beta mismatch");

  const int64_t rows = x.size(0) * x.size(1);
  const int64_t hidden = x.size(2);
  if (rows == 0) return;

  c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const half* xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  const half* rp = reinterpret_cast<const half*>(residual.data_ptr<at::Half>());
  const half* gp = reinterpret_cast<const half*>(gamma.data_ptr<at::Half>());
  const half* bp = reinterpret_cast<const half*>(beta.data_ptr<at::Half>());
  half* op = reinterpret_cast<half*>(output.data_ptr<at::Half>());
  half* ap = reinterpret_cast<half*>(added.data_ptr<at::Half>());

  if (is_launchable(x, residual, gamma, beta, output, added, hidden, rows)) {
    const unsigned grid = static_cast<unsigned>(rows / kWarpsPerBlock / (kRowLoop ? kRowLoop : 1));
    ln_1024_kernel<<<grid, kBlockThreads, 0, stream>>>(xp, rp, gp, bp, op, ap, rows);
  } else {
    ln_generic_kernel<<<static_cast<unsigned>(rows), 256, 0, stream>>>(
        xp, rp, gp, bp, op, ap, hidden);
  }
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("kernel", &kernel);
}

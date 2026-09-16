// KDA prefill: bounded-gate Kimi-K3 delta-rule recurrence, FP32 state.
//
// One thread block per (batch, head); the 128x128 state lives in registers as
// 4-row x 16-column tiles, so the per-token recurrence never touches memory
// for the state:
//
//   state = state * decay        (per key column)
//   pred  = state . k            (row-wise dot over keys)
//   delta = gain * (v - pred)
//   state += delta (x) k         (rank-1 update)
//   out   = state . q            (row-wise dot over keys)
//
// Row groups of 8 lanes share a thread tile, so partial sums stay inside the
// warp (three shfl rounds). Token tiles are staged in shared memory through a
// four-deep circular pipeline filled with cp.async: the copies for token t+3
// are in flight while token t is computed, which keeps DRAM latency off the
// critical path. One warp copies one row; the same warp derives the row's
// shared quantities (norms, gate decay) for the following token.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace {

constexpr int kHeadDim = 128;      // fixed by the problem definition
constexpr int kThreads = 256;
constexpr int kRowsPerThread = 4;  // value-dim rows per thread
constexpr int kColsPerThread = 16; // key-dim columns per thread
constexpr int kDepth = 4;          // pipeline stages
constexpr int kRowK = 0;           // warp roles
constexpr int kRowQ = 1;
constexpr int kRowV = 2;
constexpr int kRowGLo = 3;
constexpr int kRowGHi = 6;
constexpr int kRowBeta = 4;
constexpr float kNormEps = 1e-6f;

struct alignas(16) Stage {
  __nv_bfloat16 k[kHeadDim];
  __nv_bfloat16 q[kHeadDim];
  __nv_bfloat16 v[kHeadDim];
  __nv_bfloat16 g[kHeadDim];
  float dec[kHeadDim];
  float inv_nk;
  float inv_nq;
  float gain;
};

__device__ __forceinline__ void cp_async8(void* dst, const void* src) {
  const unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(smem), "l"(src));
}

__device__ __forceinline__ void cp_async4(void* dst, const void* src) {
  const unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem), "l"(src));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ float warpsum(float v) {
#pragma unroll
  for (int off = 16; off; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// Sigmoid of the beta logits / bounded gate transform of the reference operator.
__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float4 as_float4(const uint2 raw) {
  const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
  const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
  return make_float4(__low2float(lo), __high2float(lo), __low2float(hi), __high2float(hi));
}

__global__ void __launch_bounds__(kThreads, 1) kda_kernel(
    const __nv_bfloat16* __restrict__ qg, const __nv_bfloat16* __restrict__ kg,
    const __nv_bfloat16* __restrict__ vg, const __nv_bfloat16* __restrict__ gg,
    const __nv_bfloat16* __restrict__ betag, const float* __restrict__ scale,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias,
    const float* __restrict__ lower_bound, const float* __restrict__ init_state,
    __nv_bfloat16* __restrict__ outg, float* __restrict__ final_state, int tokens, int heads) {
  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int rgrp = tid >> 3;        // rows 4*rgrp .. 4*rgrp+3
  const int cgrp = tid & 7;         // keys 16*cgrp .. 16*cgrp+15
  const int crow = rgrp * kRowsPerThread;
  const int ccol = cgrp * kColsPerThread;

  __shared__ Stage s[kDepth];
  __shared__ float s_dtb[kHeadDim];
  // Scalars come straight from global memory: shared copies would need a
  // barrier before first use, and unbarriered reads race with the writers.
  const float lower_bound_v = *lower_bound;
  const float alog_exp = expf(a_log[head]);
  const float out_scale = *scale;

  // Recurrent state tile: rows crow..crow+3, columns ccol..ccol+15.
  float st[kRowsPerThread][kColsPerThread];
  {
    const float* const src = init_state + static_cast<size_t>(head) * kHeadDim * kHeadDim;
#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
      const float* const row = src + static_cast<size_t>(crow + r) * kHeadDim + ccol;
#pragma unroll
      for (int j = 0; j < kColsPerThread; j += 4) {
        const float4 v4 = *reinterpret_cast<const float4*>(row + j);
        st[r][j] = v4.x;
        st[r][j + 1] = v4.y;
        st[r][j + 2] = v4.z;
        st[r][j + 3] = v4.w;
      }
    }
  }
  for (int i = tid; i < kHeadDim; i += kThreads) s_dtb[i] = dt_bias[head * kHeadDim + i];

  const __nv_bfloat16* const k_base = kg + static_cast<size_t>(head) * kHeadDim;
  const __nv_bfloat16* const q_base = qg + static_cast<size_t>(head) * kHeadDim;
  const __nv_bfloat16* const v_base = vg + static_cast<size_t>(head) * kHeadDim;
  const __nv_bfloat16* const g_base = gg + static_cast<size_t>(head) * kHeadDim;
  const int row_stride = heads * kHeadDim;

  // Copy the role's slice of one token row into the shared stage.
  auto issue_copy = [&](int token) {
    Stage* const stg = &s[token & (kDepth - 1)];
    const int off = token * row_stride;
    if (warp == kRowK) {
      cp_async8(&stg->k[lane * 4], k_base + off + lane * 4);
    } else if (warp == kRowQ) {
      cp_async8(&stg->q[lane * 4], q_base + off + lane * 4);
    } else if (warp == kRowV) {
      cp_async8(&stg->v[lane * 4], v_base + off + lane * 4);
    } else if (warp == kRowGLo) {
      cp_async4(&stg->g[lane * 2], g_base + off + lane * 2);
    } else if (warp == kRowGHi) {
      cp_async4(&stg->g[64 + lane * 2], g_base + off + 64 + lane * 2);
    }
  };

  // Derive the shared quantities of one staged token row.
  auto derive = [&](int token) {
    Stage* const stg = &s[token & (kDepth - 1)];
    if (warp == kRowK) {
      const float4 a = as_float4(*reinterpret_cast<const uint2*>(&stg->k[lane * 4]));
      const float ss = warpsum(fmaf(a.x, a.x, fmaf(a.y, a.y, fmaf(a.z, a.z, a.w * a.w))));
      if (lane == 0) stg->inv_nk = rsqrtf(ss + kNormEps);
    } else if (warp == kRowQ) {
      const float4 a = as_float4(*reinterpret_cast<const uint2*>(&stg->q[lane * 4]));
      const float ss = warpsum(fmaf(a.x, a.x, fmaf(a.y, a.y, fmaf(a.z, a.z, a.w * a.w))));
      if (lane == 0) stg->inv_nq = rsqrtf(ss + kNormEps);
    } else if (warp == kRowGLo || warp == kRowGHi) {
      // decay = exp(lower_bound * sigmoid(exp(a_log) * (g + dt_bias)))
      const int base = (warp == kRowGLo) ? lane * 2 : 64 + lane * 2;
      const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&stg->g[base]);
      stg->dec[base] =
          expf(lower_bound_v * sigmoidf(alog_exp * (__low2float(a) + s_dtb[base])));
      stg->dec[base + 1] =
          expf(lower_bound_v * sigmoidf(alog_exp * (__high2float(a) + s_dtb[base + 1])));
    }
  };

  // Pipeline prologue: copies for tokens 0..2, derived quantities for token 0.
  if (tokens > 2) {
    issue_copy(0);
    issue_copy(1);
    issue_copy(2);
  } else {
    for (int t = 0; t < tokens; ++t) issue_copy(t);
  }
  cp_commit();
  cp_wait<0>();
  __syncthreads();
  if (warp == kRowBeta && lane == 0) s[0].gain = sigmoidf(__bfloat162float(betag[head]));
  derive(0);

  for (int token = 0; token < tokens; ++token) {
    const int buf = token & (kDepth - 1);
    __syncthreads();
    float beta_next = 0.0f;
    if (warp == kRowBeta && lane == 0 && token + 1 < tokens) {
      beta_next = __bfloat162float(betag[static_cast<size_t>(token + 1) * heads + head]);
    }
    if (token + 3 < tokens) {
      issue_copy(token + 3);
      cp_commit();
      cp_wait<2>();
    }
    if (token + 1 < tokens) {
      derive(token + 1);
      if (warp == kRowBeta && lane == 0) s[(token + 1) & (kDepth - 1)].gain = sigmoidf(beta_next);
    }
    const Stage* const sp = &s[buf];

    // This thread's key slice of the staged token.
    float key[kColsPerThread];
    float qry[kColsPerThread];
    float dec[kColsPerThread];
#pragma unroll
    for (int j = 0; j < kColsPerThread; j += 8) {
      const uint4 kr = *reinterpret_cast<const uint4*>(sp->k + ccol + j);
      const uint4 qr = *reinterpret_cast<const uint4*>(sp->q + ccol + j);
      const float4 k0 = as_float4(make_uint2(kr.x, kr.y));
      const float4 k1 = as_float4(make_uint2(kr.z, kr.w));
      const float4 q0 = as_float4(make_uint2(qr.x, qr.y));
      const float4 q1 = as_float4(make_uint2(qr.z, qr.w));
      const float4 d0 = *reinterpret_cast<const float4*>(sp->dec + ccol + j);
      const float4 d1 = *reinterpret_cast<const float4*>(sp->dec + ccol + j + 4);
      key[j] = k0.x;
      key[j + 1] = k0.y;
      key[j + 2] = k0.z;
      key[j + 3] = k0.w;
      key[j + 4] = k1.x;
      key[j + 5] = k1.y;
      key[j + 6] = k1.z;
      key[j + 7] = k1.w;
      qry[j] = q0.x;
      qry[j + 1] = q0.y;
      qry[j + 2] = q0.z;
      qry[j + 3] = q0.w;
      qry[j + 4] = q1.x;
      qry[j + 5] = q1.y;
      qry[j + 6] = q1.z;
      qry[j + 7] = q1.w;
      dec[j] = d0.x;
      dec[j + 1] = d0.y;
      dec[j + 2] = d0.z;
      dec[j + 3] = d0.w;
      dec[j + 4] = d1.x;
      dec[j + 5] = d1.y;
      dec[j + 6] = d1.z;
      dec[j + 7] = d1.w;
    }
    const float4 vf = as_float4(*reinterpret_cast<const uint2*>(sp->v + crow));
    const float val[kRowsPerThread] = {vf.x, vf.y, vf.z, vf.w};
    const float gain = sp->gain;
    const float inv_nk = sp->inv_nk;
    const float inv_nq = sp->inv_nq;

    // Decay, then predict: pred = sum_k (state * decay) * key.
    float pred[kRowsPerThread];
#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
      float acc = 0.0f;
#pragma unroll
      for (int j = 0; j < kColsPerThread; ++j) {
        const float sv = st[r][j] * dec[j];
        st[r][j] = sv;
        acc = fmaf(sv, key[j], acc);
      }
      pred[r] = acc;
    }
#pragma unroll
    for (int off = 1; off < kColsPerThread / 2; off <<= 1) {
#pragma unroll
      for (int r = 0; r < kRowsPerThread; ++r) {
        pred[r] += __shfl_xor_sync(0xffffffffu, pred[r], off);
      }
    }

    // Delta correction; the normalised key scale is folded in (key = k * inv_nk).
    float delta[kRowsPerThread];
#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
      delta[r] = gain * (val[r] - pred[r] * inv_nk) * inv_nk;
    }

    // Rank-1 update of the decayed state.
#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
#pragma unroll
      for (int j = 0; j < kColsPerThread; ++j) st[r][j] = fmaf(delta[r], key[j], st[r][j]);
    }

    // Readout of the post-update state.
    float out[kRowsPerThread];
#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
      float acc = 0.0f;
#pragma unroll
      for (int j = 0; j < kColsPerThread; ++j) acc = fmaf(st[r][j], qry[j], acc);
      out[r] = acc;
    }
#pragma unroll
    for (int off = 1; off < kColsPerThread / 2; off <<= 1) {
#pragma unroll
      for (int r = 0; r < kRowsPerThread; ++r) {
        out[r] += __shfl_xor_sync(0xffffffffu, out[r], off);
      }
    }
    if (cgrp == 0) {
      const float factor = inv_nq * out_scale;
      __nv_bfloat16* const dst =
          outg + static_cast<size_t>(token) * row_stride + static_cast<size_t>(head) * kHeadDim;
#pragma unroll
      for (int r = 0; r < kRowsPerThread; ++r) {
        dst[crow + r] = __float2bfloat16(out[r] * factor);
      }
    }

  }

  {
    float* const dst = final_state + static_cast<size_t>(head) * kHeadDim * kHeadDim;
#pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
      float* const row = dst + static_cast<size_t>(crow + r) * kHeadDim + ccol;
#pragma unroll
      for (int j = 0; j < kColsPerThread; j += 4) {
        *reinterpret_cast<float4*>(row + j) = make_float4(st[r][j], st[r][j + 1], st[r][j + 2],
                                                          st[r][j + 3]);
      }
    }
  }
}

void check(const torch::Tensor& t, at::ScalarType dtype, int dim) {
  TORCH_CHECK_VALUE(t.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK_TYPE(t.scalar_type() == dtype, "unexpected dtype");
  TORCH_CHECK_VALUE(t.dim() == dim, "unexpected rank");
  TORCH_CHECK_VALUE(t.is_contiguous(), "input must be contiguous");
}

void kernel(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
            const torch::Tensor& g, const torch::Tensor& beta, const torch::Tensor& scale,
            const torch::Tensor& a_log, const torch::Tensor& dt_bias,
            const torch::Tensor& lower_bound, const torch::Tensor& initial_state,
            const torch::Tensor& output, const torch::Tensor& final_state) {
  check(q, at::kBFloat16, 4);
  check(k, at::kBFloat16, 4);
  check(v, at::kBFloat16, 4);
  check(g, at::kBFloat16, 4);
  check(beta, at::kBFloat16, 3);
  check(scale, at::kFloat, 0);
  check(a_log, at::kFloat, 1);
  check(dt_bias, at::kFloat, 2);
  check(lower_bound, at::kFloat, 0);
  check(initial_state, at::kFloat, 4);
  check(output, at::kBFloat16, 4);
  check(final_state, at::kFloat, 4);
  const int batch = static_cast<int>(q.size(0));
  const int tokens = static_cast<int>(q.size(1));
  const int heads = static_cast<int>(q.size(2));
  const int dim = static_cast<int>(q.size(3));
  TORCH_CHECK_VALUE(dim == kHeadDim, "kernel is specialised for head_dim=128");
  TORCH_CHECK_VALUE(batch == 1, "only batch=1 is supported");

  const c10::cuda::CUDAGuard guard(q.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
  kda_kernel<<<batch * heads, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(g.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr<at::BFloat16>()),
      scale.data_ptr<float>(), a_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
      lower_bound.data_ptr<float>(), initial_state.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      final_state.data_ptr<float>(), tokens, heads);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "kernel launch failed: ", cudaGetErrorString(err));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("kernel", &kernel); }

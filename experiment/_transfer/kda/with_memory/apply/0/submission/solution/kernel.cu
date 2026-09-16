// Kimi-K3 KDA prefill: chunked delta rule on tensor cores, warp specialized.
//
// The reference walks one token at a time over a [V=128, K=128] FP32 state.
// This kernel processes 16-token chunks. Because the gate only rescales the key
// axis, the chunk-internal recurrence collapses into a small triangular system
// and all dense work becomes m16n8k16 BF16 MMA with FP32 accumulators:
//
//   pred^T = S @ Kt^T                     (state read, both readouts share it)
//   rhs^T  = gain*(v^T - pred^T)
//   delta^T = rhs^T @ TT                  (TT = M^-T, M = I + diag(gain)*A)
//   o^T    = S @ Qt^T + delta^T @ SAT
//   S      = dlast*S + delta^T @ Kpp
//
// Chunk-local notation (t, j in [0,16), d in [0,128), j < t):
//   gc[t][d]  cumulative log decay inside the chunk
//   Kt[t][d]  = kn*exp(gc)             ktT stored d-major (ldmatrix.trans)
//   KpT[d][t]= kn*exp(-gc)             d-major, A operand for A/S_A
//   Kpp[t][d] = kn*exp(gc_last - gc)   token major, update B operand
//   Qt[t][d]  = qn*exp(gc)             qtT stored d-major
//   A[t][j]   = <Kt_t, Kp_j>   (strictly lower part feeds M)
//   SAT[j][t] = S_A[t][j] = <Qt_t, Kp_j> masked to j <= t
//   dlast[d]  = exp(gc[15][d])
//
// 16 warps: warps 0-7 own the recurrent state and run the sequential MMA chain,
// warps 8-15 prepare the next chunk's operands one buffer ahead, so the gate
// SFU work and the triangular inverse never sit on the critical path.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

using bf16 = __nv_bfloat16;

constexpr int kDim = 128;
constexpr int kChunk = 16;
constexpr int kHeads = 96;
constexpr int kTokens = 4096;
constexpr int kNChunk = kTokens / kChunk;
constexpr int kHalf = kChunk / 2;
constexpr int kWarps = 16;
constexpr int kThreads = kWarps * 32;
constexpr int kDenseThreads = 256;
constexpr int kPrepThreads = 256;
constexpr float kNormEps = 1e-6f;
constexpr float kLog2e = 1.4426950408889634f;
constexpr int kOStageStride = kDim + 8;
constexpr int kX = kOStageStride;

// Named barrier ids: 1-3 prep internal, 4-5 ready (parity), 6-7 free (parity),
// 8 dense internal.
constexpr int kBarPrep = 1;
constexpr int kBarReady0 = 4;
constexpr int kBarFree0 = 6;
constexpr int kBarDense = 8;

__device__ __forceinline__ float ex2a(float x) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ float rcpa(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldm_x4t(uint32_t* r, const void* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(a));
}

__device__ __forceinline__ void bar_sync(int id, int count) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count));
}

__device__ __forceinline__ void bar_arrive(int id, int count) {
  asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(count));
}

// Two ldmatrix.x4 address patterns shared by both warp groups.
// B operand of the d-major matrices (Kt, Qt, Kp): k = d, n = t.
__device__ __forceinline__ const bf16* b_addr_dmaj(const bf16 (*m)[kChunk], int d0,
                                                   int lane) {
  const int i = lane & 7;
  const int sel = lane >> 3;  // 0..3
  return &m[d0 + ((sel & 1) ? 8 : 0) + i][(sel >> 1) ? 8 : 0];
}

// B operand of the token-major matrices (Kpp, TT, SAT): k = j/t, n = d/j.
__device__ __forceinline__ const bf16* b_addr_tmaj(const bf16 (*m)[kDim], int n0,
                                                   int lane) {
  const int i = lane & 7;
  const int sel = lane >> 3;
  return &m[((sel & 1) ? 8 : 0) + i][n0 + ((sel >> 1) ? 8 : 0)];
}

__device__ __forceinline__ const bf16* b_addr_tmaj16(const bf16 (*m)[kChunk], int n0,
                                                     int lane) {
  const int i = lane & 7;
  const int sel = lane >> 3;
  return &m[((sel & 1) ? 8 : 0) + i][n0 + ((sel >> 1) ? 8 : 0)];
}

struct Operand {
  bf16 ktT[kDim][kChunk];   // exp(gc) scaled keys, d-major
  bf16 qtT[kDim][kChunk];   // exp(gc) scaled queries, d-major
  bf16 kpT[kDim][kChunk];   // exp(-gc) scaled keys, d-major (A operand of A/SAT)
  bf16 kpp[kChunk][kDim];   // exp(gc_last - gc) scaled keys, token major
  bf16 vt[kDim][kChunk];    // gain*v, value major
  float dlast[kDim];
  float gain[kChunk];
  bf16 sat[kChunk][kChunk];
  bf16 tt[kChunk][kChunk];
};

struct Shared {
  bf16 q_raw[2][kChunk][kDim];
  bf16 k_raw[2][kChunk][kDim];
  bf16 v_raw[2][kChunk][kDim];
  bf16 g_raw[2][kChunk][kDim];
  float beta_raw[2][kChunk];
  Operand op[2];
  bf16 kn[2][kChunk][kDim];
  bf16 qn[2][kChunk][kDim];
  float mt[kChunk][kChunk];
  float inv1[kHalf][kHalf];
  float inv2[kHalf][kHalf];
  float prod[kHalf][kHalf];
  bf16 o_stage[1][kChunk][kOStageStride];
  float dtb[kDim];
  float aexp;
};

// ---------------------------------------------------------------------------
// Producer group: normalize, gate, and materialize the next chunk's operands.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void prefetch_raw(const bf16* __restrict__ q,
                                             const bf16* __restrict__ k,
                                             const bf16* __restrict__ v,
                                             const bf16* __restrict__ g,
                                             const bf16* __restrict__ beta, Shared& s,
                                             int head, int chunk, int buf, int ptid) {
  if (ptid < kChunk * 16) {
    const int t = ptid >> 4;
    const int seg = ptid & 15;
    const long long off = (static_cast<long long>(chunk) * kChunk + t) * (kHeads * kDim) +
                          head * kDim + seg * 8;
    const uint32_t sq = static_cast<uint32_t>(__cvta_generic_to_shared(&s.q_raw[buf][t][seg * 8]));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sq), "l"(q + off));
    const uint32_t sk = static_cast<uint32_t>(__cvta_generic_to_shared(&s.k_raw[buf][t][seg * 8]));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sk), "l"(k + off));
    const uint32_t sv = static_cast<uint32_t>(__cvta_generic_to_shared(&s.v_raw[buf][t][seg * 8]));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sv), "l"(v + off));
    const uint32_t sg = static_cast<uint32_t>(__cvta_generic_to_shared(&s.g_raw[buf][t][seg * 8]));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sg), "l"(g + off));
  }
  if (ptid < kChunk) {
    const int bi = (static_cast<long long>(chunk) * kChunk + ptid) * kHeads + head;
    s.beta_raw[buf][ptid] = __bfloat162float(beta[bi]);
  }
  asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ float2 unpack2(uint32_t u) {
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u));
}

// Normalize q (threads 128-255) and k (threads 0-127) for one chunk, 8 lanes per
// row with vectorized 32-byte accesses on both sides.
__device__ __forceinline__ void prep_norm(Shared& s, Operand& ob, int cur, int buf,
                                          int ptid, float scale) {
  const bool is_q = ptid >= 128;
  const int p = is_q ? ptid - 128 : ptid;
  const int row = p >> 3;
  const int seg = p & 7;
  const bf16* src = (is_q ? &s.q_raw[cur][row][0] : &s.k_raw[cur][row][0]) + seg * 16;
  const uint4 pa = *reinterpret_cast<const uint4*>(src);
  const uint4 pb = *reinterpret_cast<const uint4*>(src + 8);
  const float2 fa[4] = {unpack2(pa.x), unpack2(pa.y), unpack2(pa.z), unpack2(pa.w)};
  const float2 fb[4] = {unpack2(pb.x), unpack2(pb.y), unpack2(pb.z), unpack2(pb.w)};
  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    acc = fmaf(fa[i].x, fa[i].x, fmaf(fa[i].y, fa[i].y, acc));
    acc = fmaf(fb[i].x, fb[i].x, fmaf(fb[i].y, fb[i].y, acc));
  }
  acc += __shfl_xor_sync(0xffffffffu, acc, 1);
  acc += __shfl_xor_sync(0xffffffffu, acc, 2);
  acc += __shfl_xor_sync(0xffffffffu, acc, 4);
  const float sc = rsqrtf(acc + kNormEps) * (is_q ? scale : 1.0f);
  // Packed in memory order: fa covers elements 0-7, fb covers 8-15.
  uint32_t pk[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    pk[i] = pack2(fa[i].x * sc, fa[i].y * sc);
    pk[4 + i] = pack2(fb[i].x * sc, fb[i].y * sc);
  }
  const uint4 p0 = make_uint4(pk[0], pk[1], pk[2], pk[3]);
  const uint4 p1 = make_uint4(pk[4], pk[5], pk[6], pk[7]);
  bf16* dst = (is_q ? &s.qn[buf][row][0] : &s.kn[buf][row][0]) + seg * 16;
  *reinterpret_cast<uint4*>(dst) = p0;
  *reinterpret_cast<uint4*>(dst + 8) = p1;
  if (ptid < kChunk) {
    ob.gain[ptid] = rcpa(1.0f + ex2a(-s.beta_raw[cur][ptid] * kLog2e));
  }
}

// Gate + intra-chunk scan + operand materialization in one pass. Each thread
// owns channel d and half of the token axis; the two halves combine through a
// shuffle, so no cumulative-sum round trip through shared memory is needed.
__device__ __forceinline__ void prep_operands(Shared& s, Operand& ob, int cur, int kb,
                                              int ptid, float lb) {
  const int d = ptid >> 1;
  const int th = ptid & 1;
  float ld[8];
  float sum = 0.0f;
  const float dtb = s.dtb[d];
  const float aexp = s.aexp;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int t = th * 8 + i;
    const float x = aexp * (__bfloat162float(s.g_raw[cur][t][d]) + dtb);
    ld[i] = lb * rcpa(1.0f + ex2a(-x * kLog2e));
    sum += ld[i];
  }
  const float other = __shfl_xor_sync(0xffffffffu, sum, 1);
  const float total = sum + other;
  const float base = th ? other : 0.0f;
  const float dlast = ex2a(total * kLog2e);
  float gc = base;
  float e[8];
  float ei[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    gc += ld[i];
    e[i] = ex2a(gc * kLog2e);
    ei[i] = rcpa(e[i]);
  }
  if (th) ob.dlast[d] = dlast;
  uint32_t ktp[4];
  uint32_t qtp[4];
  uint32_t kpp_pack[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float ke[2];
    float qe[2];
    float pe[2];
#pragma unroll
    for (int e2 = 0; e2 < 2; ++e2) {
      const int i = 2 * j + e2;
      const int t = th * 8 + i;
      const float knv = __bfloat162float(s.kn[kb][t][d]);
      const float qnv = __bfloat162float(s.qn[kb][t][d]);
      ke[e2] = knv * e[i];
      qe[e2] = qnv * e[i];
      pe[e2] = knv * ei[i];
      ob.kpp[t][d] = __float2bfloat16(pe[e2] * dlast);
      ob.vt[d][t] = __float2bfloat16(ob.gain[t] * __bfloat162float(s.v_raw[cur][t][d]));
    }
    ktp[j] = pack2(ke[0], ke[1]);
    qtp[j] = pack2(qe[0], qe[1]);
    kpp_pack[j] = pack2(pe[0], pe[1]);
  }
  *reinterpret_cast<uint4*>(&ob.ktT[d][th * 8]) = *reinterpret_cast<uint4*>(ktp);
  *reinterpret_cast<uint4*>(&ob.qtT[d][th * 8]) = *reinterpret_cast<uint4*>(qtp);
  *reinterpret_cast<uint4*>(&ob.kpT[d][th * 8]) = *reinterpret_cast<uint4*>(kpp_pack);
}

// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kThreads, 1) kda_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ k,
    const bf16* __restrict__ v, const bf16* __restrict__ g,
    const bf16* __restrict__ beta, const float* __restrict__ a_log,
    const float* __restrict__ dt_bias, const float* __restrict__ init_state,
    bf16* __restrict__ out, float* __restrict__ final_state, float lb,
    float scale) {
  extern __shared__ char smem_raw[];
  Shared& s = *reinterpret_cast<Shared*>(smem_raw);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int head = blockIdx.x;
  const int gq = lane >> 2;
  const int gc4 = lane & 3;

  if (tid == 0) s.aexp = __expf(a_log[head]);
  if (tid < kDim) s.dtb[tid] = dt_bias[head * kDim + tid];
  __syncthreads();

  const long long row_stride = static_cast<long long>(kHeads) * kDim;

  if (warp >= 8) {
    // ---------------- producer ----------------
    const int ptid = tid - kDenseThreads;
    const int pwarp = warp - 8;
    prefetch_raw(q, k, v, g, beta, s, head, 0, 0, ptid);
    for (int c = 0; c < kNChunk; ++c) {
      const int p = c & 1;
      if (c >= 2) bar_sync(kBarFree0 + p, kThreads);
      asm volatile("cp.async.wait_group 0;\n");
      bar_sync(kBarPrep, kPrepThreads);
      if (c + 1 < kNChunk) prefetch_raw(q, k, v, g, beta, s, head, c + 1, p ^ 1, ptid);

      Operand& ob = s.op[p];
      prep_norm(s, ob, p, p, ptid, scale);
      bar_sync(kBarPrep, kPrepThreads);
      prep_operands(s, ob, p, p, ptid, lb);
      bar_sync(kBarPrep, kPrepThreads);

      // A^T = Kp @ Kt^T and S_A^T = Kp @ Qt^T, one warp each.
      if (pwarp < 2) {
        uint32_t afr[8][4];
#pragma unroll
        for (int m = 0; m < 8; ++m) {
          const int i = lane & 7;
          const int sel = lane >> 3;
          ldm_x4t(afr[m],
                  &ob.kpT[m * 16 + ((sel >> 1) ? 8 : 0) + i][(sel & 1) ? 8 : 0]);
        }
        float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        const bf16(*bmat)[kChunk] = (pwarp == 0) ? ob.ktT : ob.qtT;
#pragma unroll
        for (int m = 0; m < 8; ++m) {
          uint32_t bf[4];
          ldm_x4t(bf, b_addr_dmaj(bmat, m * 16, lane));
          mma_bf16(acc[0], afr[m], bf[0], bf[1]);
          mma_bf16(acc[1], afr[m], bf[2], bf[3]);
        }
        const int jr = gq;
        const int tc = gc4 * 2;
        if (pwarp == 0) {
          auto mtv = [&](int j, int t, float a) {
            return (t > j) ? ob.gain[t] * a : ((t == j) ? 1.0f : 0.0f);
          };
          *reinterpret_cast<float2*>(&s.mt[jr][tc]) =
              make_float2(mtv(jr, tc, acc[0][0]), mtv(jr, tc + 1, acc[0][1]));
          *reinterpret_cast<float2*>(&s.mt[jr][8 + tc]) =
              make_float2(mtv(jr, 8 + tc, acc[1][0]), mtv(jr, 9 + tc, acc[1][1]));
          *reinterpret_cast<float2*>(&s.mt[jr + 8][tc]) =
              make_float2(mtv(jr + 8, tc, acc[0][2]), mtv(jr + 8, tc + 1, acc[0][3]));
          *reinterpret_cast<float2*>(&s.mt[jr + 8][8 + tc]) =
              make_float2(mtv(jr + 8, 8 + tc, acc[1][2]), mtv(jr + 8, 9 + tc, acc[1][3]));
        } else {
          auto satv = [&](int j, int t, float a) { return (j <= t) ? a : 0.0f; };
          uint32_t pk[4];
          pk[0] = pack2(satv(jr, tc, acc[0][0]), satv(jr, tc + 1, acc[0][1]));
          pk[1] = pack2(satv(jr, 8 + tc, acc[1][0]), satv(jr, 9 + tc, acc[1][1]));
          pk[2] = pack2(satv(jr + 8, tc, acc[0][2]), satv(jr + 8, tc + 1, acc[0][3]));
          pk[3] = pack2(satv(jr + 8, 8 + tc, acc[1][2]), satv(jr + 8, 9 + tc, acc[1][3]));
          *reinterpret_cast<uint32_t*>(&ob.sat[jr][tc]) = pk[0];
          *reinterpret_cast<uint32_t*>(&ob.sat[jr][8 + tc]) = pk[1];
          *reinterpret_cast<uint32_t*>(&ob.sat[jr + 8][tc]) = pk[2];
          *reinterpret_cast<uint32_t*>(&ob.sat[jr + 8][8 + tc]) = pk[3];
        }
      }
      // Zero the lower-left block of T^T (never written by the inverse steps).
      if (ptid < 64) {
        ob.tt[kHalf + (ptid >> 3)][ptid & 7] = __float2bfloat16(0.0f);
      }
      bar_sync(kBarPrep, kPrepThreads);
      // Invert the two 8x8 diagonal blocks of M^T (upper unit triangular).
      if (pwarp == 0 && lane < kChunk) {
        const int blk = lane >> 3;
        const int j = lane & 7;
        const int r0 = blk * kHalf;
        float x[8];
#pragma unroll
        for (int i = 7; i >= 0; --i) {
          float acc = (i == j) ? 1.0f : 0.0f;
#pragma unroll
          for (int kk = i + 1; kk < 8; ++kk) acc -= s.mt[r0 + i][r0 + kk] * x[kk];
          x[i] = acc;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          (blk == 0 ? s.inv1 : s.inv2)[i][j] = x[i];
          ob.tt[r0 + i][r0 + j] = __float2bfloat16(x[i]);
        }
      }
      bar_sync(kBarPrep, kPrepThreads);
      // Merge the off-diagonal block: tt01 = -inv1 @ M12 @ inv2.
      if (ptid < 64) {
        const int i = ptid >> 3;
        const int j = ptid & 7;
        float acc = 0.0f;
#pragma unroll
        for (int kk = 0; kk < kHalf; ++kk) acc += s.mt[i][kHalf + kk] * s.inv2[kk][j];
        s.prod[i][j] = acc;
      }
      bar_sync(kBarPrep, kPrepThreads);
      if (ptid < 64) {
        const int i = ptid >> 3;
        const int j = ptid & 7;
        float acc = 0.0f;
#pragma unroll
        for (int kk = 0; kk < kHalf; ++kk) acc -= s.inv1[i][kk] * s.prod[kk][j];
        ob.tt[i][kHalf + j] = __float2bfloat16(acc);
      }
      bar_sync(kBarPrep, kPrepThreads);
      __threadfence_block();
      bar_arrive(kBarReady0 + p, kThreads);
    }
    return;
  }

  // ---------------- consumer ----------------
  // Recurrent state as the accumulator of the update MMA: warp w owns value
  // rows [16w, 16w+16); each lane holds 16 column tiles of 8.
  float st[16][4];
  {
    const int vrow = warp * 16 + gq;
    const float* src = init_state + head * kDim * kDim;
#pragma unroll
    for (int n = 0; n < 16; ++n) {
      const int d = n * 8 + gc4 * 2;
      st[n][0] = src[vrow * kDim + d];
      st[n][1] = src[vrow * kDim + d + 1];
      st[n][2] = src[(vrow + 8) * kDim + d];
      st[n][3] = src[(vrow + 8) * kDim + d + 1];
    }
  }

  for (int c = 0; c < kNChunk; ++c) {
    const int p = c & 1;
    bar_sync(kBarReady0 + p, kThreads);
    Operand& ob = s.op[p];

    // Prediction and state readout share the packed state fragments.
    float pred[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    float obuf[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      uint32_t afr[4];
      afr[0] = pack2(st[2 * m][0], st[2 * m][1]);
      afr[1] = pack2(st[2 * m][2], st[2 * m][3]);
      afr[2] = pack2(st[2 * m + 1][0], st[2 * m + 1][1]);
      afr[3] = pack2(st[2 * m + 1][2], st[2 * m + 1][3]);
      uint32_t bf[4];
      ldm_x4t(bf, b_addr_dmaj(ob.ktT, m * 16, lane));
      mma_bf16(pred[0], afr, bf[0], bf[1]);
      mma_bf16(pred[1], afr, bf[2], bf[3]);
      ldm_x4t(bf, b_addr_dmaj(ob.qtT, m * 16, lane));
      mma_bf16(obuf[0], afr, bf[0], bf[1]);
      mma_bf16(obuf[1], afr, bf[2], bf[3]);
    }

    // rhs^T = gain*v^T - gain*pred^T, straight from the staged value matrix.
    float rhs[2][4];
    {
      const int t0 = gc4 * 2;
      const float g0 = ob.gain[t0];
      const float g1 = ob.gain[t0 + 1];
      const float g8 = ob.gain[kHalf + t0];
      const float g9 = ob.gain[kHalf + t0 + 1];
      const int vrow = warp * 16 + gq;
      const bf16* v0 = &ob.vt[vrow][t0];
      const bf16* v1 = &ob.vt[vrow + 8][t0];
      // vt already holds gain*v, so only the prediction term is scaled here.
      rhs[0][0] = __bfloat162float(v0[0]) - g0 * pred[0][0];
      rhs[0][1] = __bfloat162float(v0[1]) - g1 * pred[0][1];
      rhs[0][2] = __bfloat162float(v1[0]) - g0 * pred[0][2];
      rhs[0][3] = __bfloat162float(v1[1]) - g1 * pred[0][3];
      rhs[1][0] = __bfloat162float(v0[8]) - g8 * pred[1][0];
      rhs[1][1] = __bfloat162float(v0[9]) - g9 * pred[1][1];
      rhs[1][2] = __bfloat162float(v1[8]) - g8 * pred[1][2];
      rhs[1][3] = __bfloat162float(v1[9]) - g9 * pred[1][3];
    }
    uint32_t rfrag[4];
    rfrag[0] = pack2(rhs[0][0], rhs[0][1]);
    rfrag[1] = pack2(rhs[0][2], rhs[0][3]);
    rfrag[2] = pack2(rhs[1][0], rhs[1][1]);
    rfrag[3] = pack2(rhs[1][2], rhs[1][3]);

    float delta[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    {
      uint32_t bf[4];
      ldm_x4t(bf, b_addr_tmaj16(ob.tt, 0, lane));
      mma_bf16(delta[0], rfrag, bf[0], bf[1]);
      mma_bf16(delta[1], rfrag, bf[2], bf[3]);
    }
    uint32_t dfrag[4];
    dfrag[0] = pack2(delta[0][0], delta[0][1]);
    dfrag[1] = pack2(delta[0][2], delta[0][3]);
    dfrag[2] = pack2(delta[1][0], delta[1][1]);
    dfrag[3] = pack2(delta[1][2], delta[1][3]);

    {
      uint32_t bf[4];
      ldm_x4t(bf, b_addr_tmaj16(ob.sat, 0, lane));
      mma_bf16(obuf[0], dfrag, bf[0], bf[1]);
      mma_bf16(obuf[1], dfrag, bf[2], bf[3]);
    }

    // Stage the readout, then emit it coalesced once every warp is done.
    {
      const int vrow = warp * 16 + gq;
      const int t0 = gc4 * 2;
      bf16(*stage)[kX] = s.o_stage[0];
      stage[t0][vrow] = __float2bfloat16(obuf[0][0]);
      stage[t0 + 1][vrow] = __float2bfloat16(obuf[0][1]);
      stage[t0][vrow + 8] = __float2bfloat16(obuf[0][2]);
      stage[t0 + 1][vrow + 8] = __float2bfloat16(obuf[0][3]);
      stage[kHalf + t0][vrow] = __float2bfloat16(obuf[1][0]);
      stage[kHalf + t0 + 1][vrow] = __float2bfloat16(obuf[1][1]);
      stage[kHalf + t0][vrow + 8] = __float2bfloat16(obuf[1][2]);
      stage[kHalf + t0 + 1][vrow + 8] = __float2bfloat16(obuf[1][3]);
    }
    bar_sync(kBarDense, kDenseThreads);
    {
      const int t = tid >> 4;
      const int seg = tid & 15;
      const long long off =
          (static_cast<long long>(c) * kChunk + t) * row_stride + head * kDim + seg * 8;
      uint4 pack;
      pack.x = *reinterpret_cast<const uint32_t*>(&s.o_stage[0][t][seg * 8]);
      pack.y = *reinterpret_cast<const uint32_t*>(&s.o_stage[0][t][seg * 8 + 2]);
      pack.z = *reinterpret_cast<const uint32_t*>(&s.o_stage[0][t][seg * 8 + 4]);
      pack.w = *reinterpret_cast<const uint32_t*>(&s.o_stage[0][t][seg * 8 + 6]);
      *reinterpret_cast<uint4*>(out + off) = pack;
    }

    // Decay the state, then apply the rank-16 update.
#pragma unroll
    for (int n = 0; n < 16; ++n) {
      const float2 dl = *reinterpret_cast<const float2*>(&ob.dlast[n * 8 + gc4 * 2]);
      st[n][0] *= dl.x;
      st[n][1] *= dl.y;
      st[n][2] *= dl.x;
      st[n][3] *= dl.y;
    }
#pragma unroll
    for (int n = 0; n < 16; n += 2) {
      uint32_t bf[4];
      ldm_x4t(bf, b_addr_tmaj(ob.kpp, n * 8, lane));
      mma_bf16(st[n], dfrag, bf[0], bf[1]);
      mma_bf16(st[n + 1], dfrag, bf[2], bf[3]);
    }
    bar_arrive(kBarFree0 + p, kThreads);
  }

  {
    const int vrow = warp * 16 + gq;
    float* dst = final_state + head * kDim * kDim;
#pragma unroll
    for (int n = 0; n < 16; ++n) {
      const int d = n * 8 + gc4 * 2;
      *reinterpret_cast<float2*>(dst + vrow * kDim + d) = make_float2(st[n][0], st[n][1]);
      *reinterpret_cast<float2*>(dst + (vrow + 8) * kDim + d) = make_float2(st[n][2], st[n][3]);
    }
  }
}

// ---------------------------------------------------------------------------
void kernel(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
            const torch::Tensor& g, const torch::Tensor& beta, const torch::Tensor& scale,
            const torch::Tensor& a_log, const torch::Tensor& dt_bias,
            const torch::Tensor& lower_bound, const torch::Tensor& initial_state,
            const torch::Tensor& output, const torch::Tensor& final_state) {
  TORCH_CHECK_VALUE(q.is_cuda() && k.is_cuda() && v.is_cuda() && g.is_cuda(),
                    "inputs must be CUDA tensors");
  TORCH_CHECK_TYPE(q.scalar_type() == torch::kBFloat16 && k.scalar_type() == torch::kBFloat16 &&
                       v.scalar_type() == torch::kBFloat16 && g.scalar_type() == torch::kBFloat16 &&
                       beta.scalar_type() == torch::kBFloat16,
                   "q/k/v/g/beta must be bfloat16");
  TORCH_CHECK_TYPE(a_log.scalar_type() == torch::kFloat32 &&
                       dt_bias.scalar_type() == torch::kFloat32 &&
                       initial_state.scalar_type() == torch::kFloat32 &&
                       scale.scalar_type() == torch::kFloat32 &&
                       lower_bound.scalar_type() == torch::kFloat32,
                   "scale/a_log/dt_bias/lower_bound/initial_state must be float32");
  TORCH_CHECK_VALUE(q.numel() == static_cast<int64_t>(kHeads) * kTokens * kDim,
                    "unexpected problem size");
  TORCH_CHECK_VALUE(v.size(1) == kTokens && v.size(2) == kHeads && v.size(3) == kDim,
                    "unexpected problem size");
  TORCH_CHECK_VALUE(initial_state.size(0) == 1 && initial_state.size(1) == kHeads &&
                        initial_state.size(2) == kDim && initial_state.size(3) == kDim,
                    "unexpected state size");
  TORCH_CHECK_VALUE(output.scalar_type() == torch::kBFloat16 &&
                        final_state.scalar_type() == torch::kFloat32,
                    "unexpected output dtype");

  c10::cuda::CUDAGuard guard(q.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();

  const int smem = static_cast<int>(sizeof(Shared));
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(kda_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured = true;
  }

  kda_kernel<<<kHeads, kThreads, smem, stream>>>(
      reinterpret_cast<const bf16*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const bf16*>(k.data_ptr<at::BFloat16>()),
      reinterpret_cast<const bf16*>(v.data_ptr<at::BFloat16>()),
      reinterpret_cast<const bf16*>(g.data_ptr<at::BFloat16>()),
      reinterpret_cast<const bf16*>(beta.data_ptr<at::BFloat16>()),
      a_log.data_ptr<float>(), dt_bias.data_ptr<float>(), initial_state.data_ptr<float>(),
      reinterpret_cast<bf16*>(output.data_ptr<at::BFloat16>()),
      final_state.data_ptr<float>(), lower_bound.item<float>(), scale.item<float>());
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("kernel", &kernel);
}

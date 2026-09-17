// Gated delta-net prefill: scalar log-decay, delta-rule state, sequential scan.
//
//   for t in 0..T-1:                    (independently per value head hv)
//     prior_t = S_t * exp(g[t,hv])
//     pred_t  = prior_t^T k_t
//     S_{t+1} = prior_t + k_t (beta_t (v_t - pred_t))^T
//     out_t   = S_{t+1}^T q_t
//
// Mapping: one CTA owns a (value head, value-column block) tile and walks T
// sequentially.  Its state slice lives in registers; all 128 key rows of a
// column sit in one lane group, so both per-token reductions are warp shuffles.
//
// Two tricks keep the scan off the memory-latency critical path:
//   * the state is anchored as S / lam, lam = prod(exp(g)), so the scalar decay
//     costs one multiply per column instead of one multiply per state element.
//     Both lam and its reciprocal run as products seeded by pack_scalars, which
//     removes the per-token reciprocal.  The anchor is renormalised once per
//     kChunk tokens, so its rescale (one multiply per state element) is
//     amortised instead of predicated per token.
//   * k, q, v and the (decay, beta) pair are software-pipelined kDepth tokens
//     ahead in registers, so every load is issued long before it is consumed.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int kK = 128;               // key dim
constexpr int kV = 128;               // value dim
constexpr int kHq = 16;               // query heads
constexpr int kHv = 32;               // value heads
constexpr int kRepeat = kHv / kHq;    // grouped-head expansion
constexpr float kL2Eps = 1e-6f;
constexpr float kRsqrtK = 0.08838834764831845f;  // 1/sqrt(128)

constexpr int kVB = 4;                // value columns per tile
constexpr int kRows = 16;             // state rows per thread
constexpr int kGrp = kK / kRows;      // lanes sharing one column
constexpr int kColsWarp = 32 / kGrp;
constexpr int kWarps = kVB / kColsWarp;
constexpr int kThreads = 32 * kWarps;
constexpr int kVBlocks = kV / kVB;
constexpr int kTiles = kHv * kVBlocks;
constexpr int kDepth = 3;             // scan pipeline depth (tokens)
constexpr int kChunk = 12;            // tokens per anchor renormalisation

// Renormalise while |log(lam)| stays under 21, i.e. inside fp32 normal range
// with several orders of magnitude of slack.
constexpr float kAnchorLo = 1e-9f;
constexpr float kAnchorHi = 1e9f;

// gdn_step either renormalises the anchor inline (few, cold tokens) or leaves
// it to its caller, which amortises the rescale over a whole chunk.
enum Renorm { kRenormInline, kRenormCaller };

// One warp per (token, query head) row of q and k: lanes own 4 consecutive
// elements, so both the 8-byte loads and the 16-byte stores are coalesced.
__global__ void norm_qk(const __nv_bfloat16* __restrict__ q,
                        const __nv_bfloat16* __restrict__ k,
                        float* __restrict__ qn, float* __restrict__ kn,
                        float* __restrict__ kq, const int nrows) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= nrows) return;
  const size_t base = ((size_t)warp << 7) + ((size_t)lane << 2);

  const __nv_bfloat162* qp = reinterpret_cast<const __nv_bfloat162*>(q + base);
  const __nv_bfloat162* kp = reinterpret_cast<const __nv_bfloat162*>(k + base);
  const float2 qa = __bfloat1622float2(qp[0]);
  const float2 qb = __bfloat1622float2(qp[1]);
  const float2 ka = __bfloat1622float2(kp[0]);
  const float2 kb = __bfloat1622float2(kp[1]);

  // k.q of the *output* vectors: the scan folds it into the delta-rule readout
  // so the second state reduction can run on the pre-update anchor.
  float ds = (qa.x * ka.x + qa.y * ka.y) + (qb.x * kb.x + qb.y * kb.y);
  float qs = (qa.x * qa.x + qa.y * qa.y) + (qb.x * qb.x + qb.y * qb.y);
  float ks = (ka.x * ka.x + ka.y * ka.y) + (kb.x * kb.x + kb.y * kb.y);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    qs += __shfl_xor_sync(0xffffffffu, qs, off);
    ks += __shfl_xor_sync(0xffffffffu, ks, off);
    ds += __shfl_xor_sync(0xffffffffu, ds, off);
  }
  const float qinv = rsqrtf(qs + kL2Eps) * kRsqrtK;
  const float kinv = rsqrtf(ks + kL2Eps);

  reinterpret_cast<float4*>(qn)[(base >> 2)] =
      make_float4(qa.x * qinv, qa.y * qinv, qb.x * qinv, qb.y * qinv);
  reinterpret_cast<float4*>(kn)[(base >> 2)] =
      make_float4(ka.x * kinv, ka.y * kinv, kb.x * kinv, kb.y * kinv);
  if (lane == 0) kq[warp] = ds * qinv * kinv;  // scaling folds into the dot
}

// Pack the per-(token, value head) scalars into one 16-byte record holding both
// the decay and its reciprocal, so the scan never evaluates a reciprocal.
__global__ void pack_scalars(const float* __restrict__ g,
                             const __nv_bfloat16* __restrict__ beta,
                             float4* __restrict__ packed, const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float decay = expf(g[i]);
  packed[i] = make_float4(decay, 1.0f / decay, __bfloat162float(beta[i]), 0.0f);
}

// One token of the scan for one lane.  kk/qq/vt/gvb come from the pipeline
// Shift the anchor back into range: move it out of the state and restart both
// running products at one.
template <int ROWS>
__device__ __forceinline__ void reanchor(float (&st)[ROWS], const float lam,
                                         float& acc, float& inv) {
#pragma unroll
  for (int i = 0; i < ROWS; ++i) st[i] *= lam;
  acc = 1.0f;
  inv = 1.0f;
}

template <int ROWS, Renorm RM>
__device__ __forceinline__ void gdn_step(float (&st)[ROWS], const float (&kk)[ROWS],
                                         const float (&qq)[ROWS], const float vt,
                                         const float4 gvb, const float kqd,
                                         float& lam, float& ilam, const int j,
                                         __nv_bfloat16* __restrict__ obase) {
  constexpr int G = kK / ROWS;

  lam *= gvb.x;
  ilam *= gvb.y;
  if (RM == kRenormInline && (lam < kAnchorLo || lam > kAnchorHi))
    reanchor<ROWS>(st, lam, lam, ilam);

  // Both state reductions read the anchor *before* the update, so the two FMA
  // chains interleave and the readout leaves the loop-carried path:
  //   pred = lam*st.k ; readout = lam*(st.q + (k.q)*res/lam)
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
#pragma unroll
  for (int i = 0; i < ROWS; i += 4) {
    a0 = fmaf(st[i + 0], kk[i + 0], a0);
    a1 = fmaf(st[i + 1], kk[i + 1], a1);
    a2 = fmaf(st[i + 2], kk[i + 2], a2);
    a3 = fmaf(st[i + 3], kk[i + 3], a3);
    b0 = fmaf(st[i + 0], qq[i + 0], b0);
    b1 = fmaf(st[i + 1], qq[i + 1], b1);
    b2 = fmaf(st[i + 2], qq[i + 2], b2);
    b3 = fmaf(st[i + 3], qq[i + 3], b3);
  }
  float psum = (a0 + a1) + (a2 + a3);
  float osum = (b0 + b1) + (b2 + b3);
#pragma unroll
  for (int off = G / 2; off > 0; off >>= 1) {
    psum += __shfl_xor_sync(0xffffffffu, psum, off, G);
    osum += __shfl_xor_sync(0xffffffffu, osum, off, G);
  }

  const float rs = gvb.z * (vt - lam * psum) * ilam;
  if (j == 0) *obase = __float2bfloat16(lam * fmaf(kqd, rs, osum));
#pragma unroll
  for (int i = 0; i < ROWS; ++i) st[i] = fmaf(kk[i], rs, st[i]);
}

template <int ROWS, int VB, int DEPTH>
__global__ void __launch_bounds__(32 * (VB / (32 / (kK / ROWS)))) gdn_scan(
    const float* __restrict__ qn, const float* __restrict__ kn,
    const __nv_bfloat16* __restrict__ v, const float4* __restrict__ gb,
    const float* __restrict__ kqv, const float* __restrict__ h0,
    __nv_bfloat16* __restrict__ out, float* __restrict__ hfin, const int T) {
  constexpr int G = kK / ROWS;
  constexpr int COLS_W = 32 / G;
  constexpr int NVB = kV / VB;

  const int hv = blockIdx.x / NVB;
  const int vb = blockIdx.x % NVB;
  const int hq = hv / kRepeat;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int sub = lane / G;
  const int j = lane % G;
  const int gcol = vb * VB + warp * COLS_W + sub;

  // This lane owns rows 4*G*m + 4*j + i of a single value column.
  float st[ROWS];
  {
    const float* p = h0 + ((size_t)hv * kK + 4 * j) * kV + gcol;
#pragma unroll
    for (int m = 0; m < ROWS / 4; ++m) {
#pragma unroll
      for (int i = 0; i < 4; ++i) st[4 * m + i] = p[(size_t)(4 * G * m + i) * kV];
    }
  }

  const __nv_bfloat16* vbase = v + (size_t)hv * kV + gcol;
  __nv_bfloat16* obase = out + (size_t)hv * kV + gcol;
  const float* kbase = kn + (size_t)hq * kK;
  const float* qbase = qn + (size_t)hq * kK;
  const float4* gbase = gb + hv;
  const float* kqbase = kqv + hq;

  float kbuf[DEPTH][ROWS], qbuf[DEPTH][ROWS], vbuf[DEPTH], kqbuf[DEPTH];
  float4 gbuf[DEPTH];

  auto stage = [&](const int t, const int s) {
    const float4* k4 = reinterpret_cast<const float4*>(kbase + (size_t)t * kHq * kK);
    const float4* q4 = reinterpret_cast<const float4*>(qbase + (size_t)t * kHq * kK);
#pragma unroll
    for (int m = 0; m < ROWS / 4; ++m) {
      const float4 ka = k4[G * m + j];
      const float4 qa = q4[G * m + j];
      kbuf[s][4 * m + 0] = ka.x; kbuf[s][4 * m + 1] = ka.y;
      kbuf[s][4 * m + 2] = ka.z; kbuf[s][4 * m + 3] = ka.w;
      qbuf[s][4 * m + 0] = qa.x; qbuf[s][4 * m + 1] = qa.y;
      qbuf[s][4 * m + 2] = qa.z; qbuf[s][4 * m + 3] = qa.w;
    }
    vbuf[s] = __bfloat162float(vbase[(size_t)t * kHv * kV]);
    gbuf[s] = gbase[(size_t)t * kHv];
    kqbuf[s] = kqbase[(size_t)t * kHq];
  };

  float lam = 1.0f;   // prod(decay) seen by this column so far
  float ilam = 1.0f;  // its reciprocal, carried as a product to avoid a divide
#pragma unroll
  for (int s = 0; s < DEPTH; ++s)
    if (s < T) stage(s, s);  // guard for T < DEPTH

  // Steady state: consume slot s, then immediately refill it DEPTH tokens
  // ahead.  Refilling after the use keeps the register array hazard-free while
  // still giving every load a full DEPTH-step head start.  Every loop stops
  // 2*DEPTH short so that refills stay inside [0, T).
  int t = 0;
  for (; t + kChunk + 2 * DEPTH <= T; t += kChunk) {
    // Chunk: the anchor rescale is hoisted out of this loop so the common path
    // pays neither for the rescale nor for a per-token predicate.
    for (int c = 0; c < kChunk; c += DEPTH) {
#pragma unroll
      for (int s = 0; s < DEPTH; ++s) {
        gdn_step<ROWS, kRenormCaller>(st, kbuf[s], qbuf[s], vbuf[s], gbuf[s], kqbuf[s],
                                      lam, ilam, j,
                                      obase + (size_t)(t + c + s) * kHv * kV);
        stage(t + c + DEPTH + s, s);
      }
    }
    if (lam < kAnchorLo || lam > kAnchorHi) reanchor<ROWS>(st, lam, lam, ilam);
  }

  // Short tail: rerun the pipeline with an inline check.
  for (; t + 2 * DEPTH <= T; t += DEPTH) {
#pragma unroll
    for (int s = 0; s < DEPTH; ++s) {
      gdn_step<ROWS, kRenormInline>(st, kbuf[s], qbuf[s], vbuf[s], gbuf[s], kqbuf[s],
                                    lam, ilam, j, obase + (size_t)(t + s) * kHv * kV);
      stage(t + DEPTH + s, s);
    }
  }
  // Tail: the slots still hold tokens t..t+DEPTH-1, so they come out of the
  // pipeline without refilling; at most DEPTH-1 tokens are left after that.
#pragma unroll
  for (int s = 0; s < DEPTH; ++s)
    if (t + s < T)
      gdn_step<ROWS, kRenormInline>(st, kbuf[s], qbuf[s], vbuf[s], gbuf[s], kqbuf[s],
                                    lam, ilam, j, obase + (size_t)(t + s) * kHv * kV);
  for (t += DEPTH; t < T; ++t) {
    stage(t, 0);
    gdn_step<ROWS, kRenormInline>(st, kbuf[0], qbuf[0], vbuf[0], gbuf[0], kqbuf[0],
                                  lam, ilam, j, obase + (size_t)t * kHv * kV);
  }

  {
    float* p = hfin + ((size_t)hv * kK + 4 * j) * kV + gcol;
#pragma unroll
    for (int m = 0; m < ROWS / 4; ++m) {
#pragma unroll
      for (int i = 0; i < 4; ++i) p[(size_t)(4 * G * m + i) * kV] = st[4 * m + i] * lam;
    }
  }
}

void check(const torch::Tensor& t, const torch::ScalarType dt, const int64_t last) {
  TORCH_CHECK_VALUE(t.is_cuda(), "expected CUDA tensor");
  TORCH_CHECK_TYPE(t.scalar_type() == dt, "dtype mismatch");
  TORCH_CHECK_VALUE(t.is_contiguous(), "expected contiguous");
  TORCH_CHECK_VALUE(t.size(t.dim() - 1) == last, "unexpected trailing dim");
}

void kernel(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
            const torch::Tensor& g, const torch::Tensor& beta,
            const torch::Tensor& init_state, const torch::Tensor& out,
            const torch::Tensor& final_state) {
  check(q, torch::kBFloat16, kK);
  check(k, torch::kBFloat16, kK);
  check(v, torch::kBFloat16, kV);
  check(g, torch::kFloat32, kHv);
  check(beta, torch::kBFloat16, kHv);
  check(init_state, torch::kFloat32, kV);
  check(out, torch::kBFloat16, kV);
  check(final_state, torch::kFloat32, kV);
  TORCH_CHECK_VALUE(q.size(2) == kHq && v.size(2) == kHv, "head count mismatch");

  const int64_t T = q.size(1);
  TORCH_CHECK_VALUE(v.size(1) == T && g.size(1) == T && out.size(1) == T, "T mismatch");

  c10::cuda::CUDAGuard guard(q.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();

  const int64_t rows = T * kHq;
  auto f32 = q.options().dtype(torch::kFloat32);
  torch::Tensor qn = torch::empty({rows, kK}, f32);
  torch::Tensor kn = torch::empty({rows, kK}, f32);
  torch::Tensor gb = torch::empty({T * kHv, 4}, f32);
  torch::Tensor kq = torch::empty({rows}, f32);

  const int nb = (int)((rows + 3) / 4);  // 4 warps/block, one row per warp
  norm_qk<<<nb, 128, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
      qn.data_ptr<float>(), kn.data_ptr<float>(), kq.data_ptr<float>(), (int)rows);

  const int np = (int)((T * kHv + 255) / 256);
  pack_scalars<<<np, 256, 0, stream>>>(
      g.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr<at::BFloat16>()),
      reinterpret_cast<float4*>(gb.data_ptr<float>()), (int)(T * kHv));

  gdn_scan<kRows, kVB, kDepth><<<kTiles, kThreads, 0, stream>>>(
      qn.data_ptr<float>(), kn.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
      reinterpret_cast<const float4*>(gb.data_ptr<float>()),
      kq.data_ptr<float>(), init_state.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      final_state.data_ptr<float>(), (int)T);

  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("kernel", &kernel); }

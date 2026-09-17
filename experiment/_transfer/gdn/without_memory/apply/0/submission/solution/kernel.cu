// Gated DeltaNet prefill, raw CUDA, no external library.
//
//   q,k : [T,Hq,K] bf16   v : [T,Hv,V] bf16   g : [T,Hv] fp32
//   beta: [T,Hv]   bf16   s0: [Hv,K,V] fp32
//
// With k^/q^ the L2 normalized rows (q^ also scaled by K^-1/2) the state S
// (K x V, fp32) follows, per value head and token:
//
//   pred = k^.(d*S) ; res = beta*(v - pred) ; S = d*S + k^ (x) res ; out = q^.S
//
// Layout
//   gdn_prep  normalizes q/k rows once into fp32 scratch (head-major, in the
//             scan-friendly permutation below) and emits the per (token,head)
//             scalars decay, 1/decay, q^.k^, beta.
//   gdn_scan  one warp per (value head, 4-column slab). The V axis carries the
//             parallelism; lane = 4*rowGroup + column holds 16 state rows of
//             one column in registers, so the state never leaves the SM.
//
// Scratch permutation
//   A warp covers 4 columns x 128 rows: lane (c,g) owns rows [16g,16g+16) and
//   consumes them as 4 float4 chunks. Storing the row in natural order makes
//   the 8 row groups of one load land 64 B apart, i.e. 4 cache lines per
//   load. Storing chunk m at slot 8*(m%4)+m/4 puts the 8 groups of round i in
//   slots [8i,8i+8) -- one line per load, all bytes useful:
//
//     natural  [g0|g1|g2|g3|g4|g5|g6|g7]        permuted  natural slot order
//     load i   ---64B--- ---64B---   (4 lines)  load i    |g0..g7|  (1 line)
//
// Decay folding
//   the scalar decay is folded out of the state (S~ = S/D, D = prod exp(g))
//   so the state is rescaled once per ~20 tokens instead of every token, and
//   1/D lets the residual be applied without a division; the output uses
//   out = D*(q^.S~) + (q^.k^)*res, so the q^.S~ reduction is independent of
//   the state update. The serial chain per token is only the decay product,
//   the residual and the rank-1 update.
//
// Latency
//   k^/q^ rows are prefetched two tokens ahead into registers (three rotating
//   row buffers); scales and v one step behind them.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kTokens = 4096;
constexpr int kHeadsQ = 16;
constexpr int kHeadsV = 32;
constexpr int kKeyDim = 128;
constexpr int kValDim = 128;
constexpr float kL2Eps = 1e-6f;
constexpr float kQScale = 0.08838834764831845f;   // K^-1/2

constexpr int kWarp = 32;
constexpr int kSlabCols = 4;                      // value columns per warp
constexpr int kSlabs = kValDim / kSlabCols;
constexpr int kGroups = 32 / kSlabCols;           // row groups per warp
constexpr int kRowsLane = kKeyDim / kGroups;      // state rows held by a lane
constexpr int kChunks = kRowsLane / 4;            // float4 chunks per lane
constexpr float kRenorm = 1.0f / 1048576.0f;      // rescale the state below 2^-20

constexpr int kPrepWarps = 4;
constexpr int kPad = 8;                           // scratch rows past the last token
constexpr int kDepth = 3;                         // rotating k/q row buffers

struct alignas(8) Bf16x4 {
  __nv_bfloat162 lo;
  __nv_bfloat162 hi;
};

// per (token, value head): decay, its reciprocal, q^.k^, beta
struct alignas(16) TokenScale {
  float decay;
  float rdecay;
  float dot;
  float rate;
};

constexpr int kRowBytes = kKeyDim * 4;               // one k^/q^ row, fp32
constexpr int kVRowBytes = kValDim * 2;              // one v row, in bytes
constexpr int kVStride = kHeadsV * kVRowBytes;       // v row stride along tokens

__device__ __forceinline__ float4 bf16x4_to_f4(const Bf16x4& w) {
  float4 f;
  f.x = __bfloat162float(w.lo.x);
  f.y = __bfloat162float(w.lo.y);
  f.z = __bfloat162float(w.hi.x);
  f.w = __bfloat162float(w.hi.y);
  return f;
}

// Chunk m of a scratch row lives at float4 slot 8*(m%4) + m/4.
__device__ __forceinline__ int perm_slot(int chunk) {
  return kGroups * (chunk % kChunks) + chunk / kChunks;
}

// One warp per (token, q/k head): row norms, q.k dot product, normalized rows.
__global__ void gdn_prep(const __nv_bfloat16* __restrict__ q,
                         const __nv_bfloat16* __restrict__ k,
                         const float* __restrict__ g,
                         const __nv_bfloat16* __restrict__ beta,
                         float* __restrict__ kh, float* __restrict__ qh,
                         TokenScale* __restrict__ scales) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int token = blockIdx.x * kPrepWarps + warp;
  const int head = blockIdx.y;
  const size_t inRow = ((size_t)token * kHeadsQ + head) * kKeyDim;
  const size_t outRow = ((size_t)head * (kTokens + kPad) + token) * kKeyDim;

  const float4 kf = bf16x4_to_f4(*reinterpret_cast<const Bf16x4*>(k + inRow + 4 * lane));
  const float4 qf = bf16x4_to_f4(*reinterpret_cast<const Bf16x4*>(q + inRow + 4 * lane));
  float ksq = kf.x * kf.x + kf.y * kf.y + kf.z * kf.z + kf.w * kf.w;
  float qsq = qf.x * qf.x + qf.y * qf.y + qf.z * qf.z + qf.w * qf.w;
  float dot = kf.x * qf.x + kf.y * qf.y + kf.z * qf.z + kf.w * qf.w;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    ksq += __shfl_xor_sync(0xffffffffu, ksq, off);
    qsq += __shfl_xor_sync(0xffffffffu, qsq, off);
    dot += __shfl_xor_sync(0xffffffffu, dot, off);
  }

  const float kn = rsqrtf(ksq + kL2Eps);
  const float qn = rsqrtf(qsq + kL2Eps) * kQScale;
  const int slot = perm_slot(lane);
  *reinterpret_cast<float4*>(kh + outRow + 4 * slot) =
      make_float4(kf.x * kn, kf.y * kn, kf.z * kn, kf.w * kn);
  *reinterpret_cast<float4*>(qh + outRow + 4 * slot) =
      make_float4(qf.x * qn, qf.y * qn, qf.z * qn, qf.w * qn);

  if (lane >= 2) return;

  const int hv = 2 * head + lane;
  const float gv = g[(size_t)token * kHeadsV + hv];
  TokenScale s;
  s.decay = __expf(gv);
  s.rdecay = __expf(-gv);
  s.dot = dot * kn * qn;
  s.rate = __bfloat162float(beta[(size_t)token * kHeadsV + hv]);
  scales[(size_t)token * kHeadsV + hv] = s;
}

// k^/q^ of one token for one lane.
struct Row {
  float4 k[kChunks];
  float4 q[kChunks];
};

// Lane group g reads slot kGroups*i + g in round i: 8 groups x 16 B aligned
// inside one 128 B line.
__device__ __forceinline__ void load_row(Row& r, const float* kp, const float* qp,
                                         int g) {
#pragma unroll
  for (int i = 0; i < kChunks; ++i) {
    const int off = (kGroups * i + g) * 4;
    r.k[i] = *reinterpret_cast<const float4*>(kp + off);
    r.q[i] = *reinterpret_cast<const float4*>(qp + off);
  }
}

__device__ __forceinline__ void scan_step(const Row& cur, const TokenScale& sc, float v,
                                          float (&S)[kRowsLane], float& decay,
                                          float& rdecay, __nv_bfloat16* op) {
  float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
  float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
#pragma unroll
  for (int i = 0; i < kChunks; ++i) {
    const float4 k4 = cur.k[i];
    const float4 q4 = cur.q[i];
    p0 = fmaf(k4.x, S[4 * i + 0], p0);
    p1 = fmaf(k4.y, S[4 * i + 1], p1);
    p2 = fmaf(k4.z, S[4 * i + 2], p2);
    p3 = fmaf(k4.w, S[4 * i + 3], p3);
    r0 = fmaf(q4.x, S[4 * i + 0], r0);
    r1 = fmaf(q4.y, S[4 * i + 1], r1);
    r2 = fmaf(q4.z, S[4 * i + 2], r2);
    r3 = fmaf(q4.w, S[4 * i + 3], r3);
  }
  float pr = (p0 + p1) + (p2 + p3);
  float ro = (r0 + r1) + (r2 + r3);

  // Lanes sharing a column differ in bits log2(kSlabCols)..4: butterfly them.
#pragma unroll
  for (int m = kSlabCols; m < kWarp; m <<= 1) {
    pr += __shfl_xor_sync(0xffffffffu, pr, m);
    ro += __shfl_xor_sync(0xffffffffu, ro, m);
  }

  decay *= sc.decay;
  rdecay *= sc.rdecay;

  const float res = sc.rate * (v - decay * pr);
  const float step = res * rdecay;
  *op = __float2bfloat16(decay * ro + sc.dot * res);

#pragma unroll
  for (int i = 0; i < kChunks; ++i) {
    const float4 k4 = cur.k[i];
    S[4 * i + 0] = fmaf(k4.x, step, S[4 * i + 0]);
    S[4 * i + 1] = fmaf(k4.y, step, S[4 * i + 1]);
    S[4 * i + 2] = fmaf(k4.z, step, S[4 * i + 2]);
    S[4 * i + 3] = fmaf(k4.w, step, S[4 * i + 3]);
  }

  if (decay >= kRenorm) return;

#pragma unroll
  for (int i = 0; i < kRowsLane; ++i) S[i] *= decay;
  decay = 1.0f;
  rdecay = 1.0f;
}

// One warp per (value head, 4 column slab); loops the whole token axis.
__global__ __launch_bounds__(kWarp) void gdn_scan(
    const float* __restrict__ kh, const float* __restrict__ qh,
    const __nv_bfloat16* __restrict__ v, const TokenScale* __restrict__ scales,
    const float* __restrict__ state0, __nv_bfloat16* __restrict__ out,
    float* __restrict__ stateN) {
  const int lane = threadIdx.x;
  const int hv = blockIdx.y;
  const int hq = hv >> 1;
  const int col = blockIdx.x * kSlabCols + lane % kSlabCols;
  const int grp = lane / kSlabCols;
  const int rowBase = grp * kRowsLane;

  const size_t colOff = (size_t)hv * (kKeyDim * kValDim) + (size_t)rowBase * kValDim + col;
  float S[kRowsLane];
#pragma unroll
  for (int i = 0; i < kRowsLane; ++i) S[i] = state0[colOff + (size_t)i * kValDim];

  const float* const krow = kh + (size_t)hq * (kTokens + kPad) * kKeyDim;
  const float* const qrow = qh + (size_t)hq * (kTokens + kPad) * kKeyDim;
  const TokenScale* const sp0 = scales + hv;
  const __nv_bfloat16* const vp0 = v + (size_t)hv * kValDim + col;
  __nv_bfloat16* const op = out + (size_t)hv * kValDim + col;
  constexpr int kOutStride = kHeadsV * kValDim;

  // Four running pointers: no per-token index math, every load is an immediate
  // offset off kp/qp/sp/vp/op.
  const float* kp = krow;
  const float* qp = qrow;
  const TokenScale* sp = sp0;
  const __nv_bfloat16* vp = vp0;
  __nv_bfloat16* op0 = op;

  Row A, B, C;
  TokenScale scA, scB, scC;
  float vA, vB, vC;
  load_row(A, kp, qp, grp);
  load_row(B, kp + kKeyDim, qp + kKeyDim, grp);
  load_row(C, kp + 2 * kKeyDim, qp + 2 * kKeyDim, grp);
  scA = sp[0];
  scB = sp[kHeadsV];
  scC = sp[2 * kHeadsV];
  vA = __bfloat162float(vp[0]);
  vB = __bfloat162float(vp[kOutStride]);
  vC = __bfloat162float(vp[2 * kOutStride]);

  float decay = 1.0f, rdecay = 1.0f;

  // Fast path: the deepest load reads token t+kDepth+2, so stop with kDepth+2
  // tokens to spare and let the scalar tail finish them.
  int t = 0;
  for (; t + 2 * kDepth <= kTokens; t += kDepth) {
    scan_step(A, scA, vA, S, decay, rdecay, op0);
    load_row(A, kp + kDepth * kKeyDim, qp + kDepth * kKeyDim, grp);
    scA = sp[kDepth * kHeadsV];
    vA = __bfloat162float(vp[kDepth * kOutStride]);

    scan_step(B, scB, vB, S, decay, rdecay, op0 + kOutStride);
    load_row(B, kp + (kDepth + 1) * kKeyDim, qp + (kDepth + 1) * kKeyDim, grp);
    scB = sp[(kDepth + 1) * kHeadsV];
    vB = __bfloat162float(vp[(kDepth + 1) * kOutStride]);

    scan_step(C, scC, vC, S, decay, rdecay, op0 + 2 * kOutStride);
    load_row(C, kp + (kDepth + 2) * kKeyDim, qp + (kDepth + 2) * kKeyDim, grp);
    scC = sp[(kDepth + 2) * kHeadsV];
    vC = __bfloat162float(vp[(kDepth + 2) * kOutStride]);

    kp += kDepth * kKeyDim;
    qp += kDepth * kKeyDim;
    sp += kDepth * kHeadsV;
    vp += kDepth * kOutStride;
    op0 += kDepth * kOutStride;
  }

  // Tail: at most 2*kDepth-1 tokens, loaded on the spot.
  for (; t < kTokens; ++t) {
    Row cur;
    load_row(cur, kp, qp, grp);
    scan_step(cur, *sp, __bfloat162float(*vp), S, decay, rdecay, op0);
    kp += kKeyDim;
    qp += kKeyDim;
    sp += kHeadsV;
    vp += kOutStride;
    op0 += kOutStride;
  }

#pragma unroll
  for (int i = 0; i < kRowsLane; ++i)
    stateN[colOff + (size_t)i * kValDim] = S[i] * decay;
}

void check(const torch::Tensor& t, const char* name, c10::ScalarType dtype) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == dtype, name, " dtype mismatch");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void kernel(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
            const torch::Tensor& g, const torch::Tensor& beta,
            const torch::Tensor& initial_state, torch::Tensor& output,
            torch::Tensor& final_state) {
  check(q, "q", at::kBFloat16);
  check(k, "k", at::kBFloat16);
  check(v, "v", at::kBFloat16);
  check(g, "g", at::kFloat);
  check(beta, "beta", at::kBFloat16);
  check(initial_state, "initial_state", at::kFloat);
  TORCH_CHECK(q.size(0) == 1 && q.size(1) == kTokens && q.size(2) == kHeadsQ &&
                  q.size(3) == kKeyDim,
              "unexpected q shape");
  TORCH_CHECK(k.sizes() == q.sizes(), "unexpected k shape");
  TORCH_CHECK(v.size(0) == 1 && v.size(1) == kTokens && v.size(2) == kHeadsV &&
                  v.size(3) == kValDim,
              "unexpected v shape");
  TORCH_CHECK(g.numel() == (int64_t)kTokens * kHeadsV, "unexpected g shape");
  TORCH_CHECK(beta.numel() == (int64_t)kTokens * kHeadsV, "unexpected beta shape");
  TORCH_CHECK(initial_state.numel() == (int64_t)kHeadsV * kKeyDim * kValDim,
              "unexpected initial_state shape");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16 && output.is_contiguous() &&
                  output.numel() == (int64_t)kTokens * kHeadsV * kValDim,
              "unexpected output");
  TORCH_CHECK(final_state.scalar_type() == at::kFloat && final_state.is_contiguous() &&
                  final_state.numel() == (int64_t)kHeadsV * kKeyDim * kValDim,
              "unexpected final_state");

  const c10::cuda::CUDAGuard guard(q.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();

  constexpr size_t kRowElem = (size_t)kHeadsQ * (kTokens + kPad) * kKeyDim;
  static float* kh = nullptr;
  static float* qh = nullptr;
  static TokenScale* scales = nullptr;
  if (kh == nullptr) {
    cudaMalloc(reinterpret_cast<void**>(&kh), kRowElem * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&qh), kRowElem * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&scales),
               sizeof(TokenScale) * (size_t)kTokens * kHeadsV);
  }

  gdn_prep<<<dim3(kTokens / kPrepWarps, kHeadsQ), kPrepWarps * kWarp, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
      g.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr<at::BFloat16>()), kh, qh,
      scales);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  gdn_scan<<<dim3(kSlabs, kHeadsV), kWarp, 0, stream>>>(
      kh, qh, reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()), scales,
      initial_state.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      final_state.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("kernel", &kernel);
}

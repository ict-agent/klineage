// Gated DeltaNet prefill (B=1, T=4096, Hq=16, Hv=32, K=V=128).
//
// Sequential scalar-gated delta recurrence. Every state element stays in
// registers for the whole token loop; the value axis of the per-head state
// S[k][v] is split across independent chains so the machine is filled while the
// key axis is reduced cooperatively inside a warp:
//
//   token t:  prior    = decay_t * S
//             pred[v]  = sum_k prior[k][v] * k_t[k]
//             res[v]   = beta_t * (v_t[v] - pred[v])
//             S        = prior + k_t (x) res
//             out_t[v] = sum_k S[k][v] * q_t[k]
//
// Layout
// ------
// A warp holds `LG` lanes per value column and covers `32/LG` groups of `NV`
// columns. Lane j owns `NK = K_DIM/LG` key rows of those columns. Rows are
// assigned in float4 chunks (chunk c = j + LG*t) so that one LDS.128 drives
// 128 contiguous shared bytes per eight-lane phase: no bank conflicts and no
// swizzle.
//
//   warp lane map (LG=8, NV=2):      rows of S            columns of S
//   +----+----+----+----+            j -> 4c..4c+3        grp -> 2 columns
//   | g0 | g1 | g2 | g3 |            c = j + 8t, t<NK/4
//   +----+----+----+----+            (g1..g3 read the same rows: broadcast)
//
// Tokens are streamed through shared memory by cp.async a few tokens ahead so
// global latency never meets the strictly serial token loop.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

namespace {

constexpr int K_DIM = 128;
constexpr int V_DIM = 128;
constexpr int N_Q_HEADS = 16;
constexpr int N_V_HEADS = 32;
constexpr int QK_STRIDE = 2 * K_DIM;
constexpr float NORM_EPS = 1e-6f;
constexpr float K_SCALE = 0.08838834764831845f;  // K_DIM ** -0.5

// ---------------------------------------------------------------------------
// Prepare: q/k row normalization and gather of {exp(g), beta}.
// ---------------------------------------------------------------------------

constexpr int PREP_WARPS = 8;                     // rows per prepare block
constexpr int PREP_THREADS = 32 * PREP_WARPS;
constexpr int GATE_PER_BLOCK = PREP_THREADS * 4;  // (token, head) pairs per block

__device__ __forceinline__ float2 unpack_bf16x2(uint32_t word) {
  // bf16 is the high half of an fp32: shifting reinterprets it exactly.
  return make_float2(__uint_as_float(word << 16), __uint_as_float(word & 0xffff0000u));
}

// One warp per (token, q-head) row; 32 lanes x 4 elements cover the 128 keys.
__device__ __forceinline__ void normalize_row(const __nv_bfloat16* __restrict__ q,
                                              const __nv_bfloat16* __restrict__ k,
                                              float* __restrict__ qk, int lane) {
  const uint32_t qw0 = *reinterpret_cast<const uint32_t*>(q + 4 * lane);
  const uint32_t qw1 = *reinterpret_cast<const uint32_t*>(q + 4 * lane + 2);
  const uint32_t kw0 = *reinterpret_cast<const uint32_t*>(k + 4 * lane);
  const uint32_t kw1 = *reinterpret_cast<const uint32_t*>(k + 4 * lane + 2);
  const float2 qa = unpack_bf16x2(qw0), qb = unpack_bf16x2(qw1);
  const float2 ka = unpack_bf16x2(kw0), kb = unpack_bf16x2(kw1);

  float qs = fmaf(qa.x, qa.x, fmaf(qa.y, qa.y, fmaf(qb.x, qb.x, qb.y * qb.y)));
  float ks = fmaf(ka.x, ka.x, fmaf(ka.y, ka.y, fmaf(kb.x, kb.x, kb.y * kb.y)));
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    qs += __shfl_xor_sync(0xffffffffu, qs, off);
    ks += __shfl_xor_sync(0xffffffffu, ks, off);
  }
  const float qscale = rsqrtf(qs + NORM_EPS) * K_SCALE;
  const float kscale = rsqrtf(ks + NORM_EPS);
  *reinterpret_cast<float4*>(qk + 4 * lane) =
      make_float4(qa.x * qscale, qa.y * qscale, qb.x * qscale, qb.y * qscale);
  *reinterpret_cast<float4*>(qk + K_DIM + 4 * lane) =
      make_float4(ka.x * kscale, ka.y * kscale, kb.x * kscale, kb.y * kscale);
}

__global__ void __launch_bounds__(PREP_THREADS) prepare_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const float* __restrict__ g, const __nv_bfloat16* __restrict__ beta,
    float* __restrict__ qk, float2* __restrict__ gb, int rows, int gate_count,
    int row_blocks) {
  if (blockIdx.x >= row_blocks) {
    // {decay, beta} gather: 4 consecutive (token, value-head) pairs per thread.
    const int base = (blockIdx.x - row_blocks) * GATE_PER_BLOCK + threadIdx.x * 4;
    if (base >= gate_count) return;
    const float4 gv = *reinterpret_cast<const float4*>(g + base);
    const uint2 bw = *reinterpret_cast<const uint2*>(beta + base);
    const float2 b0 = unpack_bf16x2(bw.x), b1 = unpack_bf16x2(bw.y);
    const float4 d =
        make_float4(__expf(gv.x), __expf(gv.y), __expf(gv.z), __expf(gv.w));
    *reinterpret_cast<float4*>(gb + base) = make_float4(d.x, b0.x, d.y, b0.y);
    *reinterpret_cast<float4*>(gb + base + 2) = make_float4(d.z, b1.x, d.w, b1.y);
    return;
  }
  const int row = blockIdx.x * PREP_WARPS + (threadIdx.x >> 5);
  if (row >= rows) return;
  normalize_row(q + (size_t)row * K_DIM, k + (size_t)row * K_DIM,
                qk + (size_t)row * QK_STRIDE, threadIdx.x & 31);
}

// ---------------------------------------------------------------------------
// Recurrence.
// ---------------------------------------------------------------------------

constexpr int LG = 8;       // lanes cooperating on one value column
constexpr int NV = 2;       // value columns owned by one lane
constexpr int WARPS = 2;    // warps per CTA
constexpr int TOK = 4;      // tokens per pipeline stage
constexpr int STAGES = 4;   // cp.async stages
constexpr int ACC = 4;      // split accumulators shorten the FMA dependency chain

// Lane j owns key rows 4*(j + LG*t) .. +3 for t < NK/4.
template <int LG_>
__device__ __forceinline__ int row_of(int j, int i) {
  return 4 * (j + LG_ * (i >> 2)) + (i & 3);
}

// Component w of a two-wide value pair (w is a compile-time constant).
__device__ __forceinline__ float comp2(const float2& v, int w) {
  return w == 0 ? v.x : v.y;
}

template <int LG_, int NV_>
__device__ __forceinline__ void group_sum(float (&x)[NV_]) {
  #pragma unroll
  for (int off = 1; off < LG_; off <<= 1)
    #pragma unroll
    for (int w = 0; w < NV_; ++w)
      x[w] += __shfl_xor_sync(0xffffffffu, x[w], off, LG_);
}

// Stage TOK tokens: q/k rows (1 KiB each), value slice, gate scalars.
// Every copy is a compile-time-unrolled, uniform-predicated cp.async: the row
// is copied one 16-byte chunk per thread, so no inner loop or extra address
// arithmetic reaches the machine.
template <int CTAS_V, int TOK_, int THREADS>
__device__ __forceinline__ void stage_issue(float (*s_qk)[QK_STRIDE],
                                            __nv_bfloat16 (*s_v)[CTAS_V],
                                            float2* s_g,
                                            const float* __restrict__ qk,
                                            const __nv_bfloat16* __restrict__ v,
                                            const float2* __restrict__ gb,
                                            int b, int hq, int hv, int cta_v,
                                            int t0, int T) {
  constexpr int QK_CHUNKS = QK_STRIDE / 4;  // 16-byte chunks per q/k row
  constexpr int QK_COPIES = (QK_CHUNKS + THREADS - 1) / THREADS;
  constexpr int V_CHUNKS = CTAS_V / 8;      // 16 bytes of bf16 values
  const int tid = threadIdx.x;
  const size_t stride_t = (size_t)N_Q_HEADS * QK_STRIDE;

  #pragma unroll
  for (int j = 0; j < TOK_; ++j) {
    if (t0 + j >= T) break;
    const float* src = qk + ((size_t)(b * T + t0 + j) * N_Q_HEADS + hq) * QK_STRIDE;
    #pragma unroll
    for (int i = 0; i < QK_COPIES; ++i) {
      const int c = i * THREADS + tid;
      if (QK_CHUNKS % THREADS == 0 || c < QK_CHUNKS)
        __pipeline_memcpy_async(s_qk[j] + 4 * c, src + 4 * c, 16);
    }
  }
  // value slice and gate pair: one 16-byte / one 8-byte copy per owner thread.
  #pragma unroll
  for (int j = 0; j < TOK_; ++j) {
    if (t0 + j >= T) break;
    const __nv_bfloat16* vsrc =
        v + ((size_t)(b * T + t0 + j) * N_V_HEADS + hv) * V_DIM + cta_v;
    if (tid < V_CHUNKS)
      __pipeline_memcpy_async(s_v[j] + 8 * tid, vsrc + 8 * tid, 16);
  }
  if (tid < TOK_ && t0 + tid < T)
    __pipeline_memcpy_async(s_g + tid,
                            gb + (size_t)(b * T + t0 + tid) * N_V_HEADS + hv, 8);
}

template <int LG_, int NV_, int WARPS_, int TOK_, int STAGES_, int ACC_>
__global__ void __launch_bounds__(WARPS_ * 32) recur_kernel(
    const float* __restrict__ qk, const float2* __restrict__ gb,
    const __nv_bfloat16* __restrict__ v, const float* __restrict__ s0,
    __nv_bfloat16* __restrict__ out, float* __restrict__ sf, int T) {
  constexpr int NK = K_DIM / LG_;       // key rows per lane
  constexpr int NG = 32 / LG_;          // lane groups per warp
  constexpr int VPW = NG * NV_;         // value columns per warp
  constexpr int CTAS_V = VPW * WARPS_;  // value columns per CTA
  constexpr int V_SPLIT = V_DIM / CTAS_V;
  constexpr int THREADS = WARPS_ * 32;
  constexpr int ROUNDS = NK / 4;        // float4 chunks per lane
  static_assert(V_SPLIT * CTAS_V == V_DIM, "value block must tile the head");
  static_assert(NK % 4 == 0 && LG_ * ROUNDS == K_DIM / 4, "chunk mapping");

  __shared__ float s_qk[STAGES_][TOK_][QK_STRIDE];
  __shared__ __nv_bfloat16 s_v[STAGES_][TOK_][CTAS_V];
  __shared__ float2 s_g[STAGES_][TOK_];

  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int b = blockIdx.z;
  const int hv = blockIdx.y;
  const int hq = hv >> 1;  // two value heads share one q/k head
  const int j = lane & (LG_ - 1);
  const int grp = lane / LG_;
  const int cta_v = blockIdx.x * CTAS_V;
  const int vl = wid * VPW + grp * NV_;  // first owned column inside the CTA

  // This lane owns rows row_of(j, i) of value columns [vl, vl+NV).
  float st[NK][NV_];
  const float* sp = s0 + ((size_t)(b * N_V_HEADS + hv) * K_DIM) * V_DIM +
                    cta_v + vl;
  #pragma unroll
  for (int i = 0; i < NK; ++i) {
    const int row = row_of<LG_>(j, i);
    #pragma unroll
    for (int w = 0; w < NV_; ++w) st[i][w] = sp[(size_t)row * V_DIM + w];
  }

  const int n_stage = (T + TOK_ - 1) / TOK_;
  #pragma unroll
  for (int s = 0; s < STAGES_ - 1; ++s) {
    if (s < n_stage)
      stage_issue<CTAS_V, TOK_, THREADS>(s_qk[s], s_v[s], s_g[s], qk, v, gb, b,
                                         hq, hv, cta_v, s * TOK_, T);
    __pipeline_commit();
  }

  for (int s = 0; s < n_stage; ++s) {
    const int cur = s % STAGES_;
    const int nxt = s + STAGES_ - 1;
    if (nxt < n_stage)
      stage_issue<CTAS_V, TOK_, THREADS>(s_qk[nxt % STAGES_], s_v[nxt % STAGES_],
                                         s_g[nxt % STAGES_], qk, v, gb, b, hq, hv,
                                         cta_v, nxt * TOK_, T);
    __pipeline_commit();
    __pipeline_wait_prior(STAGES_ - 1);
    __syncthreads();

    // Gate pairs and value slices for the whole stage, loaded once so their
    // shared-memory latency overlaps the first token's arithmetic.
    float2 gv[TOK_], vf[TOK_];
    #pragma unroll
    for (int u = 0; u < TOK_; ++u)
      gv[u] = s_g[cur][u];
    #pragma unroll
    for (int u = 0; u < TOK_; ++u)
      vf[u] = __bfloat1622float2(
          *reinterpret_cast<const __nv_bfloat162*>(s_v[cur][u] + vl));

    const int t_end = min(TOK_, T - s * TOK_);
    #pragma unroll
    for (int u = 0; u < TOK_; ++u) {
      if (u >= t_end) break;
      const float dec = gv[u].x;
      const float bta = gv[u].y;
      const float* qrow = s_qk[cur][u];
      const float* krow = qrow + K_DIM;

      // This lane's key and query slice, one LDS.128 per float4 chunk.
      float kk[NK], qq[NK];
      #pragma unroll
      for (int t = 0; t < ROUNDS; ++t) {
        const int c = j + LG_ * t;
        *reinterpret_cast<float4*>(&kk[4 * t]) =
            *reinterpret_cast<const float4*>(krow + 4 * c);
        *reinterpret_cast<float4*>(&qq[4 * t]) =
            *reinterpret_cast<const float4*>(qrow + 4 * c);
      }

      // Decay the state into place and take the prediction partials.
      float pp[ACC_][NV_];
      #pragma unroll
      for (int a = 0; a < ACC_; ++a)
        #pragma unroll
        for (int w = 0; w < NV_; ++w) pp[a][w] = 0.f;
      #pragma unroll
      for (int i = 0; i < NK; ++i)
        #pragma unroll
        for (int w = 0; w < NV_; ++w) {
          const float p = dec * st[i][w];
          pp[i & (ACC_ - 1)][w] = fmaf(p, kk[i], pp[i & (ACC_ - 1)][w]);
          st[i][w] = p;
        }
      #pragma unroll
      for (int a = 1; a < ACC_; ++a)
        #pragma unroll
        for (int w = 0; w < NV_; ++w) pp[0][w] += pp[a][w];
      group_sum<LG_, NV_>(pp[0]);

      float res[NV_];
      #pragma unroll
      for (int w = 0; w < NV_; ++w) res[w] = bta * (comp2(vf[u], w) - pp[0][w]);

      // Rank-1 state update fused with the read-out projection.
      float oo[ACC_][NV_];
      #pragma unroll
      for (int a = 0; a < ACC_; ++a)
        #pragma unroll
        for (int w = 0; w < NV_; ++w) oo[a][w] = 0.f;
      #pragma unroll
      for (int i = 0; i < NK; ++i)
        #pragma unroll
        for (int w = 0; w < NV_; ++w) {
          const float sn = fmaf(kk[i], res[w], st[i][w]);
          st[i][w] = sn;
          oo[i & (ACC_ - 1)][w] = fmaf(sn, qq[i], oo[i & (ACC_ - 1)][w]);
        }
      #pragma unroll
      for (int a = 1; a < ACC_; ++a)
        #pragma unroll
        for (int w = 0; w < NV_; ++w) oo[0][w] += oo[a][w];
      group_sum<LG_, NV_>(oo[0]);

      if (j == 0) {
        __nv_bfloat16* ow = out + ((size_t)b * T * N_V_HEADS + hv) * V_DIM +
                            (size_t)(s * TOK_ + u) * N_V_HEADS * V_DIM + cta_v + vl;
        *reinterpret_cast<__nv_bfloat162*>(ow) =
            __floats2bfloat162_rn(oo[0][0], oo[0][1]);
      }
    }
    __syncthreads();
  }

  float* fp = sf + ((size_t)(b * N_V_HEADS + hv) * K_DIM) * V_DIM + cta_v + vl;
  #pragma unroll
  for (int i = 0; i < NK; ++i) {
    const int row = row_of<LG_>(j, i);
    #pragma unroll
    for (int w = 0; w < NV_; ++w) fp[(size_t)row * V_DIM + w] = st[i][w];
  }
}

// ---------------------------------------------------------------------------
// Host wrapper
// ---------------------------------------------------------------------------

void check_input(const torch::Tensor& tensor, int64_t dims, const char* name) {
  TORCH_CHECK_VALUE(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK_VALUE(tensor.dim() == dims, name, " must have ", dims, " dims");
  TORCH_CHECK_VALUE(tensor.is_contiguous(), name, " must be contiguous");
}

void kernel(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
            const torch::Tensor& g, const torch::Tensor& beta,
            const torch::Tensor& initial_state, torch::Tensor& output,
            torch::Tensor& final_state) {
  check_input(q, 4, "q");
  check_input(k, 4, "k");
  check_input(v, 4, "v");
  check_input(g, 3, "g");
  check_input(beta, 3, "beta");
  check_input(initial_state, 4, "initial_state");
  check_input(output, 4, "output");
  check_input(final_state, 4, "final_state");
  TORCH_CHECK_TYPE(q.scalar_type() == torch::kBFloat16, "q must be bfloat16");
  TORCH_CHECK_TYPE(k.scalar_type() == torch::kBFloat16, "k must be bfloat16");
  TORCH_CHECK_TYPE(v.scalar_type() == torch::kBFloat16, "v must be bfloat16");
  TORCH_CHECK_TYPE(g.scalar_type() == torch::kFloat32, "g must be float32");
  TORCH_CHECK_TYPE(beta.scalar_type() == torch::kBFloat16, "beta must be bfloat16");
  TORCH_CHECK_TYPE(initial_state.scalar_type() == torch::kFloat32,
                   "initial_state must be float32");
  TORCH_CHECK_VALUE(q.size(2) == N_Q_HEADS && q.size(3) == K_DIM, "q shape mismatch");
  TORCH_CHECK_VALUE(k.size(2) == N_Q_HEADS && k.size(3) == K_DIM, "k shape mismatch");
  TORCH_CHECK_VALUE(v.size(2) == N_V_HEADS && v.size(3) == V_DIM, "v shape mismatch");

  const int64_t B = q.size(0);
  const int64_t T = q.size(1);
  TORCH_CHECK_VALUE(g.size(0) == B && g.size(1) == T && g.size(2) == N_V_HEADS,
                    "g shape mismatch");
  TORCH_CHECK_VALUE(initial_state.size(0) == B &&
                        initial_state.size(1) == N_V_HEADS &&
                        initial_state.size(2) == K_DIM &&
                        initial_state.size(3) == V_DIM,
                    "initial_state shape mismatch");
  if (B == 0 || T == 0) return;

  c10::cuda::CUDAGuard guard(q.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();

  const auto f32 = torch::TensorOptions().device(q.device()).dtype(torch::kFloat32);
  torch::Tensor qk = torch::empty({B, T, N_Q_HEADS, QK_STRIDE}, f32);
  torch::Tensor gb = torch::empty({B, T, N_V_HEADS, 2}, f32);

  const int rows = (int)(B * T * N_Q_HEADS);
  const int gate = (int)(B * T * N_V_HEADS);
  const int row_blocks = (rows + PREP_WARPS - 1) / PREP_WARPS;
  const int gate_blocks = (gate + GATE_PER_BLOCK - 1) / GATE_PER_BLOCK;
  prepare_kernel<<<row_blocks + gate_blocks, PREP_THREADS, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()), g.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr()),
      qk.data_ptr<float>(), reinterpret_cast<float2*>(gb.data_ptr<float>()), rows,
      gate, row_blocks);

  constexpr int CTAS_V = (32 / LG) * NV * WARPS;
  constexpr int V_SPLIT = V_DIM / CTAS_V;
  recur_kernel<LG, NV, WARPS, TOK, STAGES, ACC>
      <<<dim3(V_SPLIT, N_V_HEADS, (unsigned)B), WARPS * 32, 0, stream>>>(
          qk.data_ptr<float>(), reinterpret_cast<const float2*>(gb.data_ptr<float>()),
          reinterpret_cast<const __nv_bfloat16*>(v.data_ptr()),
          initial_state.data_ptr<float>(),
          reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
          final_state.data_ptr<float>(), (int)T);

  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("kernel", &kernel);
}

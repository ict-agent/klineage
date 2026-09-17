// Block-sparse causal attention (GQA), head dim 128, bf16 in/out, fp32 accum.
//
//   out[b,h,i,:] = softmax_k( q[b,h,i,:] . K[k,:] / sqrt(D) ) V[k,:]
//
// K/V come from the blocks listed in block_ids for the query block owning row i,
// restricted to keys whose position is not after i.
//
// One CTA handles one (batch, query block, query head): 128 queries, 1 head.
// Two warpgroups of 64 queries each drive Hopper WGMMA (m64n128k16). K/V tiles
// stream through a two stage cp.async shared pipeline. Softmax runs on the QK
// accumulators, and the packed probabilities feed PV directly as register A
// operands, so no probability ever touches memory.
//
//   grid (HQ, NB, B)      shared: Q[32K] K0[32K] V0[32K] K1[32K] V1[32K]
//   warpgroup wg -> query rows 64wg..64wg+63
//
// Operands live in the unswizzled WGMMA panel layout: element (r,c) of an
// operand with R rows sits at half offset (c/8)*R*8 + r*8 + c%8, so a 16 byte
// global chunk (8 contiguous columns) lands as one contiguous 16 byte unit and
// cp.async writes it directly.
//
//   Q,K : K-major descriptor (leading = R, stride = 8)
//   V   : MN-major descriptor (leading = 8, stride = R) consumed with trans-b

#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

using bf16 = __nv_bfloat16;

// WGMMA and the 90a feature set need an arch specific target.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define BSA_WGMMA 1
#endif

#define DEV __device__ __forceinline__

constexpr int HEAD_DIM = 128;
constexpr int BLOCK_Q = 128;  // queries per CTA
constexpr int BLOCK_KV = 128; // keys per selected block
constexpr int SLOTS = 16;     // block_ids slots per query block
constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;
constexpr int WG_ROWS = 64;   // queries per warpgroup
constexpr int K_STEPS = HEAD_DIM / 16;
constexpr int N_TILES = 16;   // n8 tiles per 128 key row
constexpr float SCALE = 0.08838834764831845f * 1.4426950408889634f;
constexpr float NEG_INF = -INFINITY;

// Shared panel layout sizes, in bf16 elements.
constexpr int PANEL = BLOCK_KV * 8;         // halves per 8 column panel
constexpr int CHUNK = 8;                    // halves per 16 byte TMA unit
constexpr int TILE = BLOCK_KV * HEAD_DIM;   // halves per Q/K/V tile
constexpr int STAGES = 3;  // K/V buffer rings
constexpr int Q_OFF = 0;
constexpr int K_OFF(int i) { return (1 + 2 * i) * TILE; }
constexpr int V_OFF(int i) { return (2 + 2 * i) * TILE; }

DEV uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

DEV void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src));
}

DEV void cp_commit() { asm volatile("cp.async.commit_group;"); }

template <int PENDING>
DEV void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(PENDING));
}

DEV void proxy_fence() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

// One elected thread feeds a whole 128x128 tile from global straight into the
// WGMMA panel layout, so the copy never crosses the LSU data pipe.
DEV void bar_init(uint64_t* bar, uint32_t arrivals) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(arrivals));
}

DEV void bar_fence() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }

DEV void bar_expect(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)),
               "r"(bytes));
}

DEV void bar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred P;\n"
      "W%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
      "@!P bra W%=;\n\t}" ::"r"(smem_u32(bar)),
      "r"(parity));
}

DEV void tma_tile(const CUtensorMap* map, bf16* dst, int row0, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(smem_u32(dst)),
      "l"((uint64_t)map), "r"(0), "r"(row0), "r"(0), "r"(smem_u32(bar))
      : "memory");
}

DEV void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }

DEV void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }

template <int N>
DEV void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}

DEV uint64_t descriptor(const void* p, int leading, int stride) {
  // Offsets are in 16 byte units, no swizzle.
  return (static_cast<uint64_t>(stride) << 32) | (static_cast<uint64_t>(leading) << 16) |
         static_cast<uint64_t>(smem_u32(p) >> 4);
}

// ex2.approx.f32 keeps the exponential single issue and flag independent.
DEV float fast_exp2(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

DEV uint32_t pack2(float lo, float hi) {
  __nv_bfloat162 pair = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&pair);
}

DEV float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}

DEV float quad_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

#ifdef BSA_WGMMA
#define D64 "%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15," \
            "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31," \
            "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47," \
            "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63"
#define ACC64(d) "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]), \
  "+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]), \
  "+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]), \
  "+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]), \
  "+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]), \
  "+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]), \
  "+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]), \
  "+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63])

// QK: A and B both from shared. add == 0 clears the accumulators.
DEV void wg_qk(float (&d)[64], uint64_t a, uint64_t b, int add) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %66, 0;\n\t"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{" D64 "}, %64, %65, p, 1, 1, 0, 0;\n\t}"
      : ACC64(d)
      : "l"(a), "l"(b), "r"(add)
      : "memory");
}

// PV: A packed probabilities in registers, B (V) from shared, MN-major.
DEV void wg_pv(float (&d)[64], const uint32_t* a, uint64_t b) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{" D64 "}, {%64,%65,%66,%67}, %68, 1, 1, 1, 1;\n\t"
      : ACC64(d)
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b)
      : "memory");
}

template <int N>
DEV void reg_fence(float (&x)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(x[i]) :: "memory");
}

template <int N>
DEV void reg_fence(uint32_t (&x)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+r"(x[i]) :: "memory");
}

// Stage one 128x128 bf16 tile into the panel layout: 16 byte global chunks,
// eight per thread.
DEV void load_tile(bf16* dst, const bf16* src, int tid) {
#pragma unroll
  for (int i = 0; i < (BLOCK_KV * (HEAD_DIM / 8)) / THREADS; ++i) {
    const int idx = tid + i * THREADS;
    const int row = idx >> 4;
    const int panel = idx & 15;
    cp_async16(dst + panel * PANEL + row * 8, src + row * HEAD_DIM + panel * 8);
  }
}

// QK for one 128 key block: eight m64n128k16 steps, query rows in panel *base*.
DEV void qk_block(float (&s)[64], const bf16* qpanel, const bf16* ktile) {
  reg_fence(s);
  wg_fence();
  const uint64_t da = descriptor(qpanel, BLOCK_Q, 8);
  const uint64_t dk = descriptor(ktile, BLOCK_KV, 8);
#pragma unroll
  for (int st = 0; st < K_STEPS; ++st) wg_qk(s, da + st * 2 * BLOCK_Q, dk + st * 2 * BLOCK_KV, st);
  wg_commit();
  wg_wait<0>();
  reg_fence(s);
}

DEV void pv_block(float (&o)[64], const uint32_t (&p)[32], const bf16* vtile) {
  reg_fence(o);
  wg_fence();
  const uint64_t dv = descriptor(vtile, 8, BLOCK_KV);
#pragma unroll
  for (int st = 0; st < K_STEPS; ++st) wg_pv(o, p + 4 * st, dv + st * 16);
  wg_commit();
  wg_wait<0>();
  reg_fence(o);
}

// Drop keys past the query row; only the diagonal block can hold such keys.
DEV void mask_block(float (&s)[64], int key0, int qrow0, int qrow1, int lane) {
  const int c2 = (lane & 3) * 2;
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    const int key = key0 + (i / 4) * 8 + c2 + (i & 1);
    const int row = (i & 2) ? qrow1 : qrow0;
    if (key > row) s[i] = NEG_INF;
  }
}

// Online softmax over 128 keys held by the QK accumulators, then pack the
// probabilities into the register A fragments consumed by PV.
DEV void softmax_block(float (&s)[64], uint32_t (&p)[32], float& m0, float& m1, float& l0,
                       float& l1, float& a0, float& a1, int lane) {
  float mx0 = NEG_INF, mx1 = NEG_INF;
#pragma unroll
  for (int i = 0; i < 64; ++i) {
    if (i & 2)
      mx1 = fmaxf(mx1, s[i]);
    else
      mx0 = fmaxf(mx0, s[i]);
  }
  mx0 = quad_max(mx0);
  mx1 = quad_max(mx1);

  const float mn0 = fmaxf(m0, mx0);
  const float mn1 = fmaxf(m1, mx1);
  a0 = fast_exp2(m0 - mn0);
  a1 = fast_exp2(m1 - mn1);
  m0 = mn0;
  m1 = mn1;

  float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
  for (int u = 0; u < 8; ++u) {
    const float v0 = fast_exp2(s[8 * u + 0] - mn0);
    const float v1 = fast_exp2(s[8 * u + 1] - mn0);
    const float v4 = fast_exp2(s[8 * u + 4] - mn0);
    const float v5 = fast_exp2(s[8 * u + 5] - mn0);
    const float v2 = fast_exp2(s[8 * u + 2] - mn1);
    const float v3 = fast_exp2(s[8 * u + 3] - mn1);
    const float v6 = fast_exp2(s[8 * u + 6] - mn1);
    const float v7 = fast_exp2(s[8 * u + 7] - mn1);
    sum0 += (v0 + v1) + (v4 + v5);
    sum1 += (v2 + v3) + (v6 + v7);
    p[4 * u + 0] = pack2(v0, v1);
    p[4 * u + 1] = pack2(v2, v3);
    p[4 * u + 2] = pack2(v4, v5);
    p[4 * u + 3] = pack2(v6, v7);
  }
  l0 = l0 * a0 + quad_sum(sum0);
  l1 = l1 * a1 + quad_sum(sum1);
}

DEV void rescale_o(float (&o)[64], float a0, float a1) {
#pragma unroll
  for (int i = 0; i < 64; ++i) o[i] *= (i & 2) ? a1 : a0;
}
#endif  // BSA_WGMMA

__global__ void __launch_bounds__(THREADS, 1)
bsa_kernel(const __grid_constant__ CUtensorMap kmap, const __grid_constant__ CUtensorMap vmap,
           const bf16* __restrict__ q, const int* __restrict__ block_ids,
           const int* __restrict__ block_counts, bf16* __restrict__ out, int nq, int nkv, int seq,
           int nblocks) {
#ifdef BSA_WGMMA
  extern __shared__ bf16 smem[];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wg = warp >> 2;                       // warpgroup: 64 query rows each
  const int wg_row = (warp & 3) * 16;             // row offset inside the warpgroup
  const int batch = blockIdx.z;
  const int head = blockIdx.x;
  const int qb = gridDim.y - 1 - blockIdx.y;      // heaviest query blocks first
  const int kv_head = head / (nq / nkv);
  const int g = lane >> 2;
  const int c2 = (lane & 3) * 2;

  const bf16* qsrc = q + ((int64_t)(batch * nq + head) * seq + qb * BLOCK_Q) * HEAD_DIM;
  load_tile(smem + Q_OFF, qsrc, tid);
  cp_commit();
  cp_wait<0>();  // Q resident for the whole CTA
  proxy_fence();
  __syncthreads();

  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + (V_OFF(STAGES - 1) + TILE));
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < STAGES; ++i) bar_init(bars + i, 1);
    bar_fence();
  }
  __syncthreads();

  const int slot = (batch * nq + head) * nblocks + qb;
  const int count = block_counts[slot];
  const int* ids = block_ids + (int64_t)slot * SLOTS;
  const int kv_row = (batch * nkv + kv_head) * seq;

  // Fill the ring: block s lands in stage s % STAGES.
  if (tid == 0) {
#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i)
      if (i < count) {
        const int bid = ids[i];
        bf16* kt = smem + K_OFF(i);
        bf16* vt = smem + V_OFF(i);
        bar_expect(bars + i, 2 * TILE * (uint32_t)sizeof(bf16));
        tma_tile(&kmap, kt, kv_row + bid * BLOCK_KV, bars + i);
        tma_tile(&vmap, vt, kv_row + bid * BLOCK_KV, bars + i);
      }
  }

  const bf16* qpanel = smem + Q_OFF + wg * WG_ROWS * 8;
  const int qrow0 = wg * WG_ROWS + wg_row + g;
  const int qrow1 = qrow0 + 8;

  float o[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) o[i] = 0.f;
  float m0 = NEG_INF, m1 = NEG_INF, l0 = 0.f, l1 = 0.f;

  for (int s = 0; s < count; ++s) {
    const int cur = s % STAGES;
    bf16* kt = smem + K_OFF(cur);
    bf16* vt = smem + V_OFF(cur);

    // Thread zero observes the tile landing; the barrier also proves every
    // warp finished reading the stage this iteration refills.
    if (tid == 0) bar_wait(bars + cur, (uint32_t)((s / STAGES) & 1));
    __syncthreads();
    if (tid == 0 && s + STAGES - 1 < count) {
      const int nxt = (s + STAGES - 1) % STAGES;
      const int bid = ids[s + STAGES - 1];
      bar_expect(bars + nxt, 2 * TILE * (uint32_t)sizeof(bf16));
      tma_tile(&kmap, smem + K_OFF(nxt), kv_row + bid * BLOCK_KV, bars + nxt);
      tma_tile(&vmap, smem + V_OFF(nxt), kv_row + bid * BLOCK_KV, bars + nxt);
    }

    const int bid = ids[s];
    const bool dead = bid > qb;

    float sacc[64];
    qk_block(sacc, qpanel, kt);
    if (dead) {
#pragma unroll
      for (int i = 0; i < 64; ++i) sacc[i] = NEG_INF;
    } else if (bid == qb) {
      mask_block(sacc, bid * BLOCK_KV, qb * BLOCK_Q + qrow0, qb * BLOCK_Q + qrow1, lane);
    }
#pragma unroll
    for (int i = 0; i < 64; ++i) sacc[i] *= SCALE;

    uint32_t p[32];
    float a0, a1;
    softmax_block(sacc, p, m0, m1, l0, l1, a0, a1, lane);
    rescale_o(o, a0, a1);
    pv_block(o, p, vt);
  }

  const float r0 = (l0 > 0.f) ? 1.f / l0 : 0.f;
  const float r1 = (l1 > 0.f) ? 1.f / l1 : 0.f;
  bf16* obase = out + ((int64_t)(batch * nq + head) * seq + qb * BLOCK_Q) * HEAD_DIM;
#pragma unroll
  for (int u = 0; u < 32; ++u) {
    const int t = u >> 1;
    const int row = wg * WG_ROWS + wg_row + g + ((u & 1) ? 8 : 0);
    bf16* dst = obase + row * HEAD_DIM + t * 8 + c2;
    *reinterpret_cast<uint32_t*>(dst) = pack2(o[4 * t + 2 * (u & 1) + 0] * ((u & 1) ? r1 : r0),
                                              o[4 * t + 2 * (u & 1) + 1] * ((u & 1) ? r1 : r0));
  }
#endif  // BSA_WGMMA
}

void check(const torch::Tensor& t, torch::ScalarType dtype, int ndim) {
  TORCH_CHECK_VALUE(t.is_cuda(), "Expected CUDA tensor");
  TORCH_CHECK_TYPE(t.scalar_type() == dtype, "Unexpected dtype");
  TORCH_CHECK_VALUE(t.dim() == ndim, "Unexpected rank");
  TORCH_CHECK_VALUE(t.is_contiguous(), "Expected contiguous tensor");
}

constexpr int kSmemBytes = (V_OFF(STAGES - 1) + TILE) * (int)sizeof(bf16) + 4 * (int)sizeof(uint64_t);

void kernel(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
            const torch::Tensor& block_ids, const torch::Tensor& block_counts,
            torch::Tensor& out) {
  check(q, torch::kBFloat16, 4);
  check(k, torch::kBFloat16, 4);
  check(v, torch::kBFloat16, 4);
  check(block_ids, torch::kInt32, 4);
  check(block_counts, torch::kInt32, 3);
  check(out, torch::kBFloat16, 4);

  const int batch = q.size(0);
  const int nq = q.size(1);
  const int seq = q.size(2);
  const int dim = q.size(3);
  TORCH_CHECK_VALUE(dim == HEAD_DIM, "Head dim must be 128");
  TORCH_CHECK_VALUE(seq % BLOCK_Q == 0, "Sequence must be a multiple of 128");
  TORCH_CHECK_VALUE(k.size(1) > 0 && nq % k.size(1) == 0, "Head count must divide");
  TORCH_CHECK_VALUE(block_ids.size(3) == SLOTS, "block_ids must have 16 slots");
  TORCH_CHECK_VALUE(block_ids.size(2) == seq / BLOCK_Q, "block_ids block count mismatch");
  TORCH_CHECK_VALUE(q.device() == out.device() && k.device() == q.device(), "Device mismatch");

  const int nblocks = seq / BLOCK_Q;
  if (batch == 0 || nblocks == 0) return;

  c10::cuda::CUDAGuard guard(q.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
  static bool smem_ready = false;
  if (!smem_ready) {
    const cudaError_t attr =
        cudaFuncSetAttribute(bsa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
    TORCH_CHECK(attr == cudaSuccess, cudaGetErrorString(attr));
    smem_ready = true;
  }

  // Row-major [B*HKV*S, 128] view: an 8x128x16 box starting at the block row is
  // exactly the unswizzled WGMMA panel layout, so one TMA move stages a tile.
  using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                                const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                                const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  static EncodeFn encode = nullptr;
  if (encode == nullptr) {
    cudaDriverEntryPointQueryResult status;
    const cudaError_t entry = cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled", reinterpret_cast<void**>(&encode), 12000, cudaEnableDefault,
        &status);
    TORCH_CHECK(entry == cudaSuccess && encode != nullptr, "cuTensorMapEncodeTiled unavailable");
  }
  const cuuint64_t rows = (cuuint64_t)batch * k.size(1) * seq;
  const cuuint64_t dims[3] = {HEAD_DIM / CHUNK, rows, HEAD_DIM / 8};
  const cuuint64_t strides[2] = {HEAD_DIM * sizeof(bf16), 8 * sizeof(bf16)};
  const cuuint32_t box[3] = {HEAD_DIM / CHUNK, BLOCK_KV, HEAD_DIM / 8};
  const cuuint32_t estride[3] = {1, 1, 1};
  CUtensorMap kmap, vmap;
  for (CUtensorMap* map : {&kmap, &vmap}) {
    const void* base = (map == &kmap) ? k.data_ptr() : v.data_ptr();
    const CUresult status =
        encode(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(base), dims, strides,
               box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
               CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(status == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)status);
  }

  const dim3 grid(nq, nblocks, batch);
  bsa_kernel<<<grid, THREADS, kSmemBytes, stream>>>(
      kmap, vmap, reinterpret_cast<const bf16*>(q.data_ptr()), block_ids.data_ptr<int>(),
      block_counts.data_ptr<int>(), reinterpret_cast<bf16*>(out.data_ptr()), nq, k.size(1), seq,
      nblocks);
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("kernel", &kernel); }

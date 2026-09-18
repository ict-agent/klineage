// Sparse MLA prefill (DeepSeek-V3.2 supplied-index) for sm_90a.
//
// Grid = 2 * tokens: blockIdx.x = 2 * token + value_half, 256 threads = two
// warpgroups.  Warpgroup h owns heads 64h..64h+63 over the full selected key
// set; value_half selects the 256 latent columns this CTA emits.
//
//   Q  [128, 576] bf16 : 9 dim tiles of 16 row slabs x 8 k-cores
//   KV [16, 576]  bf16 : 72 dim cores, each 2 key cores (+ 16 B pad)
//   P  [128, 16]  bf16 : 2 head halves of 8 row slabs x 2 k-cores
//
//   QK : wgmma.m64n16k16 x 36  (9 dims x 4 k-steps), key tile = 16 keys
//   PV : wgmma.m64n256k16 x 1  (k = 16 keys, n = 256 latent columns)
//
// A token's four head-half x value-half quadrants cannot share one CTA: their
// FP32 accumulators alone need the whole register file.  Two CTAs therefore
// split the value dimension and both walk the same key list.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cstdint>

namespace {

using bf16 = __nv_bfloat16;

constexpr int kThreads = 256;
constexpr int kHeads = 128;
constexpr int kQkDim = 576;
constexpr int kValDim = 512;
constexpr int kValHalf = kValDim / 2;
constexpr int kTopk = 2048;
constexpr float kScale = 0.1352337788608801f;  // 192^-0.5 * (1 + 0.1 ln 40)^2
constexpr float kScaleLog2e = kScale * 1.4426950408889634f;

constexpr int kHeadsWg = 64;
constexpr int kKeys = 16;                 // keys staged per pipeline stage
constexpr int kTiles = kTopk / kKeys;     // 128
constexpr int kBuffs = 3;

// Sparse row offsets, pre-encoded per token in shared memory: bits [0:30] hold
// the KV row byte offset, bit 31 marks an out-of-range (masked) index.  The
// gather then reads no global memory, so no DRAM latency sits on the tile's
// critical path.
constexpr unsigned kInvalidBit = 0x80000000u;
constexpr unsigned kOffMask = 0x7fffffffu;

constexpr int kDimTiles = kQkDim / 64;    // 9
constexpr int kCoreB = 128;               // 8x8 bf16 core matrix
constexpr int kQMBlockB = 1024;           // 8 k-cores = one 8-row Q slab
constexpr int kQDimTileB = 16 * kQMBlockB;
constexpr int kQSlabs = 16;

constexpr int kKvDimCores = kQkDim / 8;              // 72
// KV layout: dim cores are the innermost run, so a warp's 16-byte gather chunks
// land in one contiguous shared-memory span.  The old dim-core-major order put
// every chunk in its own 128-byte row, which made each LDGSTS write 32 shared
// wavefronts instead of 4.
constexpr int kKvKeyCoreB = kKvDimCores * kCoreB;    // one 8-key core, all dims
constexpr int kKvBufB = 2 * kKvKeyCoreB;             // 16 keys x 576 dims

constexpr int kPMBlockB = 2 * kCoreB;                // 2 key cores
constexpr int kPB = kHeads * kKeys * 2;

constexpr uint32_t kCoreU = kCoreB / 16;             // 8
constexpr uint32_t kQSlabU = kQMBlockB / 16;         // 64
constexpr uint32_t kKvDimCoreU = kCoreB / 16;        // 8, K step of the QK B operand
constexpr uint32_t kKvKeyCoreU = kKvKeyCoreB / 16;   // 576, key-core step
constexpr uint32_t kPSlabU = kPMBlockB / 16;         // 16
constexpr uint32_t kKStepB = 2 * kCoreB;             // 16 dims = 2 k-cores

// 16-byte-unit variants used to build wgmma descriptors without any shifts.
constexpr uint32_t kQDimTileU = kQDimTileB / 16;
constexpr uint32_t kQSlabOffU = (8 * kQMBlockB) / 16;
constexpr uint32_t kPSlabOffU = (8 * kPMBlockB) / 16;
constexpr uint32_t kKStepU = kKStepB / 16;
constexpr uint32_t kKvBufU = kKvBufB / 16;
constexpr uint32_t kValHalfU = (32 * kCoreB) / 16;   // 32 dim cores = 256 values

struct Shared {
  alignas(128) unsigned char q[kDimTiles * kQDimTileB];
  alignas(128) unsigned char kv[kBuffs][kKvBufB];
  alignas(128) unsigned char p[kPB];
  alignas(128) int idx[kTopk];
  alignas(16) unsigned char allvalid[kBuffs];
};

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// wgmma shared-memory descriptor: [0:13] address/16, [16:29] leading-byte
// offset/16, [32:45] stride-byte offset/16.  Every shared address in this
// kernel is below 0x4000 once shifted, so `lo` needs no masking and a
// descriptor costs one integer add per tile step instead of four.
__device__ __forceinline__ uint64_t desc(uint32_t lo, uint32_t hi) {
  return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
}

// exp2 through the MUFU.EX2 approximation.  Arguments are non-positive here,
// so overflow is impossible and the denormal path of exp2f is irrelevant.
__device__ __forceinline__ float exp2_approx(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}

template <int Pending>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(Pending) : "memory");
}

__device__ __forceinline__ void async_fence() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// D[64,16] += A[64,k] * B[16,k]^T; both operands live in shared memory.
__device__ __forceinline__ void qk_mma(float (&d)[8], uint64_t a, uint64_t b, bool accum) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %10, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, 1, 1, 0, 0;\n"
      "}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "l"(a), "l"(b), "r"(static_cast<int>(accum)));
}

// D[64,256] += P[64,16] * V[16,256]; V is read key-major (transposed B).
__device__ __forceinline__ void pv_mma(float (&d)[128], uint64_t a, uint64_t b) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,"
      "%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,"
      "%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,"
      "%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,"
      "%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, "
      "%128, %129, 1, 1, 1, 0, 1;\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]),
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]),
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]),
        "+f"(d[64]), "+f"(d[65]), "+f"(d[66]), "+f"(d[67]), "+f"(d[68]), "+f"(d[69]), "+f"(d[70]), "+f"(d[71]),
        "+f"(d[72]), "+f"(d[73]), "+f"(d[74]), "+f"(d[75]), "+f"(d[76]), "+f"(d[77]), "+f"(d[78]), "+f"(d[79]),
        "+f"(d[80]), "+f"(d[81]), "+f"(d[82]), "+f"(d[83]), "+f"(d[84]), "+f"(d[85]), "+f"(d[86]), "+f"(d[87]),
        "+f"(d[88]), "+f"(d[89]), "+f"(d[90]), "+f"(d[91]), "+f"(d[92]), "+f"(d[93]), "+f"(d[94]), "+f"(d[95]),
        "+f"(d[96]), "+f"(d[97]), "+f"(d[98]), "+f"(d[99]), "+f"(d[100]), "+f"(d[101]), "+f"(d[102]), "+f"(d[103]),
        "+f"(d[104]), "+f"(d[105]), "+f"(d[106]), "+f"(d[107]), "+f"(d[108]), "+f"(d[109]), "+f"(d[110]), "+f"(d[111]),
        "+f"(d[112]), "+f"(d[113]), "+f"(d[114]), "+f"(d[115]), "+f"(d[116]), "+f"(d[117]), "+f"(d[118]), "+f"(d[119]),
        "+f"(d[120]), "+f"(d[121]), "+f"(d[122]), "+f"(d[123]), "+f"(d[124]), "+f"(d[125]), "+f"(d[126]), "+f"(d[127])
      : "l"(a), "l"(b));
}

__device__ __forceinline__ unsigned int pack_bf16(float x, float y) {
  const __nv_bfloat162 h = __floats2bfloat162_rn(x, y);
  return *reinterpret_cast<const unsigned int*>(&h);
}

// Stage one 16-key tile of KV rows into the blocked layout: key r, dim core c
// lands at (r/8)*kKvKeyCoreB + c*kCoreB + (r%8)*16.
//
// Each lane owns one 16-byte chunk and moves it with one vectorized load and
// one vectorized store.  Chunks are ordered dim-core first, so the lanes of a
// warp read a contiguous span that the L1 coalesces into whole 128-byte lines:
// the fewest possible in-flight L2 read requests per tile.
__device__ __forceinline__ void gather_tile(Shared& sm, const bf16* __restrict__ kv,
                                            const int* __restrict__ enc, int buf, int tid) {
  constexpr int kChunks = kKeys * kKvDimCores;
  constexpr int kIters = (kChunks + kThreads - 1) / kThreads;
  constexpr int kDcStep = kThreads / kKeys;  // dim cores covered per iteration

  // Lanes 0..15 carry key 0..15, so one warp ballot covers the whole tile.
  const unsigned bits = __ballot_sync(0xffffffffu, enc[tid & (kKeys - 1)] >= 0);
  if (tid == 0) sm.allvalid[buf] = ((bits & 0xffffu) == 0xffffu);

  // A warp's first sixteen lanes take every key of one dim core and its last
  // sixteen every key of the next, so one store fills four whole 128-byte
  // shared rows instead of touching 32 of them.
  const int r = tid & (kKeys - 1);
  const int off = enc[r] & static_cast<int>(kOffMask);
  unsigned char* dst =
      sm.kv[buf] + (r >> 3) * kKvKeyCoreB + (r & 7) * 16 + (tid >> 4) * kCoreB;
  const char* src = reinterpret_cast<const char*>(kv) + off + (tid >> 4) * 16;
  #pragma unroll
  for (int j = 0; j < kIters; ++j) {
    if (tid + j * kThreads >= kChunks) break;
    const uint4 v = *reinterpret_cast<const uint4*>(src + j * (kDcStep * 16));
    *reinterpret_cast<uint4*>(dst + j * (kDcStep * kCoreB)) = v;
  }
  async_fence();
}

__global__ void __launch_bounds__(kThreads, 1)
sparse_mla_kernel(const bf16* __restrict__ q, const bf16* __restrict__ kv,
                  const int* __restrict__ indices, bf16* __restrict__ out,
                  float* __restrict__ maxlog, float* __restrict__ lse, int tokens) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  Shared& sm = *reinterpret_cast<Shared*>(smem_raw);

  const int tok = blockIdx.x >> 1;
  const int vhalf = blockIdx.x & 1;
  const int tid = threadIdx.x;
  const int wg = tid >> 7;
  const int tw = tid & 127;
  const int warp = tw >> 5;
  const int lane = tw & 31;
  const int l4 = lane & 3;
  const int brow = warp * 16 + (lane >> 2);

  // ---- encode this token's sparse row offsets once, in shared memory ----
  {
    const int* src = indices + static_cast<size_t>(tok) * kTopk;
    #pragma unroll
    for (int j = 0; j < kTopk / kThreads; ++j) {
      const int i = tid + j * kThreads;
      const int raw = src[i];
      const bool ok = (raw >= 0) && (raw < tokens);
      const int row = ok ? raw : 0;
      const unsigned foff = static_cast<unsigned>(row) * (kQkDim * sizeof(bf16));
      sm.idx[i] = static_cast<int>(ok ? foff : (foff | kInvalidBit));
    }
  }

  // ---- Q: 128 heads x 576, 8 warps cover slabs g and g+8 of every dim tile ----
  {
    const bf16* qrow = q + static_cast<size_t>(tok) * kHeads * kQkDim;
    const int g = tid >> 5;
    const int a = (tid & 31) >> 3;
    const int b = tid & 7;
    #pragma unroll
    for (int s = 0; s < 2 * kDimTiles; ++s) {
      const int half = s & 1;
      const int t = s >> 1;
      const int dst = t * kQDimTileB + (g + half * 8) * kQMBlockB + a * 256 + b * 16;
      const bf16* src = qrow + static_cast<size_t>(half * 64 + g * 8 + b) * kQkDim +
                        (t * 4 + a) * 16;
      __pipeline_memcpy_async(sm.q + dst, src, 16);
      __pipeline_memcpy_async(sm.q + dst + kCoreB, src + 8, 16);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
  }

  // ---- KV pipeline: kBuffs-1 tiles in flight, gather issued kBuffs-1 ahead ----
  #pragma unroll
  for (int u = 0; u < kBuffs - 1; ++u) {
    gather_tile(sm, kv, sm.idx + u * kKeys, u, tid);
  }

  float o[128];
  #pragma unroll
  for (int i = 0; i < 128; ++i) o[i] = 0.0f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

  for (int t = 0; t < kTiles; ++t) {
    const int buf = t % kBuffs;
    __syncthreads();

    // ---- QK: 9 dim tiles x 4 k-steps ----
    float s[8];
    {
      const uint32_t qlo = (smem_addr(sm.q) >> 4) + wg * kQSlabOffU + (kCoreU << 16);
      const uint32_t klo = (smem_addr(sm.kv[buf]) >> 4) + (kKvDimCoreU << 16);
      wgmma_fence();
      #pragma unroll
      for (int dt = 0; dt < kDimTiles; ++dt) {
        #pragma unroll
        for (int st = 0; st < 4; ++st) {
          const uint64_t da = desc(qlo + dt * kQDimTileU + st * kKStepU, kQSlabU);
          const uint64_t db = desc(klo + (dt * 8 + st * 2) * kKvDimCoreU, kKvKeyCoreU);
          qk_mma(s, da, db, (dt | st) != 0);
        }
      }
      wgmma_commit();

      // The next tile's gather overlaps this QK's tensor work: the loads only
      // have to land before the warp reaches the first dependent store.
      const int u = t + kBuffs - 1;
      if (u < kTiles) gather_tile(sm, kv, sm.idx + u * kKeys, u % kBuffs, tid);

      wgmma_wait<0>();
    }

    // ---- online softmax over the 16 staged keys ----
    if (sm.allvalid[buf] == 0) {
      const int* tile_enc = sm.idx + t * kKeys;
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int col = 8 * (i >> 2) + 2 * l4 + (i & 1);
        if (tile_enc[col] < 0) s[i] = -INFINITY;
      }
    }
    float tm0 = fmaxf(fmaxf(s[0], s[1]), fmaxf(s[4], s[5]));
    float tm1 = fmaxf(fmaxf(s[2], s[3]), fmaxf(s[6], s[7]));
    tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 1));
    tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 2));
    tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 1));
    tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 2));

    const float nm0 = fmaxf(m0, tm0);
    const float nm1 = fmaxf(m1, tm1);
    const float c0 = (nm0 > m0) ? exp2_approx((m0 - nm0) * kScaleLog2e) : 1.0f;
    const float c1 = (nm1 > m1) ? exp2_approx((m1 - nm1) * kScaleLog2e) : 1.0f;
    m0 = nm0;
    m1 = nm1;

    float p[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float mm = ((i & 2) == 0) ? nm0 : nm1;
      p[i] = exp2_approx((s[i] - mm) * kScaleLog2e);
    }
    float rs0 = (p[0] + p[1]) + (p[4] + p[5]);
    float rs1 = (p[2] + p[3]) + (p[6] + p[7]);
    rs0 += __shfl_xor_sync(0xffffffffu, rs0, 1);
    rs0 += __shfl_xor_sync(0xffffffffu, rs0, 2);
    rs1 += __shfl_xor_sync(0xffffffffu, rs1, 1);
    rs1 += __shfl_xor_sync(0xffffffffu, rs1, 2);
    l0 = l0 * c0 + rs0;
    l1 = l1 * c1 + rs1;

    if (c0 != 1.0f || c1 != 1.0f) {
      #pragma unroll
      for (int i = 0; i < 128; ++i) o[i] *= ((i & 2) == 0) ? c0 : c1;
    }

    #pragma unroll
    for (int i = 0; i < 8; i += 2) {
      const int row = brow + 8 * ((i & 2) >> 1);
      const int col = 8 * (i >> 2) + 2 * l4;
      *reinterpret_cast<unsigned int*>(sm.p + wg * (8 * kPMBlockB) + (row >> 3) * kPMBlockB +
                                       (col >> 3) * kCoreB + (row & 7) * 16 + (col & 7) * 2) =
          pack_bf16(p[i], p[i + 1]);
    }
    async_fence();

    // ---- PV: 16 keys x 256 latent columns ----
    {
      const uint32_t plo = (smem_addr(sm.p) >> 4) + wg * kPSlabOffU + (kCoreU << 16);
      const uint32_t vlo =
          (smem_addr(sm.kv[buf]) >> 4) + vhalf * kValHalfU + (kKvKeyCoreU << 16);
      wgmma_fence();
      pv_mma(o, desc(plo, kPSlabU), desc(vlo, kKvDimCoreU));
      wgmma_commit();
      wgmma_wait<0>();
    }
  }

  // ---- finalize: row-normalize and emit this CTA's value half ----
  const float inv0 = (l0 > 0.0f) ? 1.0f / l0 : 0.0f;
  const float inv1 = (l1 > 0.0f) ? 1.0f / l1 : 0.0f;
  bf16* orow = out + static_cast<size_t>(tok) * kHeads * kValDim + wg * kHeadsWg * kValDim;
  #pragma unroll
  for (int i = 0; i < 128; i += 2) {
    const float iv = ((i & 2) == 0) ? inv0 : inv1;
    const int row = brow + 8 * ((i & 2) >> 1);
    const int col = vhalf * kValHalf + 8 * (i >> 2) + 2 * l4;
    *reinterpret_cast<unsigned int*>(orow + static_cast<size_t>(row) * kValDim + col) =
        pack_bf16(o[i] * iv, o[i + 1] * iv);
  }
  if (l4 == 0 && vhalf == 0) {
    float* ml = maxlog + static_cast<size_t>(tok) * kHeads + wg * kHeadsWg + brow;
    float* ls = lse + static_cast<size_t>(tok) * kHeads + wg * kHeadsWg + brow;
    ml[0] = m0 * kScale;
    ml[8] = m1 * kScale;
    ls[0] = logf(l0) + m0 * kScale;
    ls[8] = logf(l1) + m1 * kScale;
  }
}

void sparse_mla(torch::Tensor q, torch::Tensor kv, torch::Tensor indices,
                torch::Tensor output, torch::Tensor max_logits, torch::Tensor lse) {
  TORCH_CHECK(q.is_cuda() && kv.is_cuda() && indices.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && kv.scalar_type() == torch::kBFloat16,
              "q/kv must be bfloat16");
  TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must be int32");
  TORCH_CHECK(q.dim() == 3 && q.size(1) == kHeads && q.size(2) == kQkDim, "unexpected q shape");
  TORCH_CHECK(kv.dim() == 3 && kv.size(1) == 1 && kv.size(2) == kQkDim, "unexpected kv shape");
  TORCH_CHECK(indices.dim() == 3 && indices.size(1) == 1 && indices.size(2) == kTopk,
              "unexpected indices shape");
  TORCH_CHECK(output.dim() == 3 && output.size(1) == kHeads && output.size(2) == kValDim,
              "unexpected output shape");
  TORCH_CHECK(max_logits.scalar_type() == torch::kFloat32 && lse.scalar_type() == torch::kFloat32,
              "statistics must be float32");
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16, "output must be bfloat16");

  const int tokens = static_cast<int>(q.size(0));
  if (tokens == 0) return;

  c10::cuda::CUDAGuard guard(q.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(sparse_mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(sizeof(Shared)));
    configured = true;
  }
  sparse_mla_kernel<<<2 * tokens, kThreads, sizeof(Shared), stream>>>(
      reinterpret_cast<const bf16*>(q.data_ptr()), reinterpret_cast<const bf16*>(kv.data_ptr()),
      indices.data_ptr<int>(), reinterpret_cast<bf16*>(output.data_ptr()),
      max_logits.data_ptr<float>(), lse.data_ptr<float>(), tokens);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("kernel", &sparse_mla);
}

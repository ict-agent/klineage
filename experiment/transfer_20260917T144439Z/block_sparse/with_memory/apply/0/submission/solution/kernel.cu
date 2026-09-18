// Causal block-sparse attention, BF16 in/out with FP32 accumulation.
//
// One CTA per (batch, query head, query block) gathers the selected key blocks
// (<= 16 blocks of 128 keys) into 64-key tiles and runs an online softmax.
// 256 threads form two warpgroups, each owning 64 query rows:
//
//   wg 0: rows   0..63   +---------------------------+
//   wg 1: rows  64..127  | S(64x64)  = Q K^T    8x   | wgmma m64n64k16
//                        | softmax(S) -> P bf16      |
//                        | O(64x128) += P V     4x   | wgmma m64n128k16
//                        +---------------------------+
//
// WGMMA takes its B operand from shared memory as 8x8 core matrices whose 16 B
// always span the contraction axis. K contracts over the feature dim and is
// staged key-major, while V contracts over the key index and is therefore
// staged transposed by a one-off prepass over the whole cache:
//
//   K (k = dim):  il (key,dim) = (key>>3)*1024 + (dim>>3)*64 + (key&7)*8 + (dim&7)
//   V (k = key):  ilt(key,dim) = (key>>3)*1024 + (dim>>3)*64 + (dim&7)*8 + (key&7)
//
// LBO/SBO are the core-matrix strides along k and n, so the two operands need
// swapped descriptor fields. A stays in registers (RS), so Q is loaded once per
// CTA and P never leaves the registers. Blocks past the diagonal are dropped:
// every score of such a block is masked, so it cannot contribute.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

using bf16 = __nv_bfloat16;

// Problem constants (definition: HQ=32, HKV=8, D=128, block 128, capacity 16).
constexpr int kHeads = 32;
constexpr int kKVHeads = 8;
constexpr int kDim = 128;
constexpr int kCapacity = 16;

// Tiling.
constexpr int kRows = 128;                       // query rows per CTA
constexpr int kKeys = 64;                        // keys per pipeline stage
constexpr int kTilesPerBlock = kRows / kKeys;
constexpr int kMaxTiles = kCapacity * kTilesPerBlock;
constexpr int kThreads = 256;                    // two 128-thread warpgroups
constexpr int kStages = 3;
constexpr int kTileElems = kKeys * kDim;
constexpr int kChunks = kTileElems / 8;          // 16 B cp.async chunks per tile
constexpr int kChunksPerThread = kChunks / kThreads;

// Transposed V plane: 8 keys x 128 dims per core-matrix row block.
constexpr int kVBlockKeys = 8;
constexpr int kVBlockElems = kDim * kVBlockKeys;
constexpr int kKeyChunks = kDim;                 // 16 B chunks per key block

constexpr uint32_t kLboK = 128, kSboK = 2048;
constexpr uint32_t kLboV = 2048, kSboV = 128;
constexpr int kKStepAdv = 128;                   // halves per 16-feature k-step
constexpr int kVStepAdv = 2048;                  // halves per 16-key k-step
constexpr float kScaleLog2e = 1.4426950408889634f / 11.313708498984761f;

constexpr int kThreadsTranspose = 256;

// Staging buffers first (16 B aligned for wgmma descriptors), then tile tables.
struct alignas(16) Shared {
  bf16 ks[kStages][kTileElems];
  bf16 vs[kStages][kTileElems];
  int ntiles;
  int tile_ksrc[kMaxTiles];  // element offset of the key tile inside a K plane
  int tile_vsrc[kMaxTiles];  // half offset of the key tile inside a transposed plane
  int tile_key[kMaxTiles];   // tile offset inside its block (causal mask coordinate)
  int tile_diag[kMaxTiles];  // 1 when the causal mask applies
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Matrix descriptor: base address, leading and stride byte offsets of the core
// matrices (both given in bytes, encoded in 16 B units).
__device__ __forceinline__ uint64_t gdesc(const void* p, uint32_t lbo, uint32_t sbo) {
  uint64_t d = (uint64_t)(((uint64_t)smem_u32(p) >> 4) & 0x3FFF);
  d |= ((uint64_t)(lbo >> 4) & 0x3FFF) << 16;
  d |= ((uint64_t)(sbo >> 4) & 0x3FFF) << 32;
  return d;
}

__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int kPending>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(kPending) : "memory");
}
// Publish generic-proxy shared writes (cp.async) to the async proxy wgmma reads.
__device__ __forceinline__ void async_fence() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

#define WGMMA_M64N64(S, A, DESC, EN)                                              \
  asm volatile(                                                                   \
      "{\n.reg .pred p;\nsetp.ne.b32 p, %32, 0;\n"                                \
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "                     \
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,"   \
      "%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"                         \
      "{%33,%34,%35,%36}, %37, p, 1, 1, 0;\n}\n"                                  \
      : "+f"(S[0]), "+f"(S[1]), "+f"(S[2]), "+f"(S[3]), "+f"(S[4]), "+f"(S[5]),   \
        "+f"(S[6]), "+f"(S[7]), "+f"(S[8]), "+f"(S[9]), "+f"(S[10]), "+f"(S[11]), \
        "+f"(S[12]), "+f"(S[13]), "+f"(S[14]), "+f"(S[15]), "+f"(S[16]),          \
        "+f"(S[17]), "+f"(S[18]), "+f"(S[19]), "+f"(S[20]), "+f"(S[21]),          \
        "+f"(S[22]), "+f"(S[23]), "+f"(S[24]), "+f"(S[25]), "+f"(S[26]),          \
        "+f"(S[27]), "+f"(S[28]), "+f"(S[29]), "+f"(S[30]), "+f"(S[31])           \
      : "r"(EN), "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "l"(DESC))

#define WGMMA_M64N128(D, A, DESC, EN)                                             \
  asm volatile(                                                                   \
      "{\n.reg .pred p;\nsetp.ne.b32 p, %64, 0;\n"                                \
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "                    \
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,"   \
      "%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,"  \
      "%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,"  \
      "%56,%57,%58,%59,%60,%61,%62,%63},"                                         \
      "{%65,%66,%67,%68}, %69, p, 1, 1, 0;\n}\n"                                 \
      : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3]), "+f"(D[4]), "+f"(D[5]),   \
        "+f"(D[6]), "+f"(D[7]), "+f"(D[8]), "+f"(D[9]), "+f"(D[10]), "+f"(D[11]), \
        "+f"(D[12]), "+f"(D[13]), "+f"(D[14]), "+f"(D[15]), "+f"(D[16]),          \
        "+f"(D[17]), "+f"(D[18]), "+f"(D[19]), "+f"(D[20]), "+f"(D[21]),          \
        "+f"(D[22]), "+f"(D[23]), "+f"(D[24]), "+f"(D[25]), "+f"(D[26]),          \
        "+f"(D[27]), "+f"(D[28]), "+f"(D[29]), "+f"(D[30]), "+f"(D[31]),          \
        "+f"(D[32]), "+f"(D[33]), "+f"(D[34]), "+f"(D[35]), "+f"(D[36]),          \
        "+f"(D[37]), "+f"(D[38]), "+f"(D[39]), "+f"(D[40]), "+f"(D[41]),          \
        "+f"(D[42]), "+f"(D[43]), "+f"(D[44]), "+f"(D[45]), "+f"(D[46]),          \
        "+f"(D[47]), "+f"(D[48]), "+f"(D[49]), "+f"(D[50]), "+f"(D[51]),          \
        "+f"(D[52]), "+f"(D[53]), "+f"(D[54]), "+f"(D[55]), "+f"(D[56]),          \
        "+f"(D[57]), "+f"(D[58]), "+f"(D[59]), "+f"(D[60]), "+f"(D[61]),          \
        "+f"(D[62]), "+f"(D[63])                                                  \
      : "r"(EN), "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "l"(DESC))

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }

template <int kPending>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(kPending));
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  const __nv_bfloat162 pair = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&pair);
}

// Repack V into the transposed core-matrix layout. One thread produces one 16 B
// run: 8 consecutive keys of a single feature dim, gathered from 8 rows of the
// key-major source plane. Covers HKV x (seq/8) key blocks x 128 dims.
__global__ void transpose_v_kernel(const bf16* __restrict__ v, bf16* __restrict__ vt,
                                   int64_t chunks, int seq, int keyblocks) {
  const int64_t chunk = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (chunk >= chunks) return;

  const int c = (int)(chunk % kKeyChunks);
  const int kb = (int)((chunk / kKeyChunks) % keyblocks);
  const int64_t plane = chunk / ((int64_t)kKeyChunks * keyblocks);
  const int db = c >> 3, dr = c & 7;

  const bf16* src = v + (plane * seq + (size_t)kb * kVBlockKeys) * kDim + db * 8 + dr;
  bf16* dst = vt + (size_t)(plane * keyblocks + kb) * kVBlockElems + db * 64 + dr * 8;

  uint32_t reg[4];
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const __nv_bfloat162 pair = __halves2bfloat162(src[(2 * j) * kDim], src[(2 * j + 1) * kDim]);
    reg[j] = *reinterpret_cast<const uint32_t*>(&pair);
  }
  *reinterpret_cast<uint4*>(dst) = make_uint4(reg[0], reg[1], reg[2], reg[3]);
}

__global__ void __launch_bounds__(kThreads, 1)
block_sparse_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k,
                    const bf16* __restrict__ vt, const int* __restrict__ block_ids,
                    const int* __restrict__ block_counts, bf16* __restrict__ out, int seq,
                    int nblocks) {
  extern __shared__ unsigned char smem_raw[];
  Shared& sm = *reinterpret_cast<Shared*>(smem_raw);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wg = warp >> 2;           // warpgroup 0 or 1
  const int wgw = warp & 3;           // warp inside the warpgroup
  const int gid = lane >> 2;
  const int tig = lane & 3;
  const int wg_row = wg * kKeys;      // first of the 64 rows owned by this warpgroup
  const int row_lo = wg_row + wgw * 16 + gid;

  const int qb = blockIdx.x;
  const int hq = blockIdx.y;
  const int b = blockIdx.z;
  const int hkv = hq / (kHeads / kKVHeads);

  // Q is consumed as a register A-fragment; lane (gid,tig) holds the
  // (row, 2tig..2tig+1) and (row+8, 2tig..2tig+1) pairs of each 16-wide k-step.
  uint32_t aq[8][4];
  {
    const bf16* qrow =
        q + (((size_t)b * kHeads + hq) * seq + (size_t)qb * kRows + row_lo) * kDim + 2 * tig;
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      aq[ks][0] = *reinterpret_cast<const uint32_t*>(qrow + ks * 16);
      aq[ks][1] = *reinterpret_cast<const uint32_t*>(qrow + 8 * kDim + ks * 16);
      aq[ks][2] = *reinterpret_cast<const uint32_t*>(qrow + ks * 16 + 8);
      aq[ks][3] = *reinterpret_cast<const uint32_t*>(qrow + 8 * kDim + ks * 16 + 8);
    }
  }

  // Flatten the selected blocks into 64-key tiles; skip blocks entirely past
  // the diagonal, whose every score is masked.
  if (tid == 0) {
    const size_t head = ((size_t)b * kHeads + hq) * nblocks + qb;
    const int count = block_counts[head];
    const int* ids = block_ids + head * kCapacity;
    int n = 0;
    for (int j = 0; j < count; ++j) {
      const int bid = ids[j];
      if (bid < 0 || bid > qb) continue;
      #pragma unroll
      for (int t = 0; t < kTilesPerBlock; ++t) {
        const int key0 = bid * kRows + t * kKeys;
        sm.tile_ksrc[n] = key0 * kDim;
        sm.tile_vsrc[n] = (key0 / kVBlockKeys) * kVBlockElems;
        sm.tile_key[n] = t * kKeys;
        sm.tile_diag[n] = (bid == qb) ? 1 : 0;
        ++n;
      }
    }
    sm.ntiles = n;
  }

  const bf16* kbase = k + (((size_t)b * kKVHeads + hkv) * seq) * kDim;
  const bf16* vbase = vt + ((size_t)b * kKVHeads + hkv) * (seq / kVBlockKeys) * kVBlockElems;

  // K is staged into its core-matrix layout; the transposed V plane is a plain
  // contiguous copy at 16 B granularity.
  auto prefetch = [&](int ti, int stage) {
    const bf16* kp = kbase + sm.tile_ksrc[ti];
    const bf16* vp = vbase + sm.tile_vsrc[ti];
    #pragma unroll
    for (int i = 0; i < kChunksPerThread; ++i) {
      const int e = tid + i * kThreads;
      const int key = e >> 4, dim8 = e & 15;
      cp_async16(&sm.ks[stage][((key >> 3) * 16 + dim8) * 64 + (key & 7) * 8],
                 kp + (size_t)key * kDim + dim8 * 8);
      cp_async16(&sm.vs[stage][e * 8], vp + (size_t)e * 8);
    }
    cp_commit();
  };

  float acc[64];
  #pragma unroll
  for (int i = 0; i < 64; ++i) acc[i] = 0.f;
  float rmax[2] = {-INFINITY, -INFINITY};
  float rsum[2] = {0.f, 0.f};

  __syncthreads();
  const int ntiles = sm.ntiles;
  if (ntiles > 0) prefetch(0, 0);
  if (ntiles > 1) prefetch(1, 1 % kStages);

  for (int ti = 0; ti < ntiles; ++ti) {
    if (ti + 1 < ntiles) cp_wait<1>();
    else cp_wait<0>();
    __syncthreads();
    async_fence();
    if (ti + kStages - 1 < ntiles) prefetch(ti + kStages - 1, (ti + kStages - 1) % kStages);

    // A diagonal tile entirely past this warpgroup's last row holds no allowed
    // score. The guard must be uniform across the warpgroup: wgmma is issued by
    // all 128 threads together.
    if (sm.tile_diag[ti] && sm.tile_key[ti] > wg_row + kKeys - 1) continue;

    const int stage = ti % kStages;
    const bf16* ktile = sm.ks[stage];
    const bf16* vtile = sm.vs[stage];

    // S = Q K^T over 64 keys: 8 k-steps of m64n64k16.
    float s[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) s[i] = 0.f;
    wgmma_fence();
    #pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const uint64_t dk = gdesc(ktile + ks * kKStepAdv, kLboK, kSboK);
      WGMMA_M64N64(s, aq[ks], dk, 1);
    }
    wgmma_commit();
    wgmma_wait<0>();

    // Base-2 scores; the diagonal tile masks everything past the row position.
    #pragma unroll
    for (int i = 0; i < 32; ++i) s[i] *= kScaleLog2e;
    if (sm.tile_diag[ti]) {
      const int key0 = sm.tile_key[ti];
      #pragma unroll
      for (int cg = 0; cg < 8; ++cg) {
        #pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
          const int row = row_lo + 8 * rr;
          const int c = key0 + 8 * cg + 2 * tig;
          s[4 * cg + 2 * rr + 0] = (c <= row) ? s[4 * cg + 2 * rr + 0] : -INFINITY;
          s[4 * cg + 2 * rr + 1] = (c + 1 <= row) ? s[4 * cg + 2 * rr + 1] : -INFINITY;
        }
      }
    }

    // Online softmax: four lanes cover the 64 key columns of one row.
    float mx0 = -INFINITY, mx1 = -INFINITY;
    #pragma unroll
    for (int cg = 0; cg < 8; ++cg) {
      mx0 = fmaxf(mx0, fmaxf(s[4 * cg + 0], s[4 * cg + 1]));
      mx1 = fmaxf(mx1, fmaxf(s[4 * cg + 2], s[4 * cg + 3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));

    const float m0 = fmaxf(rmax[0], mx0);
    const float m1 = fmaxf(rmax[1], mx1);
    const float a0 = ex2(rmax[0] - m0);
    const float a1 = ex2(rmax[1] - m1);
    rmax[0] = m0;
    rmax[1] = m1;

    // The PV accumulator spans 16 column groups of the 128 features, unlike the
    // 8-group score matrix, so every group has to be rescaled.
    #pragma unroll
    for (int cg = 0; cg < 16; ++cg) {
      acc[4 * cg + 0] *= a0;
      acc[4 * cg + 1] *= a0;
      acc[4 * cg + 2] *= a1;
      acc[4 * cg + 3] *= a1;
    }

    float sum0 = 0.f, sum1 = 0.f;
    #pragma unroll
    for (int cg = 0; cg < 8; ++cg) {
      s[4 * cg + 0] = ex2(s[4 * cg + 0] - m0);
      s[4 * cg + 1] = ex2(s[4 * cg + 1] - m0);
      s[4 * cg + 2] = ex2(s[4 * cg + 2] - m1);
      s[4 * cg + 3] = ex2(s[4 * cg + 3] - m1);
      sum0 += s[4 * cg + 0] + s[4 * cg + 1];
      sum1 += s[4 * cg + 2] + s[4 * cg + 3];
    }
    sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1);
    sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
    sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1);
    sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
    rsum[0] = rsum[0] * a0 + sum0;
    rsum[1] = rsum[1] * a1 + sum1;

    // P in the A-fragment layout of P V: the score slots of two adjacent column
    // groups pack straight into one register.
    uint32_t p[4][4];
    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      p[ks][0] = pack_bf16(s[8 * ks + 0], s[8 * ks + 1]);
      p[ks][1] = pack_bf16(s[8 * ks + 2], s[8 * ks + 3]);
      p[ks][2] = pack_bf16(s[8 * ks + 4], s[8 * ks + 5]);
      p[ks][3] = pack_bf16(s[8 * ks + 6], s[8 * ks + 7]);
    }

    // O += P V over the same 64 keys and all 128 features: 4 k-steps of m64n128k16.
    wgmma_fence();
    #pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const uint64_t dv = gdesc(vtile + ks * kVStepAdv, kLboV, kSboV);
      WGMMA_M64N128(acc, p[ks], dv, 1);
    }
    wgmma_commit();
    wgmma_wait<0>();
  }

  // Normalize and store rows row_lo and row_lo+8 of this thread.
  const float inv0 = rsum[0] > 0.f ? 1.f / rsum[0] : 0.f;
  const float inv1 = rsum[1] > 0.f ? 1.f / rsum[1] : 0.f;
  bf16* orow = out + (((size_t)b * kHeads + hq) * seq + (size_t)qb * kRows + row_lo) * kDim + 2 * tig;
  #pragma unroll
  for (int cg = 0; cg < 16; ++cg) {
    *reinterpret_cast<uint32_t*>(orow + cg * 8) = pack_bf16(acc[4 * cg + 0] * inv0, acc[4 * cg + 1] * inv0);
    *reinterpret_cast<uint32_t*>(orow + 8 * kDim + cg * 8) =
        pack_bf16(acc[4 * cg + 2] * inv1, acc[4 * cg + 3] * inv1);
  }
}

void check_input(const torch::Tensor& t, torch::ScalarType dtype) {
  TORCH_CHECK_VALUE(t.is_cuda(), "expected CUDA tensor");
  TORCH_CHECK_TYPE(t.scalar_type() == dtype, "unexpected dtype");
  TORCH_CHECK_VALUE(t.is_contiguous(), "expected contiguous tensor");
}

// Transposed V scratch, grown on demand and kept for the process lifetime so a
// steady-state call performs no allocation.
bf16* transpose_scratch(const torch::Tensor& v, int64_t halves) {
  static bf16* buf = nullptr;
  static int64_t capacity = 0;
  static int device = -1;
  const int dev = v.get_device();
  if (buf && capacity >= halves && device == dev) return buf;
  if (buf) cudaFree(buf);

  cudaMalloc(&buf, (size_t)halves * sizeof(bf16));
  capacity = halves;
  device = dev;
  return buf;
}

void kernel(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
            const torch::Tensor& block_ids, const torch::Tensor& block_counts,
            const torch::Tensor& out) {
  check_input(q, torch::kBFloat16);
  check_input(k, torch::kBFloat16);
  check_input(v, torch::kBFloat16);
  check_input(out, torch::kBFloat16);
  check_input(block_ids, torch::kInt32);
  check_input(block_counts, torch::kInt32);

  TORCH_CHECK_VALUE(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "expected rank 4 q/k/v");
  TORCH_CHECK_VALUE(block_ids.dim() == 4 && block_counts.dim() == 3, "expected rank 4/3 masks");
  const int64_t batch = q.size(0);
  const int64_t heads = q.size(1);
  const int64_t seq = q.size(2);
  const int64_t dim = q.size(3);
  const int64_t nblocks = block_ids.size(2);
  TORCH_CHECK_VALUE(heads == kHeads && dim == kDim, "heads/dim mismatch");
  TORCH_CHECK_VALUE(k.size(1) == kKVHeads && v.size(1) == kKVHeads, "kv head mismatch");
  TORCH_CHECK_VALUE(k.size(2) == seq && v.size(2) == seq, "kv sequence mismatch");
  TORCH_CHECK_VALUE(block_ids.size(1) == heads && block_ids.size(3) == kCapacity,
                    "mask shape mismatch");
  TORCH_CHECK_VALUE(block_counts.size(0) == batch && block_counts.size(1) == heads &&
                    block_counts.size(2) == nblocks, "count shape mismatch");
  TORCH_CHECK_VALUE(nblocks > 0 && seq == nblocks * kRows, "block size must be 128");
  TORCH_CHECK_VALUE(out.sizes() == q.sizes(), "output shape mismatch");
  TORCH_CHECK_VALUE(q.device() == out.device() && q.device() == k.device(), "device mismatch");
  if (batch == 0) return;

  c10::cuda::CUDAGuard guard(q.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();

  const int smem_bytes = sizeof(Shared);
  static bool smem_configured = false;
  if (!smem_configured) {
    cudaFuncSetAttribute(block_sparse_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);
    smem_configured = true;
  }

  const int keyblocks = (int)(seq / kVBlockKeys);
  bf16* vt = transpose_scratch(v, batch * kKVHeads * keyblocks * kVBlockElems);
  const int64_t chunks = batch * kKVHeads * keyblocks * kKeyChunks;
  transpose_v_kernel<<<(unsigned)((chunks + kThreadsTranspose - 1) / kThreadsTranspose),
                       kThreadsTranspose, 0, stream>>>(
      reinterpret_cast<const bf16*>(v.data_ptr()), vt, chunks, (int)seq, keyblocks);

  const dim3 grid((unsigned)nblocks, (unsigned)heads, (unsigned)batch);
  block_sparse_kernel<<<grid, kThreads, smem_bytes, stream>>>(
      reinterpret_cast<const bf16*>(q.data_ptr()), reinterpret_cast<const bf16*>(k.data_ptr()),
      vt, block_ids.data_ptr<int>(), block_counts.data_ptr<int>(),
      reinterpret_cast<bf16*>(out.data_ptr()), (int)seq, (int)nblocks);
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("kernel", &kernel); }

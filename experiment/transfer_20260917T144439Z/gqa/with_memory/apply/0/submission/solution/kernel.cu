// Causal grouped-query attention prefill, raw CUDA, Hopper wgmma (sm_90a).
//
// One CTA is a single warpgroup (128 threads) that owns 64 query rows of one
// head and walks the causal key range in 64-key tiles:
//
//   grid.x = mTiles * HQ, longest query tile first
//   smem   = Q[64x128] | K[2 x 64x128] | V[3 x 64x128]
//
// Every operand is staged in the canonical unswizzled atom layout: panels of 8
// contiguous head dims laid over 64 rows,
//
//   elem(row, dim) = (dim/8)*512 + row*8 + dim%8
//
// so the 16 B chunk a thread copies out of global memory stays contiguous and
// the descriptors stay valid:
//
//   Q, K  K-major  : lbo = 1024 B (dim panels), sbo = 128 B (row groups)
//   V     MN-major : lbo =  128 B (row groups), sbo = 1024 B (dim panels)
//
//   S = Q K^T : wgmma m64n64k16  x8 k-steps   -> s[32] fp32
//   softmax   : log2 domain, online row max/sum, rescale of o[64]
//   O += P V  : wgmma m64n128k16 with A = P held in registers as bf16x2
//
// V keeps three stages: its PV wgmma is still reading the current tile while
// the next tile is staged. K is consumed by QK alone, so two stages suffice.
//
// The sm_80 mma.sync path below is the fallback for fatbins without sm_90a.
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

// ---------------------------------------------------------------------------
// Problem constants
// ---------------------------------------------------------------------------
constexpr int kDim = 128;                 // head dimension, fixed
constexpr float kSqrtDim = 11.313708498984761f;
constexpr float kLog2e = 1.4426950408889634f;
constexpr float kScaleL2e = kLog2e / kSqrtDim;   // score scale folded into log2
constexpr float kMasked = -1.0e30f;              // stands in for -inf

// ---------------------------------------------------------------------------
// CTA tile: 64 query rows x 64 keys, 4 warps of 16 query rows
// ---------------------------------------------------------------------------
constexpr int kThreads = 128;
constexpr int kWarps = 4;
constexpr int kBr = 64;
constexpr int kBc = 64;
constexpr int kRowsPerWarp = kBr / kWarps;
constexpr int kPad = 8;                   // elements, breaks ldmatrix bank conflicts
constexpr int kQS = kDim + kPad;
constexpr int kKS = kDim + kPad;
constexpr int kVS = kDim + kPad;
constexpr int kChunks = kDim / 8;         // 16 B chunks per row
constexpr int kTileLoads = kBc * kChunks / kThreads;
constexpr int kQLoads = kBr * kChunks / kThreads;

constexpr size_t kSmemElems = (size_t)kBr * kQS + 4 * (size_t)kBc * kKS;
constexpr int kSmemBytesMma = (int)(kSmemElems * sizeof(__nv_bfloat16));

// ---------------------------------------------------------------------------
// wgmma (sm_90a) tile geometry
// ---------------------------------------------------------------------------
constexpr int kChunksPerRow = kDim / 8;                     // 16 B chunks per row
constexpr int kLog2ChunksPerRow = 4;
constexpr int kChunksPerTile = kBr * kChunksPerRow;         // 1024
constexpr int kChunksPerThread = kChunksPerTile / kThreads; // 8
constexpr int kTileElems = kBr * kDim;                      // 8192 bf16
constexpr int kDimPanel = kBr * 8;                          // 512: dim panel stride
constexpr int kKStages = 2;
constexpr int kVStages = 3;
constexpr int kWarpRows = 16;                               // query rows per warp

// Descriptor fields, in 16 byte units.
constexpr uint32_t kQkLbo = 1024;   // K-major: stride between 8-dim panels
constexpr uint32_t kQkSbo = 128;    // K-major: stride between 8-row groups
constexpr uint32_t kVtLbo = 128;    // MN-major V: stride between 8-row groups
constexpr uint32_t kVtSbo = 1024;   // MN-major V: stride between 8-dim panels
constexpr uint64_t kQkStep = 128;   // 16 head dims = 2 dim panels = 2048 B
constexpr uint64_t kPvStep = 16;    // 16 keys = 256 B

constexpr int kQkRegs = kBc / 2;    // m64n64k16 -> 32 fp32 accumulators per thread
constexpr int kPvRegs = kDim / 2;   // m64n128k16 -> 64 fp32 accumulators per thread

// K is consumed by QK alone, so two stages suffice; V is still read by the PV
// wgmma of the tile in flight, so it needs a third.
constexpr int kSmemBytesWgmma = (1 + kKStages + kVStages) * kTileElems * (int)sizeof(__nv_bfloat16);
constexpr int kSmemBytes = kSmemBytesWgmma > kSmemBytesMma ? kSmemBytesWgmma : kSmemBytesMma;

// ---------------------------------------------------------------------------
// PTX helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int kPending>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(kPending));
}

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldsm4(uint32_t addr, uint32_t* r) {
  asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void ldsm4t(uint32_t addr, uint32_t* r) {
  asm("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

// 16x16 operand tile: matrix order (rows 0-7, cols 0-7), (rows 8-15, cols 0-7),
// (rows 0-7, cols 8-15), (rows 8-15, cols 8-15).
__device__ __forceinline__ uint32_t a_row(int lane) { return ((lane >> 3) & 1) * 8 + (lane & 7); }
__device__ __forceinline__ uint32_t a_col(int lane) { return (lane >> 4) * 8; }

// 16(k) x 8(n) B tile of QK^T: k = head dim (contiguous in K's row), n = key.
__device__ __forceinline__ uint32_t k_row(int lane) { return ((lane >> 4) & 1) * 8 + (lane & 7); }
__device__ __forceinline__ uint32_t k_col(int lane) { return ((lane >> 3) & 1) * 8; }

__device__ __forceinline__ float exp2_approx(float x) {
  float r;
  asm("ex2.approx.f32 %0, %1;\n" : "=f"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  const __nv_bfloat162 pair = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&pair);
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// wgmma helpers
// ---------------------------------------------------------------------------
// Generic-proxy shared writes (cp.async) must be published before the async
// proxy (wgmma) reads them.
__device__ __forceinline__ void fence_async_proxy() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// No-swizzle GMMA descriptor: 14 bit shared address, lbo and sbo in 16 B units.
__device__ __forceinline__ uint64_t mk_desc(const void* p, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((smem_u32(p) >> 4) & 0x3FFFu) | ((uint64_t)((lbo >> 4) & 0x3FFFu) << 16) |
         ((uint64_t)((sbo >> 4) & 0x3FFFu) << 32);
}

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n");
}
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

// S = Q K^T: m64n64k16, both operands described in shared memory.
__device__ __forceinline__ void qk_mma(float (&d)[kQkRegs], uint64_t da, uint64_t db) {
  asm volatile(
      "{\n\t"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, 1, 1, 1, 0, 0;\n\t"
      "}\n"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
      : "l"(da), "l"(db));
}

// O += P V: m64n128k16, A = P in registers (bf16x2), B = V in shared memory.
__device__ __forceinline__ void pv_mma(float (&d)[kPvRegs], const uint32_t (&a)[4],
                                       uint64_t db) {
  asm volatile(
      "{\n\t"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %68, 1, 1, 1, 1;\n\t"
      "}\n"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),"+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
}

// ---------------------------------------------------------------------------
// Canonical unswizzled staging: 8 contiguous head dims per panel, 64 rows.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int chunk_off(int chunk) {
  return (chunk & (kChunksPerRow - 1)) * kDimPanel + (chunk >> kLog2ChunksPerRow) * 8;
}

__device__ __forceinline__ size_t chunk_src(int chunk, int stride) {
  return (size_t)(chunk >> kLog2ChunksPerRow) * (size_t)stride +
         (chunk & (kChunksPerRow - 1)) * 8;
}

// Stage one 64 x 128 K/V tile pair, 16 B per thread per step.
__device__ __forceinline__ void stage_kv(__nv_bfloat16* kDst, __nv_bfloat16* vDst,
                                         const __nv_bfloat16* kSrc, const __nv_bfloat16* vSrc,
                                         int stride, int tid) {
#pragma unroll
  for (int i = 0; i < kChunksPerThread; ++i) {
    const int chunk = i * kThreads + tid;
    const int dst = chunk_off(chunk);
    const size_t src = chunk_src(chunk, stride);
    cp_async16(&kDst[dst], kSrc + src);
    cp_async16(&vDst[dst], vSrc + src);
  }
}

__global__ void __launch_bounds__(kThreads) gqa_fwd_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ o,
    const int mTiles, const int heads, const int kvHeads, const int group) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* qS = smem;
  __nv_bfloat16* kS = qS + kTileElems;
  __nv_bfloat16* vS = kS + kKStages * kTileElems;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  // Longest query tile first: the tail of the launch is the cheap work.
  const int head = blockIdx.x % heads;
  const int mIdx = mTiles - 1 - blockIdx.x / heads;
  const int kvHead = head / group;
  const int qRow0 = mIdx * kBr;
  const int headStride = heads * kDim;
  const int kvStride = kvHeads * kDim;

  const __nv_bfloat16* qPtr = q + (size_t)qRow0 * headStride + head * kDim;
  const __nv_bfloat16* kBase = k + kvHead * kDim;
  const __nv_bfloat16* vBase = v + kvHead * kDim;

  // Resident Q tile, then the first K/V tile.
#pragma unroll
  for (int i = 0; i < kChunksPerThread; ++i) {
    const int chunk = i * kThreads + tid;
    cp_async16(&qS[chunk_off(chunk)], qPtr + chunk_src(chunk, headStride));
  }
  cp_commit();
  stage_kv(kS, vS, kBase, vBase, kvStride, tid);
  cp_commit();

  float s[kQkRegs];
  float oAcc[kPvRegs];
#pragma unroll
  for (int i = 0; i < kPvRegs; ++i) oAcc[i] = 0.f;

  float rowMax0 = kMasked, rowMax1 = kMasked;
  float rowSum0 = 0.f, rowSum1 = 0.f;

  const uint64_t dq = mk_desc(qS, kQkLbo, kQkSbo);
  const int row0 = warp * kWarpRows + (lane >> 2);
  const int row1 = row0 + 8;
  const int col0 = 2 * (lane & 3);
  const int nTiles = mIdx + 1;

  int kBuf = 0, vBuf = 0, kStg = 1, vStg = 1;

  for (int j = 0; j < nTiles; ++j) {
    // Prefetch the next tile: 16 B per thread per tensor.
    if (j + 1 < nTiles) {
      const size_t next = (size_t)(j + 1) * kBc * kvStride;
      stage_kv(kS + kStg * kTileElems, vS + vStg * kTileElems, kBase + next, vBase + next,
               kvStride, tid);
      kStg = (kStg + 1) & (kKStages - 1);
      if (++vStg == kVStages) vStg = 0;
      cp_commit();
      cp_wait<1>();
    } else {
      cp_wait<0>();
    }
    fence_async_proxy();
    __syncthreads();

    const uint64_t dk = mk_desc(kS + kBuf * kTileElems, kQkLbo, kQkSbo);
    const uint64_t dv = mk_desc(vS + vBuf * kTileElems, kVtLbo, kVtSbo);

    // ---- S = Q K^T -------------------------------------------------------
#pragma unroll
    for (int i = 0; i < kQkRegs; ++i) s[i] = 0.f;
    wgmma_fence();
#pragma unroll
    for (int ks = 0; ks < kDim / 16; ++ks) qk_mma(s, dq + ks * kQkStep, dk + ks * kQkStep);
    wgmma_commit();
    wgmma_wait();

    // ---- causal mask, only on the tile straddling the diagonal -----------
    if (j == mIdx) {
#pragma unroll
      for (int nt = 0; nt < kQkRegs / 4; ++nt) {
        const int c = nt * 8 + col0;
        s[nt * 4 + 0] = c > row0 ? kMasked : s[nt * 4 + 0];
        s[nt * 4 + 1] = c + 1 > row0 ? kMasked : s[nt * 4 + 1];
        s[nt * 4 + 2] = c > row1 ? kMasked : s[nt * 4 + 2];
        s[nt * 4 + 3] = c + 1 > row1 ? kMasked : s[nt * 4 + 3];
      }
    }

    // ---- online softmax, log2 domain -------------------------------------
    float mx0 = kMasked, mx1 = kMasked;
#pragma unroll
    for (int nt = 0; nt < kQkRegs / 4; ++nt) {
      mx0 = fmaxf(mx0, fmaxf(s[nt * 4 + 0], s[nt * 4 + 1]));
      mx1 = fmaxf(mx1, fmaxf(s[nt * 4 + 2], s[nt * 4 + 3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    const float mNew0 = fmaxf(rowMax0, mx0 * kScaleL2e);
    const float mNew1 = fmaxf(rowMax1, mx1 * kScaleL2e);
    const float alpha0 = exp2_approx(rowMax0 - mNew0);
    const float alpha1 = exp2_approx(rowMax1 - mNew1);
    rowMax0 = mNew0;
    rowMax1 = mNew1;

#pragma unroll
    for (int nt = 0; nt < kPvRegs / 4; ++nt) {
      oAcc[nt * 4 + 0] *= alpha0;
      oAcc[nt * 4 + 1] *= alpha0;
      oAcc[nt * 4 + 2] *= alpha1;
      oAcc[nt * 4 + 3] *= alpha1;
    }

    float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
    for (int nt = 0; nt < kQkRegs / 4; ++nt) {
      s[nt * 4 + 0] = exp2_approx(fmaf(s[nt * 4 + 0], kScaleL2e, -mNew0));
      s[nt * 4 + 1] = exp2_approx(fmaf(s[nt * 4 + 1], kScaleL2e, -mNew0));
      s[nt * 4 + 2] = exp2_approx(fmaf(s[nt * 4 + 2], kScaleL2e, -mNew1));
      s[nt * 4 + 3] = exp2_approx(fmaf(s[nt * 4 + 3], kScaleL2e, -mNew1));
      sum0 += s[nt * 4 + 0] + s[nt * 4 + 1];
      sum1 += s[nt * 4 + 2] + s[nt * 4 + 3];
    }
    sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1);
    sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
    sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1);
    sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
    rowSum0 = rowSum0 * alpha0 + sum0;
    rowSum1 = rowSum1 * alpha1 + sum1;

    // P bf16 pairs for the PV A operand: 8 probability columns per k-step.
    uint32_t p[kQkRegs / 8][4];
#pragma unroll
    for (int st = 0; st < kQkRegs / 8; ++st)
#pragma unroll
      for (int i = 0; i < 4; ++i) p[st][i] = pack_bf16(s[st * 8 + 2 * i], s[st * 8 + 2 * i + 1]);

    // ---- O += P V --------------------------------------------------------
    wgmma_fence();
#pragma unroll
    for (int st = 0; st < kQkRegs / 8; ++st) pv_mma(oAcc, p[st], dv + st * kPvStep);
    wgmma_commit();

    kBuf = (kBuf + 1) & (kKStages - 1);
    if (++vBuf == kVStages) vBuf = 0;
  }

  wgmma_wait();

  // ---- epilogue ----------------------------------------------------------
  const float inv0 = 1.f / rowSum0;
  const float inv1 = 1.f / rowSum1;
  __nv_bfloat16* oRow = o + (size_t)(qRow0 + row0) * headStride + head * kDim;
#pragma unroll
  for (int nt = 0; nt < kPvRegs / 4; ++nt) {
    const int col = nt * 8 + col0;
    *reinterpret_cast<uint32_t*>(&oRow[col]) =
        pack_bf16(oAcc[nt * 4 + 0] * inv0, oAcc[nt * 4 + 1] * inv0);
    *reinterpret_cast<uint32_t*>(&oRow[headStride * 8 + col]) =
        pack_bf16(oAcc[nt * 4 + 2] * inv1, oAcc[nt * 4 + 3] * inv1);
  }
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
  // Tensor-core paths need sm_80+; the fatbin still carries other targets.
  (void)q; (void)k; (void)v; (void)o;
  (void)mTiles; (void)heads; (void)kvHeads; (void)group;
#else
  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* qS = smem;
  __nv_bfloat16* kS = qS + (size_t)kBr * kQS;
  __nv_bfloat16* vS = kS + 2 * (size_t)kBc * kKS;
  
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  // Longest query tile first: the tail of the launch is the cheap work.
  const int head = blockIdx.x % heads;
  const int mIdx = mTiles - 1 - blockIdx.x / heads;
  const int kvHead = head / group;
  const int qRow0 = mIdx * kBr;
  const int headStride = heads * kDim;
  const int kvStride = kvHeads * kDim;

  const __nv_bfloat16* qPtr = q + (size_t)qRow0 * headStride + head * kDim;
  const __nv_bfloat16* kBase = k + kvHead * kDim;
  const __nv_bfloat16* vBase = v + kvHead * kDim;

  // Stage the Q tile, 16 B per thread per step.
#pragma unroll
  for (int i = 0; i < kQLoads; ++i) {
    const int c = i * kThreads + tid;
    const int row = c / kChunks;
    const int col = (c % kChunks) * 8;
    cp_async16(&qS[row * kQS + col], qPtr + (size_t)row * headStride + col);
  }
  cp_commit();

  // Stage the first K/V tile.
  {
    const __nv_bfloat16* kTile = kBase;
    const __nv_bfloat16* vTile = vBase;
#pragma unroll
    for (int i = 0; i < kTileLoads; ++i) {
      const int c = i * kThreads + tid;
      const int row = c / kChunks;
      const int col = (c % kChunks) * 8;
      cp_async16(&kS[row * kKS + col], kTile + (size_t)row * kvStride + col);
      cp_async16(&vS[row * kVS + col], vTile + (size_t)row * kvStride + col);
    }
  }
  cp_commit();

  float oAcc[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) oAcc[i][j] = 0.f;

  float rowMax0 = kMasked, rowMax1 = kMasked;
  float rowSum0 = 0.f, rowSum1 = 0.f;

  cp_wait<0>();
  __syncthreads();

  // Q fragments live in registers for the whole tile loop.
  uint32_t qFrag[8][4];
#pragma unroll
  for (int ks = 0; ks < 8; ++ks) {
    const int r = warp * kRowsPerWarp + a_row(lane);
    const int c = ks * 16 + a_col(lane);
    ldsm4(smem_u32(&qS[r * kQS + c]), qFrag[ks]);
  }

  const int nTiles = mIdx + 1;
  for (int j = 0; j < nTiles; ++j) {
    const int buf = j & 1;
    const __nv_bfloat16* kTile = kS + buf * (kBc * kKS);
    const __nv_bfloat16* vTile = vS + buf * (kBc * kVS);

    // Prefetch the next tile while the current one is consumed.
    if (j + 1 < nTiles) {
      const __nv_bfloat16* kNext = kBase + (size_t)((j + 1) * kBc) * kvStride;
      const __nv_bfloat16* vNext = vBase + (size_t)((j + 1) * kBc) * kvStride;
      __nv_bfloat16* kDst = kS + (buf ^ 1) * (kBc * kKS);
      __nv_bfloat16* vDst = vS + (buf ^ 1) * (kBc * kVS);
#pragma unroll
      for (int i = 0; i < kTileLoads; ++i) {
        const int c = i * kThreads + tid;
        const int row = c / kChunks;
        const int col = (c % kChunks) * 8;
        cp_async16(&kDst[row * kKS + col], kNext + (size_t)row * kvStride + col);
        cp_async16(&vDst[row * kVS + col], vNext + (size_t)row * kvStride + col);
      }
      cp_commit();
      cp_wait<1>();
    } else {
      cp_wait<0>();
    }
    __syncthreads();

    // ---- S = Q K^T -------------------------------------------------------
    float s[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) s[nt][i] = 0.f;

#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t bFrag[4];
        ldsm4(smem_u32(&kTile[(np * 16 + k_row(lane)) * kKS + ks * 16 + k_col(lane)]), bFrag);
        mma16816(s[np * 2 + 0], qFrag[ks], bFrag[0], bFrag[1]);
        mma16816(s[np * 2 + 1], qFrag[ks], bFrag[2], bFrag[3]);
      }
    }

    // ---- causal mask (only the tile straddling the diagonal) -------------
    if (j == mIdx) {
      const int r0 = warp * kRowsPerWarp + (lane >> 2);
      const int r1 = r0 + 8;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int c0 = nt * 8 + 2 * (lane & 3);
        const int c1 = c0 + 1;
        s[nt][0] = (c0 <= r0) ? s[nt][0] : kMasked;
        s[nt][1] = (c1 <= r0) ? s[nt][1] : kMasked;
        s[nt][2] = (c0 <= r1) ? s[nt][2] : kMasked;
        s[nt][3] = (c1 <= r1) ? s[nt][3] : kMasked;
      }
    }

    // ---- online softmax --------------------------------------------------
    float mx0 = kMasked, mx1 = kMasked;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
      mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));

    const float alpha0 = exp2_approx((rowMax0 - mx0) * kScaleL2e);
    const float alpha1 = exp2_approx((rowMax1 - mx1) * kScaleL2e);
    rowMax0 = mx0;
    rowMax1 = mx1;

    if (__any_sync(0xffffffffu, (alpha0 != 1.f) || (alpha1 != 1.f))) {
#pragma unroll
      for (int nt = 0; nt < 16; ++nt) {
        oAcc[nt][0] *= alpha0;
        oAcc[nt][1] *= alpha0;
        oAcc[nt][2] *= alpha1;
        oAcc[nt][3] *= alpha1;
      }
    }

    const float mScaled0 = mx0 * kScaleL2e;
    const float mScaled1 = mx1 * kScaleL2e;
    float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      s[nt][0] = exp2_approx(fmaf(s[nt][0], kScaleL2e, -mScaled0));
      s[nt][1] = exp2_approx(fmaf(s[nt][1], kScaleL2e, -mScaled0));
      s[nt][2] = exp2_approx(fmaf(s[nt][2], kScaleL2e, -mScaled1));
      s[nt][3] = exp2_approx(fmaf(s[nt][3], kScaleL2e, -mScaled1));
      sum0 += s[nt][0] + s[nt][1];
      sum1 += s[nt][2] + s[nt][3];
    }
    sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1);
    sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
    sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1);
    sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
    rowSum0 = rowSum0 * alpha0 + sum0;
    rowSum1 = rowSum1 * alpha1 + sum1;

    // ---- P in bf16 A-fragment registers ---------------------------------
    uint32_t pFrag[4][4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      pFrag[kk][0] = pack_bf16(s[kk * 2 + 0][0], s[kk * 2 + 0][1]);
      pFrag[kk][1] = pack_bf16(s[kk * 2 + 0][2], s[kk * 2 + 0][3]);
      pFrag[kk][2] = pack_bf16(s[kk * 2 + 1][0], s[kk * 2 + 1][1]);
      pFrag[kk][3] = pack_bf16(s[kk * 2 + 1][2], s[kk * 2 + 1][3]);
    }

    // ---- O += P V --------------------------------------------------------
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
      for (int np = 0; np < 8; ++np) {
        uint32_t vFrag[4];
        ldsm4t(smem_u32(&vTile[(ks * 16 + a_row(lane)) * kVS + np * 16 + a_col(lane)]), vFrag);
        mma16816(oAcc[np * 2 + 0], pFrag[ks], vFrag[0], vFrag[1]);
        mma16816(oAcc[np * 2 + 1], pFrag[ks], vFrag[2], vFrag[3]);
      }
    }

    __syncthreads();
  }

  // ---- epilogue ----------------------------------------------------------
  const float inv0 = 1.f / rowSum0;
  const float inv1 = 1.f / rowSum1;
  __nv_bfloat16* oRow =
      o + (size_t)(qRow0 + warp * kRowsPerWarp + (lane >> 2)) * headStride + head * kDim;
#pragma unroll
  for (int nt = 0; nt < 16; ++nt) {
    const int col = nt * 8 + 2 * (lane & 3);
    *reinterpret_cast<uint32_t*>(&oRow[col]) = pack_bf16(oAcc[nt][0] * inv0, oAcc[nt][1] * inv0);
    *reinterpret_cast<uint32_t*>(&oRow[headStride * 8 + col]) =
        pack_bf16(oAcc[nt][2] * inv1, oAcc[nt][3] * inv1);
  }
#endif
}


// ---------------------------------------------------------------------------
// Host wrapper
// ---------------------------------------------------------------------------
void check_input(const torch::Tensor& t, int64_t rows, int64_t heads) {
  TORCH_CHECK_VALUE(t.is_cuda(), "expected a CUDA tensor");
  TORCH_CHECK_TYPE(t.scalar_type() == torch::kBFloat16, "expected bfloat16");
  TORCH_CHECK_VALUE(t.dim() == 4, "expected rank 4 BSHD");
  TORCH_CHECK_VALUE(t.size(0) == 1, "expected batch 1");
  TORCH_CHECK_VALUE(t.size(1) == rows, "sequence length mismatch");
  TORCH_CHECK_VALUE(t.size(2) == heads, "head count mismatch");
  TORCH_CHECK_VALUE(t.size(3) == kDim, "expected head dim 128");
  TORCH_CHECK_VALUE(t.is_contiguous(), "expected contiguous BSHD");
}

void kernel(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
            const torch::Tensor& o) {
  const int64_t seq = q.size(1);
  const int64_t heads = q.size(2);
  const int64_t kvHeads = k.size(2);
  check_input(q, seq, heads);
  check_input(k, seq, kvHeads);
  check_input(v, seq, kvHeads);
  TORCH_CHECK_VALUE(o.is_cuda() && o.scalar_type() == torch::kBFloat16, "bad output");
  TORCH_CHECK_VALUE(o.sizes() == q.sizes(), "output shape must match q");
  TORCH_CHECK_VALUE(o.is_contiguous(), "expected contiguous output");
  TORCH_CHECK_VALUE(q.device() == k.device() && q.device() == v.device() && q.device() == o.device(),
                    "device mismatch");
  TORCH_CHECK_VALUE(heads % kvHeads == 0, "query heads must be a multiple of kv heads");
  TORCH_CHECK_VALUE(seq % kBr == 0, "sequence length must be a multiple of ", kBr);

  c10::cuda::CUDAGuard guard(q.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();

  static bool configured = false;
  if (!configured) {
    const cudaError_t attr = cudaFuncSetAttribute(
        gqa_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
    TORCH_CHECK(attr == cudaSuccess, "smem opt-in failed: ", cudaGetErrorString(attr));
    configured = true;
  }

  const int mTiles = (int)((seq + kBr - 1) / kBr);
  const int blocks = mTiles * (int)heads;
  // at::BFloat16 and __nv_bfloat16 share layout; libtorch only instantiates the former.
  const auto* qPtr = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>());
  const auto* kPtr = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>());
  const auto* vPtr = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>());
  auto* oPtr = reinterpret_cast<__nv_bfloat16*>(o.data_ptr<at::BFloat16>());
  gqa_fwd_kernel<<<blocks, kThreads, kSmemBytes, stream>>>(qPtr, kPtr, vPtr, oPtr, mTiles,
                                                           (int)heads, (int)kvHeads,
                                                           (int)(heads / kvHeads));
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "gqa launch failed: ", cudaGetErrorString(err));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("kernel", &kernel);
}

// DeepSeek-V3.2 supplied-index sparse MLA prefill.
//
// One CTA owns 64 query heads of a single token; the token's 2048 selected KV
// rows stream through shared memory in 64-row tiles. Every warp keeps a
// 32 heads x 128 values slice of the output accumulator, so a CTA covers
// 64 heads x 512 values and the gathered KV rows are re-read only twice
// (once per head half of the token).
//
//   grid  = (tokens, 128 / 64)      block = 8 warps = 256 threads
//   warp  = (mg, g): mg selects 32 heads, g selects 16 keys (QK) / 128 values (PV)
//
//   QK : S[32][16] = Q[32][576] @ K[16][576]^T      (36 k-steps of m16n8k16)
//   PV : O[32][128] += P[32][64] @ V[64][128]       (4 k-steps of m16n8k16)
//
// A head group's four warps each evaluate a 16-key slice of the scores, so the
// probabilities are exchanged through shared memory once per tile:
//
//   warp0 S[32][ 0..15] --.
//   warp1 S[32][16..31] --+--> row max -> P[32][64] --> every warp does PV
//   warp2 S[32][32..47] --+
//   warp3 S[32][48..63] --'
//
// Shared memory (all tiles XOR swizzled on 16 B chunks, 128 B group period):
//   Qs  [64][576] bf16   73728 B   query tile, staged once
//   KVs [2][64][576]    147456 B   double buffered selected KV rows
//   Ps  [2][32][64]       8192 B   probabilities, private to a head group
//   stats                 2048 B   per-tile row max / row sum partials
//                       -------
//                       231424 B

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

using bf16 = __nv_bfloat16;

constexpr int   kHeads      = 128;
constexpr int   kQkDim      = 576;
constexpr int   kValueDim   = 512;
constexpr int   kTopK       = 2048;
constexpr float kScale      = 0.1352337788608801f;

constexpr int kHeadBlock    = 64;                 // heads per CTA
constexpr int kKeyTile      = 64;                 // selected keys per tile
constexpr int kNumTiles     = kTopK / kKeyTile;   // 32
constexpr int kWarps        = 8;
constexpr int kThreads      = kWarps * 32;
constexpr int kHeadsPerHalf = kHeadBlock / 2;     // 32 heads tracked by a warp
constexpr int kValPerWarp   = kValueDim / 4;      // 128 values per warp
constexpr int kKeysPerQk    = kKeyTile / 4;       // 16 keys per warp in QK
constexpr int kChunksPerRow = kQkDim / 8;         // 72 x 16 B chunks per row
constexpr int kStepQk       = kQkDim / 16;        // 36 k-steps
constexpr int kStepPv       = kKeyTile / 16;      // 4 k-steps
constexpr int kSwzMask      = 7;

constexpr int kQElems    = kHeadBlock * kQkDim;
constexpr int kKvElems   = kKeyTile * kQkDim;
constexpr int kPElems    = 2 * 32 * kKeyTile;
constexpr int kStatElems = 2 * 32 * 4;
constexpr int kSmemBytes =
    (kQElems + 2 * kKvElems + kPElems) * (int)sizeof(bf16) +
    2 * kStatElems * (int)sizeof(float);

__device__ __forceinline__ uint32_t sm_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// XOR swizzle: 16-byte chunk c of row r moves to c ^ (r & 7) inside its 128-byte
// group, so every ldmatrix phase lands on 32 distinct banks.
__device__ __forceinline__ int swz_chunk(int row, int c) {
  return (c & ~kSwzMask) | ((c ^ row) & kSwzMask);
}

__device__ __forceinline__ uint32_t tile_addr(const bf16* base, int rowoff,
                                              int rlow, int chunk) {
  return sm_addr(base + (rowoff + swz_chunk(rlow, chunk)) * 8);
}

// Probability tile rows are 64 keys wide (8 chunks) instead of 576 wide.
__device__ __forceinline__ uint32_t p_addr(const bf16* base, int row, int chunk) {
  return sm_addr(base + (row * 8 + swz_chunk(row, chunk)) * 8);
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src,
                                           int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst),
               "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}
__device__ __forceinline__ void cp_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}
__device__ __forceinline__ void bar_sync(int id, int nthreads) {
  asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(nthreads) : "memory");
}

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t* r) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t* r) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(addr));
}

__device__ __forceinline__ void mma_bf16(float* d, const uint32_t* a, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint16_t to_bf16(float x) {
  return __bfloat16_as_ushort(__float2bfloat16(x));
}
__device__ __forceinline__ float from_bf16(uint16_t h) {
  return __bfloat162float(__ushort_as_bfloat16(h));
}

// One PV pass: acc[32 heads][128 values] += P[32][64] @ V[64][128].
__device__ __forceinline__ void pv_tile(const bf16* kvbuf, const bf16* pgrp,
                                        int gid, int lane,
                                        float (&acc)[2][16][4]) {
  const int lg = lane >> 3;
  const int lr = lane & 7;
  const int cadd = lg >> 1;
  const int vrow = ((lg & 1) << 3) + lr;
  const int vlow = vrow & kSwzMask;
  const int vchunk0 = gid * 16;
#pragma unroll
  for (int kk = 0; kk < kStepPv; ++kk) {
    uint32_t a[2][4];
    ldsm_x4(p_addr(pgrp, ((lg & 1) << 3) + lr, kk * 2 + cadd), a[0]);
    ldsm_x4(p_addr(pgrp, ((lg & 1) << 3) + lr + 16, kk * 2 + cadd), a[1]);
    const bf16* vbase = kvbuf + (16 * kk) * kQkDim;
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) {
      uint32_t b[4];
      ldsm_x4_t(
          tile_addr(vbase, vrow * kChunksPerRow, vlow, vchunk0 + jj * 2 + cadd),
          b);
      mma_bf16(acc[0][jj * 2 + 0], a[0], b[0], b[1]);
      mma_bf16(acc[0][jj * 2 + 1], a[0], b[2], b[3]);
      mma_bf16(acc[1][jj * 2 + 0], a[1], b[0], b[1]);
      mma_bf16(acc[1][jj * 2 + 1], a[1], b[2], b[3]);
    }
  }
}

// P = exp(s - rowmax) as bf16, plus this warp's partial row sums.
__device__ __forceinline__ void emit_probs(const float (&s)[2][2][4],
                                           const float (&msh)[4], bf16* pgrp,
                                           int gid, int lane, int cq,
                                           float (&lsum)[4]) {
  const int hq = lane >> 2;
#pragma unroll
  for (int i = 0; i < 4; ++i) lsum[i] = 0.f;
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int i = mt * 2 + j;
        const int row = mt * 16 + hq + j * 8;
        const uint16_t h0 = to_bf16(__expf(s[mt][nt][j * 2 + 0] - msh[i]));
        const uint16_t h1 = to_bf16(__expf(s[mt][nt][j * 2 + 1] - msh[i]));
        *reinterpret_cast<uint32_t*>(
            pgrp + (row * 8 + swz_chunk(row, gid * 2 + nt)) * 8 + cq * 2) =
            (uint32_t)h0 | ((uint32_t)h1 << 16);
        lsum[i] += from_bf16(h0) + from_bf16(h1);
      }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float v = lsum[i];
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    lsum[i] = v;
  }
}

// Stage 64 rows of a [rows][576] bf16 source straight into a swizzled tile.
// A warp owns 8 rows; 72 chunks per row are covered by 3 passes over 32 lanes.
__device__ __forceinline__ void stage_dense(bf16* dst, const bf16* src,
                                            int tid) {
  const int lane = tid & 31;
  const int wid = tid >> 5;
  constexpr int kRowsPerWarp = kKeyTile / kWarps;
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) {
    const int row = wid * kRowsPerWarp + r;
    const bf16* srow = src + (size_t)row * kQkDim;
    bf16* drow = dst + row * kQkDim;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const int c = j * 32 + lane;
      if (c < kChunksPerRow)
        cp_async16(sm_addr(drow + swz_chunk(row, c) * 8), srow + c * 8, 16);
    }
  }
}

// Same, but rows come from the sparse index list; an invalid index is staged as
// zero (cp.async src_bytes = 0 zero-fills) and later masked to -inf.
__device__ __forceinline__ void stage_gather(bf16* dst, const bf16* kv,
                                             const int* idx, int ntokens,
                                             int tid) {
  const int lane = tid & 31;
  const int wid = tid >> 5;
  constexpr int kRowsPerWarp = kKeyTile / kWarps;
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) {
    const int row = wid * kRowsPerWarp + r;
    const int raw = __ldg(idx + row);
    const bool ok = (raw >= 0) && (raw < ntokens);
    const bf16* srow = kv + (size_t)(ok ? raw : 0) * kQkDim;
    bf16* drow = dst + row * kQkDim;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const int c = j * 32 + lane;
      if (c < kChunksPerRow)
        cp_async16(sm_addr(drow + swz_chunk(row, c) * 8), srow + c * 8,
                   ok ? 16 : 0);
    }
  }
}

__global__ void __launch_bounds__(kThreads, 1)
sparse_mla_kernel(const bf16* __restrict__ q, const bf16* __restrict__ kv,
                  const int* __restrict__ ind, int ntokens,
                  bf16* __restrict__ out, float* __restrict__ maxlog,
                  float* __restrict__ lse) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  extern __shared__ char smem[];
  bf16* Qs = reinterpret_cast<bf16*>(smem);
  bf16* KVs = Qs + kQElems;
  bf16* Ps = KVs + 2 * kKvElems;
  float* pMax = reinterpret_cast<float*>(Ps + kPElems);
  float* pSum = pMax + kStatElems;

  const int tok = blockIdx.x;
  const int h0 = blockIdx.y * kHeadBlock;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int mg = wid >> 2;   // head half of this warp
  const int gid = wid & 3;   // key slice (QK) and value slice (PV)
  const int lr = lane & 7;   // row inside an ldmatrix tile
  const int lg = lane >> 3;  // ldmatrix tile selector
  const int hq = lane >> 2;  // row pair inside a 16x8 fragment
  const int cq = lane & 3;   // column pair inside a 16x8 fragment
  const int barid = 1 + mg;

  // A fragment rows: head rows of this warp's 32-head half.
  const int qrow0 = mg * kHeadsPerHalf + ((lg & 1) << 3) + lr;
  const int qrow1 = qrow0 + 16;
  const int qbase0 = qrow0 * kChunksPerRow;
  const int qbase1 = qrow1 * kChunksPerRow;
  const int qlow0 = qrow0 & kSwzMask;
  const int qlow1 = qrow1 & kSwzMask;
  // B fragment rows: key rows of this warp's 16-key slice.
  const int krow = gid * kKeysPerQk + ((lg & 1) << 3) + lr;
  const int kbase = krow * kChunksPerRow;
  const int klow = krow & kSwzMask;
  const int cadd = lg >> 1;  // second chunk of a 16 element k-step

  const int* myidx = ind + (size_t)tok * kTopK;
  const bf16* qsrc = q + (size_t)tok * kHeads * kQkDim + (size_t)h0 * kQkDim;

  // ---- prologue: query tile + the first key tile pair --------------------
  stage_dense(Qs, qsrc, tid);
  stage_gather(KVs, kv, myidx, ntokens, tid);
  cp_commit();
  stage_gather(KVs + kKvElems, kv, myidx + kKeyTile, ntokens, tid);
  cp_commit();

  float acc[2][16][4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < 16; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[mt][nt][i] = 0.f;

  float mrun[4];
  float msh[4];
  float lrun[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    mrun[i] = -INFINITY;
    msh[i] = 0.f;
    lrun[i] = 0.f;
  }

  // Both key tiles of a pair share every query fragment load, so the query
  // tile is read half as often.
  for (int t = 0; t < kNumTiles; t += 2) {
    cp_wait_all();
    __syncthreads();

    const bf16* kv0 = KVs + (t & 1) * kKvElems;
    const bf16* kv1 = KVs + ((t + 1) & 1) * kKvElems;

    // ---- QK^T over the pair's 128 keys -----------------------------------
    float s0[2][2][4];
    float s1[2][2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          s0[mt][nt][i] = 0.f;
          s1[mt][nt][i] = 0.f;
        }

#pragma unroll
    for (int ks = 0; ks < kStepQk; ++ks) {
      const int c = ks * 2 + cadd;
      uint32_t a0[4], a1[4], b0[4], b1[4];
      ldsm_x4(tile_addr(Qs, qbase0, qlow0, c), a0);
      ldsm_x4(tile_addr(Qs, qbase1, qlow1, c), a1);
      ldsm_x4(tile_addr(kv0, kbase, klow, c), b0);
      ldsm_x4(tile_addr(kv1, kbase, klow, c), b1);
      mma_bf16(s0[0][0], a0, b0[0], b0[2]);
      mma_bf16(s0[0][1], a0, b0[1], b0[3]);
      mma_bf16(s0[1][0], a1, b0[0], b0[2]);
      mma_bf16(s0[1][1], a1, b0[1], b0[3]);
      mma_bf16(s1[0][0], a0, b1[0], b1[2]);
      mma_bf16(s1[0][1], a0, b1[1], b1[3]);
      mma_bf16(s1[1][0], a1, b1[0], b1[2]);
      mma_bf16(s1[1][1], a1, b1[1], b1[3]);
    }

    // ---- mask, scale, per-row partial max over the pair's 128 keys -------
    float valid[2][2][2];  // [tile][n-tile][column of the pair]
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int col = gid * kKeysPerQk + nt * 8 + cq * 2 + j;
        const int r0 = __ldg(myidx + t * kKeyTile + col);
        const int r1 = __ldg(myidx + (t + 1) * kKeyTile + col);
        valid[0][nt][j] = (r0 >= 0 && r0 < ntokens) ? 1.f : 0.f;
        valid[1][nt][j] = (r1 >= 0 && r1 < ntokens) ? 1.f : 0.f;
      }

#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const float m00 = valid[0][nt][0];
          const float m01 = valid[0][nt][1];
          const float m10 = valid[1][nt][0];
          const float m11 = valid[1][nt][1];
          s0[mt][nt][j * 2 + 0] = m00 ? s0[mt][nt][j * 2 + 0] * kScale : -INFINITY;
          s0[mt][nt][j * 2 + 1] = m01 ? s0[mt][nt][j * 2 + 1] * kScale : -INFINITY;
          s1[mt][nt][j * 2 + 0] = m10 ? s1[mt][nt][j * 2 + 0] * kScale : -INFINITY;
          s1[mt][nt][j * 2 + 1] = m11 ? s1[mt][nt][j * 2 + 1] * kScale : -INFINITY;
        }

    float tmax[4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        float m = fmaxf(fmaxf(s0[mt][0][j * 2], s0[mt][0][j * 2 + 1]),
                        fmaxf(s0[mt][1][j * 2], s0[mt][1][j * 2 + 1]));
        m = fmaxf(m, fmaxf(fmaxf(s1[mt][0][j * 2], s1[mt][0][j * 2 + 1]),
                           fmaxf(s1[mt][1][j * 2], s1[mt][1][j * 2 + 1])));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        tmax[mt * 2 + j] = m;
      }
    if (cq == 0) {
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int j = 0; j < 2; ++j)
          pMax[mg * (32 * 4) + (mt * 16 + hq + j * 8) * 4 + gid] = tmax[mt * 2 + j];
    }
    bar_sync(barid, 128);

    // ---- rescale the accumulator once for the whole pair -----------------
    float alpha[4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int i = mt * 2 + j;
        const float4 pm =
            *reinterpret_cast<const float4*>(pMax + mg * (32 * 4) +
                                             (mt * 16 + hq + j * 8) * 4);
        const float tile = fmaxf(fmaxf(pm.x, pm.y), fmaxf(pm.z, pm.w));
        const float mnew = fmaxf(mrun[i], tile);
        msh[i] = (mnew == -INFINITY) ? 0.f : mnew;
        alpha[i] = __expf(mrun[i] - msh[i]);
        mrun[i] = mnew;
      }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nt = 0; nt < 16; ++nt) {
        const float a0 = alpha[mt * 2 + 0];
        const float a1 = alpha[mt * 2 + 1];
        acc[mt][nt][0] *= a0;
        acc[mt][nt][1] *= a0;
        acc[mt][nt][2] *= a1;
        acc[mt][nt][3] *= a1;
      }

    // ---- probabilities and PV for the first tile -------------------------
    bf16* pgrp = Ps + mg * (32 * kKeyTile);
    float lsum[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) lrun[i] *= alpha[i];
    emit_probs(s0, msh, pgrp, gid, lane, cq, lsum);
#pragma unroll
    for (int i = 0; i < 4; ++i) lrun[i] += lsum[i];
    // PV consumes all 64 key columns, i.e. every warp's slice of P.
    bar_sync(barid, 128);
    pv_tile(kv0, pgrp, gid, lane, acc);

    // The next gather needs every warp to be done with this tile's V rows.
    __syncthreads();
    if (t + 2 < kNumTiles) {
      stage_gather(KVs + (t & 1) * kKvElems, kv, myidx + (t + 2) * kKeyTile,
                   ntokens, tid);
      cp_commit();
    }

    // ---- probabilities and PV for the second tile ------------------------
    float lsum2[4];
    emit_probs(s1, msh, pgrp, gid, lane, cq, lsum2);
#pragma unroll
    for (int i = 0; i < 4; ++i) lrun[i] += lsum2[i];
    bar_sync(barid, 128);
    pv_tile(kv1, pgrp, gid, lane, acc);

    __syncthreads();
    if (t + 3 < kNumTiles) {
      stage_gather(KVs + ((t + 1) & 1) * kKvElems, kv,
                   myidx + (t + 3) * kKeyTile, ntokens, tid);
      cp_commit();
    }
  }

  // ---- epilogue ----------------------------------------------------------
  // lrun holds this warp's four key-slice partial sums. Fold the slices of the
  // row together once here rather than round-tripping shared memory per tile.
  if (cq == 0) {
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int r = (mt * 16 + hq + j * 8) * 4;
        pSum[mg * (32 * 4) + r + gid] = lrun[mt * 2 + j];
      }
  }
  bar_sync(barid, 128);
  float tot[4];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const float4 ps = *reinterpret_cast<const float4*>(
          pSum + mg * (32 * 4) + (mt * 16 + hq + j * 8) * 4);
      tot[mt * 2 + j] = (ps.x + ps.y) + (ps.z + ps.w);
    }

  float rcp[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) rcp[i] = (tot[i] > 0.f) ? 1.f / tot[i] : 0.f;
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      const int head = h0 + mg * kHeadsPerHalf + mt * 16 + hq;
      const int vcol = gid * kValPerWarp + nt * 8 + cq * 2;
      bf16* dst = out + (size_t)tok * kHeads * kValueDim +
                  (size_t)head * kValueDim + vcol;
      const float r0 = rcp[mt * 2 + 0];
      const float r1 = rcp[mt * 2 + 1];
      const uint16_t lo0 = to_bf16(acc[mt][nt][0] * r0);
      const uint16_t hi0 = to_bf16(acc[mt][nt][1] * r0);
      const uint16_t lo1 = to_bf16(acc[mt][nt][2] * r1);
      const uint16_t hi1 = to_bf16(acc[mt][nt][3] * r1);
      *reinterpret_cast<uint32_t*>(dst) =
          (uint32_t)lo0 | ((uint32_t)hi0 << 16);
      *reinterpret_cast<uint32_t*>(dst + 8 * kValueDim) =
          (uint32_t)lo1 | ((uint32_t)hi1 << 16);
    }
  if (cq == 0) {
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int i = mt * 2 + j;
        const int head = h0 + mg * kHeadsPerHalf + mt * 16 + hq + j * 8;
        maxlog[(size_t)tok * kHeads + head] = mrun[i];
        lse[(size_t)tok * kHeads + head] = mrun[i] + logf(tot[i]);
      }
  }
#endif
}

void check_input(const torch::Tensor& t, at::ScalarType dtype, int dim,
                 const char* name) {
  TORCH_CHECK_VALUE(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK_TYPE(t.scalar_type() == dtype, name, " dtype mismatch");
  TORCH_CHECK_VALUE(t.dim() == dim, name, " rank mismatch");
  TORCH_CHECK_VALUE(t.is_contiguous(), name, " must be contiguous");
}

void kernel(const torch::Tensor& q, const torch::Tensor& kv,
            const torch::Tensor& indices, const torch::Tensor& output,
            const torch::Tensor& max_logits, const torch::Tensor& lse) {
  check_input(q, torch::kBFloat16, 3, "q");
  check_input(kv, torch::kBFloat16, 3, "kv");
  check_input(indices, torch::kInt32, 3, "indices");
  check_input(output, torch::kBFloat16, 3, "output");
  check_input(max_logits, torch::kFloat32, 2, "max_logits");
  check_input(lse, torch::kFloat32, 2, "lse");

  const int64_t ntokens = q.size(0);
  TORCH_CHECK_VALUE(q.size(1) == kHeads && kv.size(0) == ntokens &&
                        indices.size(0) == ntokens,
                    "unexpected problem shape");
  TORCH_CHECK_VALUE(q.size(2) == kQkDim && kv.size(2) == kQkDim &&
                        indices.size(2) == kTopK && indices.size(1) == 1,
                    "unexpected problem shape");
  TORCH_CHECK_VALUE(output.size(2) == kValueDim && output.size(1) == kHeads,
                    "unexpected output shape");
  TORCH_CHECK_VALUE(q.device() == kv.device() && q.device() == indices.device() &&
                        q.device() == output.device() &&
                        q.device() == max_logits.device(),
                    "device mismatch");

  c10::cuda::CUDAGuard guard(q.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(q.get_device()).stream();

  static bool configured = false;
  if (!configured) {
    const cudaError_t attr = cudaFuncSetAttribute(
        sparse_mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSmemBytes);
    TORCH_CHECK(attr == cudaSuccess, cudaGetErrorString(attr));
    configured = true;
  }

  const dim3 grid((unsigned)ntokens, kHeads / kHeadBlock);
  sparse_mla_kernel<<<grid, kThreads, kSmemBytes, stream>>>(
      reinterpret_cast<const bf16*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const bf16*>(kv.data_ptr<at::BFloat16>()),
      indices.data_ptr<int>(), (int)ntokens,
      reinterpret_cast<bf16*>(output.data_ptr<at::BFloat16>()),
      max_logits.data_ptr<float>(), lse.data_ptr<float>());
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("kernel", &kernel);
}

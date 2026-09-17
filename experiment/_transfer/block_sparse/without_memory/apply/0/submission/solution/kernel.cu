// Causal block-sparse attention: gather the selected key blocks, apply ordinary
// masked attention (FlashAttention-2 style) over the compressed key list.
//
//   queries per block : 128          (block_ids.shape[-2] == S / 128)
//   head dimension    : 128
//   key blocks        : up to `cap` ids taken from block_ids, -1 pads
//   diagonal block    : causal; strictly earlier blocks are fully visible
//   keys from padded slots or from blocks beyond the query block are dead
//
// Fast path: one CTA per (batch, head, query block), 8 warps x 16 rows each,
// 128x128 bf16 tiles in shared memory, mma.m16n8k16 tensor-core math.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

using bf16 = __nv_bfloat16;
using bf16x2 = __nv_bfloat162;

constexpr int kHeadDim = 128;    // head dimension handled by the fast kernel
constexpr int kBlockQ = 128;     // queries per query block
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kRowsWarp = kBlockQ / kWarps;  // 16
constexpr int kChunkKeys = 128;              // keys per online-softmax step
constexpr int kChunks = kBlockQ / kChunkKeys;
constexpr int kKgPerChunk = kChunkKeys / 16;   // 16-key mma groups per chunk
constexpr int kNtPerChunk = kChunkKeys / 8;    // 8-key score tiles per chunk
constexpr int kCapMax = 64;                  // largest block_ids capacity honored
constexpr int kRowPad = 8;                 // elements of padding per tile row
constexpr int kRowStride = kBlockQ + kRowPad;
constexpr int kTileElems = kBlockQ * kRowStride;
constexpr int kTileBytes = kTileElems * 2;
constexpr float kLog2e = 1.4426950408889634f;
// 1/sqrt(128); folded into the exp2 argument so the softmax runs on raw dots.
constexpr float kScoreScale = 0.08838834764831845f;
constexpr float kExpScale = kScoreScale * kLog2e;
constexpr float kMasked = -1.0e30f;

// bf16 mma needs sm_80; older targets keep only the scalar fallback. The host
// pass sees an undefined __CUDA_ARCH__ and keeps the tensor-core path.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
#define KLINEAGE_TC 0
#else
#define KLINEAGE_TC 1
#endif

// ---------------------------------------------------------------- primitives

#if KLINEAGE_TC

// 128x128 bf16 tile padded to a 136-element (272 B) row stride. Row r starts at
// bank 4r, so the eight rows of an ldmatrix tile cover all 32 banks with no
// conflict, and every tile address stays affine in the row/column indices.
// (row, chunk) -> element offset, chunk in 16-byte (8 element) units.
__device__ __forceinline__ int tile_ofs(int row, int chunk) {
  return row * kRowStride + chunk * 8;
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  const uint32_t s = (uint32_t)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Stage one 128x128 bf16 tile: 2048 chunks of 16 bytes, 8 per thread.
__device__ __forceinline__ void load_tile(bf16* dst, const bf16* src, int tid) {
  const bf16* base = src;
  bf16* dbase = dst;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int idx = tid + (i << 8);
    cp_async16(dbase + tile_ofs(idx >> 4, idx & 15), base + idx * 8);
  }
}

__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                        uint32_t& r3, const bf16* p) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(a));
}

__device__ __forceinline__ void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                          uint32_t& r3, const bf16* p) {
  const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(a));
}

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

#endif  // KLINEAGE_TC

// Shared helpers available on every target.
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  const bf16x2 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ float warp_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}

__device__ __forceinline__ float warp_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

// ex2.approx is one instruction; exp2f adds a range compare plus a scaled
// second EX2 for |x| >= 126, which the softmax never needs (its arguments are
// non-positive).
__device__ __forceinline__ float fast_exp2(float x) {
  float r;
  asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}

// ------------------------------------------------------------------- kernels

__global__ __launch_bounds__(kThreads)
void sparse_attn_fast(const bf16* __restrict__ q, const bf16* __restrict__ k,
                      const bf16* __restrict__ v, const int* __restrict__ ids,
                      const int* __restrict__ counts, bf16* __restrict__ out,
                      const int batch, const int heads, const int seq,
                      const int kv_heads, const int nblk, const int cap) {
#if KLINEAGE_TC
  extern __shared__ __align__(16) char smem[];
  bf16* qs = reinterpret_cast<bf16*>(smem);
  bf16* kbuf = qs + kTileElems;
  bf16* vbuf = kbuf + 2 * kTileElems;

  __shared__ int s_ids[kCapMax];
  __shared__ int s_nslots;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int row0 = warp * kRowsWarp;
  const int gid = lane >> 2;
  const int tg = lane & 3;

  // Work unit: (batch, head, query block). Consecutive CTAs share a query
  // block, so the diagonal tile is fetched by up to `heads` CTAs at once.
  const int per_blk = batch * heads;
  const int qblk = nblk - 1 - blockIdx.x / per_blk;
  const int bh = blockIdx.x % per_blk;
  const int h = bh % heads;
  const int b = bh / heads;
  const int kvh = h / (heads / kv_heads);

  // Compress the selected blocks: reversed order (nearest block first), -1 pads
  // and blocks past the query block dropped since every key they own is masked.
  if (tid == 0) {
    const int* myids = ids + ((size_t)bh * nblk + qblk) * cap;
    const int raw = counts[(size_t)bh * nblk + qblk];
    const int cnt = raw < cap ? raw : cap;
    int n = 0;
    for (int slot = cnt - 1; slot >= 0; --slot) {
      const int bid = myids[slot];
      if (bid < 0 || bid > qblk) continue;
      s_ids[n++] = bid;
    }
    s_nslots = n;
  }
  __syncthreads();

  bf16* orow = out + (((size_t)b * heads + h) * seq + (size_t)qblk * kBlockQ +
                      row0 + gid) * kHeadDim;
  const int nslots = s_nslots;
  if (nslots == 0) {
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      const bf16x2 z = __floats2bfloat162_rn(0.f, 0.f);
      *reinterpret_cast<bf16x2*>(orow + nt * 8 + 2 * tg) = z;
      *reinterpret_cast<bf16x2*>(orow + 8 * kHeadDim + nt * 8 + 2 * tg) = z;
    }
    return;
  }

  const bf16* ksrc = k + ((size_t)b * kv_heads + kvh) * seq * kHeadDim;
  const bf16* vsrc = v + ((size_t)b * kv_heads + kvh) * seq * kHeadDim;
  const bf16* qsrc =
      q + (((size_t)b * heads + h) * seq + (size_t)qblk * kBlockQ) * kHeadDim;

  // Prologue: Q tile plus the first key/value tile, one commit group apart.
  load_tile(qs, qsrc, tid);
  cp_commit();
  load_tile(kbuf, ksrc + (size_t)s_ids[0] * kBlockQ * kHeadDim, tid);
  load_tile(vbuf, vsrc + (size_t)s_ids[0] * kBlockQ * kHeadDim, tid);
  cp_commit();
  cp_wait<1>();
  __syncthreads();

  const int lr_q = (lane & 7) + ((lane & 8) ? 8 : 0);
  const int lc_q = (lane >> 4) & 1;

  float o[16][4];
#pragma unroll
  for (int nt = 0; nt < 16; ++nt)
#pragma unroll
    for (int j = 0; j < 4; ++j) o[nt][j] = 0.f;
  float mrow[2] = {kMasked, kMasked};
  float lrow[2] = {0.f, 0.f};

  const int lr_k = (lane & 7) + ((lane & 16) ? 8 : 0);
  const int lc_k = (lane >> 3) & 1;
  const int lr_v = (lane & 7) + ((lane & 8) ? 8 : 0);
  const int lc_v = (lane >> 4) & 1;

  // Per-thread smem bases. Every ldmatrix address below is base + constant, so
  // ptxas folds the whole swizzle arithmetic into the load's immediate offset.
  const int q_off = tile_ofs(row0 + lr_q, lc_q);
  const int k_off = tile_ofs(lr_k, lc_k);
  const int v_off = tile_ofs(lr_v, lc_v);
  const int qrow_a = row0 + gid;
  const int qrow_b = qrow_a + 8;

  for (int slot = 0; slot < nslots; ++slot) {
    const int bid = s_ids[slot];
    const bf16* kcur = kbuf + (slot & 1) * kTileElems;
    const bf16* vcur = vbuf + (slot & 1) * kTileElems;

    if (slot + 1 < nslots) {
      const size_t off = (size_t)s_ids[slot + 1] * kBlockQ * kHeadDim;
      load_tile(kbuf + ((slot + 1) & 1) * kTileElems, ksrc + off, tid);
      load_tile(vbuf + ((slot + 1) & 1) * kTileElems, vsrc + off, tid);
      cp_commit();
      cp_wait<1>();
    } else {
      cp_wait<0>();
    }
    __syncthreads();

    // The diagonal block is causal: a warp stops at the last chunk its rows can
    // reach, and only that chunk can hold masked keys.
    const bool causal = (bid == qblk);
    const int reach = (row0 + kRowsWarp + kChunkKeys - 1) / kChunkKeys;
    const int chn = causal && reach < kChunks ? reach : kChunks;

    for (int ch = 0; ch < chn; ++ch) {
      const int key_base = ch * kChunkKeys;
    const int krow_off = key_base * kRowStride;
      float s[kNtPerChunk][4];
#pragma unroll
      for (int nt = 0; nt < kNtPerChunk; ++nt) {
        s[nt][0] = 0.f;
        s[nt][1] = 0.f;
        s[nt][2] = 0.f;
        s[nt][3] = 0.f;
      }

      // S = Q @ K^T for 16 rows x kChunkKeys keys.
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t qa[4];
        ldsm_x4(qa[0], qa[1], qa[2], qa[3],
                qs + (q_off + ks * 16));
#pragma unroll
        for (int kg = 0; kg < kKgPerChunk; ++kg) {
          uint32_t b0, b1, b2, b3;
          ldsm_x4(b0, b1, b2, b3,
                  kcur + (k_off + krow_off + kg * (16 * kRowStride) + ks * 16));
          mma16816(s[kg * 2], qa, b0, b1);
          mma16816(s[kg * 2 + 1], qa, b2, b3);
        }
      }

      if (causal && ch == row0 / kChunkKeys) {
#pragma unroll
        for (int nt = 0; nt < kNtPerChunk; ++nt) {
          const int kb = key_base + nt * 8 + 2 * tg;
          if (kb > qrow_a) s[nt][0] = kMasked;
          if (kb + 1 > qrow_a) s[nt][1] = kMasked;
          if (kb > qrow_b) s[nt][2] = kMasked;
          if (kb + 1 > qrow_b) s[nt][3] = kMasked;
        }
      }

      // Online softmax over this chunk.
      float mx0 = fmaxf(s[0][0], s[0][1]);
      float mx1 = fmaxf(s[0][2], s[0][3]);
#pragma unroll
      for (int nt = 1; nt < kNtPerChunk; ++nt) {
        mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
        mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
      }
      const float mnew0 = fmaxf(mrow[0], warp_max(mx0));
      const float mnew1 = fmaxf(mrow[1], warp_max(mx1));
      // exp2(x * e) = exp2(fma(x, e, t)) with t = -m * e; one FFMA per element.
      const float t0 = -mnew0 * kExpScale;
      const float t1 = -mnew1 * kExpScale;
      const float a0 = fast_exp2(fmaf(mrow[0], kExpScale, t0));
      const float a1 = fast_exp2(fmaf(mrow[1], kExpScale, t1));
#pragma unroll
      for (int nt = 0; nt < 16; ++nt) {
        o[nt][0] *= a0;
        o[nt][1] *= a0;
        o[nt][2] *= a1;
        o[nt][3] *= a1;
      }

      float p0 = 0.f;
      float p1 = 0.f;
#pragma unroll
      for (int nt = 0; nt < kNtPerChunk; ++nt) {
        s[nt][0] = fast_exp2(fmaf(s[nt][0], kExpScale, t0));
        s[nt][1] = fast_exp2(fmaf(s[nt][1], kExpScale, t0));
        s[nt][2] = fast_exp2(fmaf(s[nt][2], kExpScale, t1));
        s[nt][3] = fast_exp2(fmaf(s[nt][3], kExpScale, t1));
        p0 += s[nt][0] + s[nt][1];
        p1 += s[nt][2] + s[nt][3];
      }
      lrow[0] = lrow[0] * a0 + warp_sum(p0);
      lrow[1] = lrow[1] * a1 + warp_sum(p1);
      mrow[0] = mnew0;
      mrow[1] = mnew1;

      // O += P @ V, with P taken straight from the score fragments.
#pragma unroll
      for (int kk = 0; kk < kKgPerChunk; ++kk) {
        const uint32_t pa[4] = {pack2(s[2 * kk][0], s[2 * kk][1]),
                                pack2(s[2 * kk][2], s[2 * kk][3]),
                                pack2(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                                pack2(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
        for (int dg = 0; dg < 8; ++dg) {
          uint32_t b0, b1, b2, b3;
          ldsm_x4_t(b0, b1, b2, b3,
                    vcur + (v_off + krow_off + kk * (16 * kRowStride) + dg * 16));
          mma16816(o[2 * dg], pa, b0, b1);
          mma16816(o[2 * dg + 1], pa, b2, b3);
        }
      }
    }
    __syncthreads();
  }

  const float inv0 = lrow[0] > 0.f ? 1.f / lrow[0] : 0.f;
  const float inv1 = lrow[1] > 0.f ? 1.f / lrow[1] : 0.f;
#pragma unroll
  for (int nt = 0; nt < 16; ++nt) {
    *reinterpret_cast<bf16x2*>(orow + nt * 8 + 2 * tg) =
        __floats2bfloat162_rn(o[nt][0] * inv0, o[nt][1] * inv0);
    *reinterpret_cast<bf16x2*>(orow + 8 * kHeadDim + nt * 8 + 2 * tg) =
        __floats2bfloat162_rn(o[nt][2] * inv1, o[nt][3] * inv1);
  }
#endif  // KLINEAGE_TC
}


// One thread per query row, three passes (max, sum, weighted value). Correct
// for any head dimension and any query block size; fallback only.
struct NaiveCtx {
  const bf16* qrow;
  const bf16* khead;
  const bf16* vhead;
  const int* myids;
  int cnt;
  int bs;
  int64_t qpos;
  float scale;

  __device__ __forceinline__ float dot(const bf16* krow) const {
    float acc = 0.f;
    for (int d = 0; d < kHeadDim; ++d)
      acc += __bfloat162float(qrow[d]) * __bfloat162float(krow[d]);
    return acc * scale;
  }

  template <typename Fn>
  __device__ __forceinline__ void each_key(Fn&& fn) const {
    for (int s = 0; s < cnt; ++s) {
      const int bid = myids[s];
      if (bid < 0) continue;
      const int64_t base = (int64_t)bid * bs;
      if (base > qpos) continue;
      const int lim = (int)(qpos - base) + 1;
      const int n = lim < bs ? lim : bs;
      for (int off = 0; off < n; ++off) {
        fn(khead + (base + off) * kHeadDim, vhead + (base + off) * kHeadDim);
      }
    }
  }
};

__global__ void sparse_attn_naive(const bf16* __restrict__ q,
                                  const bf16* __restrict__ k,
                                  const bf16* __restrict__ v,
                                  const int* __restrict__ ids,
                                  const int* __restrict__ counts,
                                  bf16* __restrict__ out, const int batch,
                                  const int heads, const int seq,
                                  const int kv_heads, const int nblk,
                                  const int cap, const float scale) {
  const int bs = seq / nblk;
  const int64_t total = (int64_t)batch * heads * nblk * bs;
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const int r = (int)(idx % bs);
  const int qblk = (int)(idx / bs % nblk);
  const int h = (int)(idx / ((int64_t)bs * nblk) % heads);
  const int b = (int)(idx / ((int64_t)bs * nblk * heads));
  const int kvh = h / (heads / kv_heads);
  const int64_t qpos = (int64_t)qblk * bs + r;

  const int* myids = ids + ((int64_t)(b * heads + h) * nblk + qblk) * cap;
  const int raw = counts[(int64_t)(b * heads + h) * nblk + qblk];
  NaiveCtx ctx{(const bf16*)(q + (((int64_t)b * heads + h) * seq + qpos) * kHeadDim),
               k + ((int64_t)b * kv_heads + kvh) * seq * kHeadDim,
               v + ((int64_t)b * kv_heads + kvh) * seq * kHeadDim,
               myids,
               raw < cap ? raw : cap,
               bs,
               qpos,
               scale};
  bf16* orow = out + (((int64_t)b * heads + h) * seq + qpos) * kHeadDim;

  float m = -INFINITY;
  ctx.each_key([&](const bf16* krow, const bf16* vrow) { m = fmaxf(m, ctx.dot(krow)); });
  if (!(m > -INFINITY)) {
    for (int d = 0; d < kHeadDim; ++d) orow[d] = __float2bfloat16(0.f);
    return;
  }

  float l = 0.f;
  ctx.each_key([&](const bf16* krow, const bf16* vrow) {
    l += exp2f((ctx.dot(krow) - m) * kLog2e);
  });
  const float inv = l > 0.f ? 1.f / l : 0.f;

  for (int d = 0; d < kHeadDim; ++d) {
    float acc = 0.f;
    ctx.each_key([&](const bf16* krow, const bf16* vrow) {
      acc += exp2f((ctx.dot(krow) - m) * kLog2e) * __bfloat162float(vrow[d]);
    });
    orow[d] = __float2bfloat16(acc * inv);
  }
}

// --------------------------------------------------------------------- host

void check_common(const torch::Tensor& t, torch::ScalarType dtype, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.scalar_type() == dtype, name, " has the wrong dtype");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void kernel(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
            const torch::Tensor& block_ids, const torch::Tensor& block_counts,
            torch::Tensor& out) {
  check_common(q, torch::kBFloat16, "q");
  check_common(k, torch::kBFloat16, "k");
  check_common(v, torch::kBFloat16, "v");
  check_common(out, torch::kBFloat16, "out");
  check_common(block_ids, torch::kInt32, "block_ids");
  check_common(block_counts, torch::kInt32, "block_counts");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && out.dim() == 4,
              "q/k/v/out must be rank 4");
  TORCH_CHECK(block_ids.dim() == 4 && block_counts.dim() == 3,
              "block_ids/block_counts have the wrong rank");

  const int64_t batch = q.size(0);
  const int64_t heads = q.size(1);
  const int64_t seq = q.size(2);
  const int64_t dim = q.size(3);
  const int64_t kv_heads = k.size(1);
  const int64_t nblk = block_ids.size(2);
  const int64_t cap = block_ids.size(3);
  TORCH_CHECK(dim == kHeadDim && k.size(3) == kHeadDim && v.size(3) == kHeadDim,
              "head dimension must be ", kHeadDim);
  TORCH_CHECK(k.size(0) == batch && k.size(2) == seq && v.size(0) == batch &&
                  v.size(1) == kv_heads && v.size(2) == seq,
              "k/v shapes must match q");
  TORCH_CHECK(out.sizes() == q.sizes(), "out must match q");
  TORCH_CHECK(block_ids.size(0) == batch && block_ids.size(1) == heads &&
                  block_counts.size(0) == batch && block_counts.size(1) == heads &&
                  block_counts.size(2) == nblk,
              "block_ids/block_counts must match q");
  TORCH_CHECK(heads % kv_heads == 0, "heads must be a multiple of kv heads");
  TORCH_CHECK(seq % nblk == 0, "sequence must divide evenly into blocks");
  TORCH_CHECK(cap >= 1 && cap <= kCapMax, "block capacity must be in [1, ", kCapMax, "]");

  if (batch == 0 || heads == 0 || seq == 0) return;

  c10::cuda::CUDAGuard guard(q.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(q.get_device()).stream();

  const bf16* qp = reinterpret_cast<const bf16*>(q.data_ptr<at::BFloat16>());
  const bf16* kp = reinterpret_cast<const bf16*>(k.data_ptr<at::BFloat16>());
  const bf16* vp = reinterpret_cast<const bf16*>(v.data_ptr<at::BFloat16>());
  bf16* op = reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>());
  const int* idp = block_ids.data_ptr<int>();
  const int* ctp = block_counts.data_ptr<int>();

  if (seq / nblk == kBlockQ) {
    const int smem_bytes = (1 + 4) * kTileBytes;
    static bool configured = false;
    if (!configured) {
      cudaFuncSetAttribute(sparse_attn_fast, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           smem_bytes);
      configured = true;
    }
    const int grid = (int)(batch * heads * nblk);
    sparse_attn_fast<<<grid, kThreads, smem_bytes, stream>>>(
        qp, kp, vp, idp, ctp, op, (int)batch, (int)heads, (int)seq, (int)kv_heads,
        (int)nblk, (int)cap);
  } else {
    const int64_t total = batch * heads * seq;
    const int threads = 128;
    const int grid = (int)((total + threads - 1) / threads);
    sparse_attn_naive<<<grid, threads, 0, stream>>>(
        qp, kp, vp, idp, ctp, op, (int)batch, (int)heads, (int)seq, (int)kv_heads,
        (int)nblk, (int)cap, 1.0f / sqrtf((float)kHeadDim));
  }
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("kernel", &kernel); }

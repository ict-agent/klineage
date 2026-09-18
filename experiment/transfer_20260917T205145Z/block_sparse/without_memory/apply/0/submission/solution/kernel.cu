// Causal block-sparse attention, raw CUDA.
//
//   out[b, h, i, :] = softmax_j(q[b, h, i, :] @ K_j^T / sqrt(D)) @ V_j
//
// where j runs over the key tokens of the blocks listed in block_ids for the
// query block holding i. Keys after the query position are masked, so a listed
// block contributes only when its id is <= the query block id.
//
// One CTA per (batch, query head, query block): 128 query rows, 8 warps with one
// m16 row stripe each, block_dim = 128 keys per step, flash-attention online
// softmax in fp32. Keys are staged through smem with cp.async double buffering.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>

namespace {

using bf16 = __nv_bfloat16;

// Definition axes.
constexpr int kHeadsQ = 32;
constexpr int kHeadsKV = 8;
constexpr int kGroup = kHeadsQ / kHeadsKV;  // query heads per kv head
constexpr int kDim = 128;
constexpr int kBlock = 128;  // tokens per query and key block
constexpr int kCap = 16;     // key blocks selectable per query block

// 8 halves of row padding: a 16B row slice of 8 consecutive rows then covers
// all 32 banks exactly once, so ldmatrix runs conflict free.
constexpr int kPad = 8;
constexpr int kStride = kDim + kPad;
constexpr int kTile = kBlock * kStride;
constexpr int kChunks = kBlock * kDim / 8;  // 16B chunks in one 128x128 tile

constexpr int kThreads = 256;
constexpr float kScaleL2e = 1.4426950408889634f / 11.313708498984761f;
constexpr float kNegInf = -INFINITY;

struct alignas(16) Shared {
  int count;      // tiles that survive compaction
  int ids[kCap];  // compacted block ids, order preserved
  bf16 q[kTile];
  bf16 k[2][kTile];
  bf16 v[2][kTile];
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src));
}

__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;");
}

template <int Pending>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(Pending));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t& a, uint32_t& b, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(a), "=r"(b)
               : "r"(smem_u32(p)));
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t& a, uint32_t& b, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
               : "=r"(a), "=r"(b)
               : "r"(smem_u32(p)));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d,
                                            const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
               : "r"(smem_u32(p)));
}

// D += A(m16k16) * B(k16n8), both bf16, fp32 accumulator.
__device__ __forceinline__ void mma_bf16(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float exp2_fast(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

// ldmatrix.x4 address map of an m16k16 A fragment at row stripe row0.
__device__ __forceinline__ const bf16* a_frag_addr(const bf16* tile, int lane, int row0, int k0) {
  const int row = row0 + ((lane >> 3) & 1) * 8 + (lane & 7);
  const int col = k0 + ((lane >> 4) & 1) * 8;
  return tile + row * kStride + col;
}

// B fragment of q @ k^T: k tile is [key][dim] = [n][k], loaded untransposed.
__device__ __forceinline__ const bf16* kb_frag_addr(const bf16* tile, int lane, int n0, int k0) {
  return tile + (n0 + (lane & 7)) * kStride + k0 + ((lane >> 3) & 1) * 8;
}

// B fragment of p @ v: v tile is [key][dim] = [k][n], loaded transposed.
__device__ __forceinline__ const bf16* vb_frag_addr(const bf16* tile, int lane, int k0, int n0) {
  return tile + (k0 + (lane & 7) + ((lane >> 3) & 1) * 8) * kStride + n0;
}

// 128x128 halves, contiguous in gmem and smem: 8 fully coalesced 16B chunks/thread.
__device__ __forceinline__ void load_tile(bf16* dst, const bf16* src, int tid) {
#pragma unroll
  for (int i = 0; i < kChunks / kThreads; ++i) {
    const int j = tid + i * kThreads;
    const int row = j >> 4;
    const int col = (j & 15) << 3;
    cp_async16(dst + row * kStride + col, src + row * kDim + col);
  }
}

__global__ void __launch_bounds__(kThreads, 1)
block_sparse_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k,
                    const bf16* __restrict__ v, const int* __restrict__ ids,
                    const int* __restrict__ counts, bf16* __restrict__ out, int seq, int nblk) {
  extern __shared__ char raw[];
  Shared& sm = *reinterpret_cast<Shared*>(raw);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int qblk = blockIdx.x;
  const int head = blockIdx.y;
  const int batch = blockIdx.z;

  const int row0 = qblk * kBlock;
  const int slot = (batch * kHeadsQ + head) * nblk + qblk;
  const int* row_ids = ids + slot * kCap;

  // Drop padding (-1) and every block past the diagonal: causality masks them
  // completely, so the main loop only streams tiles that contribute.
  if (warp == 0) {
    const int n = counts[slot];
    const bool held = lane < n;
    const int id = held ? row_ids[lane] : 0;
    const bool keep = held && id >= 0 && id <= qblk;
    const unsigned ballot = __ballot_sync(0xffffffffu, keep);
    if (keep) sm.ids[__popc(ballot & ((1u << lane) - 1u))] = id;
    if (lane == 0) sm.count = __popc(ballot);
  }

  const bf16* qsrc = q + (size_t)(batch * kHeadsQ + head) * seq * kDim + (size_t)row0 * kDim;
  const bf16* ksrc = k + (size_t)(batch * kHeadsKV + head / kGroup) * seq * kDim;
  const bf16* vsrc = v + (size_t)(batch * kHeadsKV + head / kGroup) * seq * kDim;

  load_tile(sm.q, qsrc, tid);
  cp_commit();
  __syncthreads();

  const int ntile = sm.count;
  if (ntile > 0) {
    const size_t off = (size_t)sm.ids[0] * kBlock * kDim;
    load_tile(sm.k[0], ksrc + off, tid);
    load_tile(sm.v[0], vsrc + off, tid);
  }
  cp_commit();
  cp_wait<1>();  // q resident, first tile still in flight
  __syncthreads();

  // Query stripe of this warp: rows [16*warp, 16*warp + 16), all k16 chunks.
  uint32_t qfrag[8][4];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    ldmatrix_x4(qfrag[i][0], qfrag[i][1], qfrag[i][2], qfrag[i][3],
                a_frag_addr(sm.q, lane, warp * 16, 16 * i));
  }

  float acc[16][4];
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
  }
  float rmax[2] = {kNegInf, kNegInf};
  float rsum[2] = {0.f, 0.f};

  for (int it = 0; it < ntile; ++it) {
    const int next = it + 1;
    if (next < ntile) {
      const size_t off = (size_t)sm.ids[next] * kBlock * kDim;
      load_tile(sm.k[next & 1], ksrc + off, tid);
      load_tile(sm.v[next & 1], vsrc + off, tid);
      cp_commit();
      cp_wait<1>();
    } else {
      cp_wait<0>();
    }
    __syncthreads();

    const bf16* kt = sm.k[it & 1];
    const bf16* vt = sm.v[it & 1];
    const bool diag = (sm.ids[it] == qblk);

    // S = Q K^T / sqrt(D), folded into the exp2 domain.
    float sc[16][4];
#pragma unroll
    for (int j = 0; j < 16; j += 2) {
      float c0[4] = {0.f, 0.f, 0.f, 0.f};
      float c1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        uint32_t b0, b1, b2, b3;
        ldmatrix_x2(b0, b1, kb_frag_addr(kt, lane, 8 * j, 16 * i));
        ldmatrix_x2(b2, b3, kb_frag_addr(kt, lane, 8 * j + 8, 16 * i));
        mma_bf16(c0, qfrag[i], b0, b1);
        mma_bf16(c1, qfrag[i], b2, b3);
      }
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        sc[j][e] = c0[e];
        sc[j + 1][e] = c1[e];
      }
    }

    // Causal mask on the diagonal tile only: key j of the tile is masked when
    // j > i. Every other kept tile is fully visible.
    float tile_max[2] = {kNegInf, kNegInf};
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int col = 8 * j + 2 * (lane & 3);
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int row = (lane >> 2) + 8 * e;
        float v0 = sc[j][2 * e];
        float v1 = sc[j][2 * e + 1];
        if (diag) {
          if (col > row) v0 = kNegInf;
          if (col + 1 > row) v1 = kNegInf;
        }
        sc[j][2 * e] = v0;
        sc[j][2 * e + 1] = v1;
        tile_max[e] = fmaxf(tile_max[e], fmaxf(v0, v1));
      }
    }
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      tile_max[e] = fmaxf(tile_max[e], __shfl_xor_sync(0xffffffffu, tile_max[e], 1));
      tile_max[e] = fmaxf(tile_max[e], __shfl_xor_sync(0xffffffffu, tile_max[e], 2));
    }

    // Online softmax: rescale the running accumulator, then emit P = exp2(S - m).
    const float m0 = fmaxf(rmax[0], tile_max[0]);
    const float m1 = fmaxf(rmax[1], tile_max[1]);
    const float a0 = exp2_fast((rmax[0] - m0) * kScaleL2e / kScaleL2e);
    const float a1 = exp2_fast((rmax[1] - m1) * kScaleL2e / kScaleL2e);
    rmax[0] = m0;
    rmax[1] = m1;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      acc[j][0] *= a0;
      acc[j][1] *= a0;
      acc[j][2] *= a1;
      acc[j][3] *= a1;
    }

    float tile_sum[2] = {0.f, 0.f};
#pragma unroll
    for (int j = 0; j < 16; ++j) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const float m = e ? m1 : m0;
        const float p0 = exp2_fast(sc[j][2 * e] * kScaleL2e - m * kScaleL2e);
        const float p1 = exp2_fast(sc[j][2 * e + 1] * kScaleL2e - m * kScaleL2e);
        sc[j][2 * e] = p0;
        sc[j][2 * e + 1] = p1;
        tile_sum[e] += p0 + p1;
      }
    }
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      tile_sum[e] += __shfl_xor_sync(0xffffffffu, tile_sum[e], 1);
      tile_sum[e] += __shfl_xor_sync(0xffffffffu, tile_sum[e], 2);
      rsum[e] = rsum[e] * (e ? a1 : a0) + tile_sum[e];
    }

    // O += P V.
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t pa[4] = {pack_bf16(sc[2 * i][0], sc[2 * i][1]),
                              pack_bf16(sc[2 * i][2], sc[2 * i][3]),
                              pack_bf16(sc[2 * i + 1][0], sc[2 * i + 1][1]),
                              pack_bf16(sc[2 * i + 1][2], sc[2 * i + 1][3])};
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        uint32_t b0, b1;
        ldmatrix_x2_trans(b0, b1, vb_frag_addr(vt, lane, 16 * i, 8 * j));
        mma_bf16(acc[j], pa, b0, b1);
      }
    }
    __syncthreads();
  }

  // Normalize and store the fp32 accumulator as bf16.
  const float inv0 = 1.f / rsum[0];
  const float inv1 = 1.f / rsum[1];
  bf16* dst = out + (size_t)(batch * kHeadsQ + head) * seq * kDim + (size_t)row0 * kDim;
  const int r0 = lane >> 2;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const int col = 8 * j + 2 * (lane & 3);
    *reinterpret_cast<uint32_t*>(dst + (size_t)r0 * kDim + col) =
        pack_bf16(acc[j][0] * inv0, acc[j][1] * inv0);
    *reinterpret_cast<uint32_t*>(dst + (size_t)(r0 + 8) * kDim + col) =
        pack_bf16(acc[j][2] * inv1, acc[j][3] * inv1);
  }
}

void check_cuda(const torch::Tensor& t, torch::ScalarType type) {
  TORCH_CHECK_VALUE(t.is_cuda(), "Expected CUDA tensor");
  TORCH_CHECK_TYPE(t.scalar_type() == type, "Unexpected dtype");
  TORCH_CHECK_VALUE(t.is_contiguous(), "Expected contiguous tensor");
}

void kernel(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
            const torch::Tensor& block_ids, const torch::Tensor& block_counts,
            torch::Tensor& out) {
  check_cuda(q, torch::kBFloat16);
  check_cuda(k, torch::kBFloat16);
  check_cuda(v, torch::kBFloat16);
  check_cuda(block_ids, torch::kInt32);
  check_cuda(block_counts, torch::kInt32);
  check_cuda(out, torch::kBFloat16);

  const int64_t batch = q.size(0);
  const int64_t seq = q.size(2);
  const int64_t nblk = block_ids.size(2);
  const int64_t cap = block_ids.size(3);
  TORCH_CHECK_VALUE(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "Expected rank 4 q/k/v");
  TORCH_CHECK_VALUE(q.size(1) == kHeadsQ && k.size(1) == kHeadsKV, "Unexpected head counts");
  TORCH_CHECK_VALUE(k.size(2) == seq && v.size(2) == seq, "Sequence mismatch");
  TORCH_CHECK_VALUE(q.size(3) == kDim && k.size(3) == kDim, "Unexpected head dim");
  TORCH_CHECK_VALUE(cap == kCap, "Unexpected selected-block capacity");
  TORCH_CHECK_VALUE(block_counts.size(0) == batch && block_counts.size(1) == kHeadsQ &&
                        block_counts.size(2) == nblk,
                    "block_counts shape mismatch");
  TORCH_CHECK_VALUE(block_ids.size(0) == batch && block_ids.size(1) == kHeadsQ,
                    "block_ids shape mismatch");
  TORCH_CHECK_VALUE(seq == nblk * kBlock, "Expected 128 tokens per block");
  TORCH_CHECK_VALUE(out.sizes() == q.sizes(), "Output shape mismatch");

  c10::cuda::CUDAGuard guard(q.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();

  const int smem = (int)sizeof(Shared);
  static bool configured = false;
  if (!configured) {
    const auto status =
        cudaFuncSetAttribute(block_sparse_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    TORCH_CHECK(status == cudaSuccess, "smem opt-in failed: ", cudaGetErrorString(status));
    configured = true;
  }

  const dim3 grid((unsigned)nblk, (unsigned)kHeadsQ, (unsigned)batch);
  block_sparse_kernel<<<grid, kThreads, smem, stream>>>(
      reinterpret_cast<const bf16*>(q.data_ptr()), reinterpret_cast<const bf16*>(k.data_ptr()),
      reinterpret_cast<const bf16*>(v.data_ptr()), block_ids.data_ptr<int>(),
      block_counts.data_ptr<int>(), reinterpret_cast<bf16*>(out.data_ptr()), (int)seq, (int)nblk);
  const auto error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("kernel", &kernel);
}

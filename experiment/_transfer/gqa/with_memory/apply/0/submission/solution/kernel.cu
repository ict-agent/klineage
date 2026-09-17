// Causal grouped-query attention prefill, BF16, head dim 128.
//
// Layout: q [B,S,HQ,D], k/v [B,S,HKV,D], o [B,S,HQ,D], all contiguous BSHD.
// One CTA owns 64 query rows of one query head and streams every visible KV
// tile, so each K/V tile is read once per query tile and Q/O exactly once.
//
//   S = Q(64x128) @ K^T(128x64)   tensor core, fp32 accumulate
//   online softmax on the 64x64 score tile, causal mask on the diagonal tile
//   O += P(64x64) @ V(64x128)     tensor core, fp32 accumulate
//
// Operands use mma.m16n8k16. K and V come from padded shared tiles through
// ldmatrix (8 halves of row padding make every ldmatrix phase bank-conflict
// free); Q is read once into registers straight from global memory in the
// A-fragment layout, so it needs no shared tile. K and V use one cp.async
// double buffer; the next tile is issued right after the barrier that releases
// the buffer it overwrites.

#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

using bf16 = __nv_bfloat16;

constexpr int kDim = 128;      // head dimension
constexpr int kRows = 128;     // query rows per CTA
constexpr int kCols = 64;      // KV rows per tile
constexpr int kRowsPerWarp = 16;
constexpr int kWarps = kRows / kRowsPerWarp;
constexpr int kThreads = kWarps * 32;
constexpr int kRowStride = kDim + 8;  // padded shared row, in halves
constexpr int kStages = 2;
constexpr int kChunk = 8;             // halves per 16B copy chunk
constexpr int kChunksPerRow = kDim / kChunk;
constexpr int kKvChunks = kCols * kChunksPerRow;    // 16B chunks in a KV tile
constexpr int kCopies = kKvChunks / kThreads;       // 16B copies per thread
constexpr int kRowsPerStep = kThreads / kChunksPerRow;

constexpr int kKvElems = kCols * kRowStride;
constexpr int kSmemBytes = kStages * 2 * kKvElems * sizeof(bf16);

constexpr float kLog2e = 1.4426950408889634f;

// The bundle is compiled for every architecture torch ships; the tensor-core
// instructions below exist from sm_80 on, so older passes compile an empty
// kernel instead of failing the build.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define GQA_TENSOR_CORE 1
#else
#define GQA_TENSOR_CORE 0
#endif

// ---------------------------------------------------------------------------
// PTX helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t smem_u32(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
#if GQA_TENSOR_CORE
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)),
               "l"(src));
#endif
}

__device__ __forceinline__ void cp_commit() {
#if GQA_TENSOR_CORE
  asm volatile("cp.async.commit_group;");
#endif
}

__device__ __forceinline__ void cp_wait_all() {
#if GQA_TENSOR_CORE
  asm volatile("cp.async.wait_group 0;");
#endif
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&regs)[4], uint32_t addr) {
#if GQA_TENSOR_CORE
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
      : "r"(addr));
#endif
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&regs)[4],
                                                  uint32_t addr) {
#if GQA_TENSOR_CORE
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
      : "r"(addr));
#endif
}

// D = A(16x16) * B(16x8) + C with bf16 inputs and fp32 accumulate.
__device__ __forceinline__ void mma_16816(float (&d)[4], const uint32_t (&a)[4],
                                          const uint32_t (&b)[2]) {
#if GQA_TENSOR_CORE
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
#if GQA_TENSOR_CORE
  const __nv_bfloat162 pair = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&pair);
#else
  return 0u;
#endif
}

// Row statistics live in the four lanes of a quad; butterfly makes them agree.
__device__ __forceinline__ float quad_sum(float value) {
  value += __shfl_xor_sync(0xffffffffu, value, 1);
  return value + __shfl_xor_sync(0xffffffffu, value, 2);
}

__device__ __forceinline__ float quad_max(float value) {
  value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 1));
  return fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 2));
}

// Stage one [kCols x 128] bf16 tile from a BSHD view into padded shared memory.
// Every thread owns one 16B column and walks kRowsPerStep rows per step: reads are
// fully coalesced 16B chunks and shared writes land in distinct banks.
__device__ __forceinline__ void stage_tile(bf16* dst, const bf16* src,
                                           int row_pitch, int step_rows) {
#pragma unroll
  for (int i = 0; i < kCopies; ++i) {
    cp_async16(dst + i * step_rows * kRowStride,
               src + static_cast<int64_t>(i) * step_rows * row_pitch);
  }
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(kThreads, 1)
gqa_kernel(const bf16* __restrict__ qg, const bf16* __restrict__ kg,
           const bf16* __restrict__ vg, bf16* __restrict__ og, int seq,
           int q_pitch, int kv_pitch, int heads_per_kv, int q_tiles) {
#if !GQA_TENSOR_CORE
  return;
#endif
  extern __shared__ bf16 smem[];
  bf16* k_smem = smem;
  bf16* v_smem = k_smem + kStages * kKvElems;

  // Longest query tiles launch first so the tail CTAs are cheap.
  const int q_tile = q_tiles - 1 - blockIdx.x;
  const int head = blockIdx.y;
  const int batch = blockIdx.z;
  const int kv_head = head / heads_per_kv;
  const int q0 = q_tile * kRows;
  const int n_tiles = (q0 + kRows) / kCols;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int col = (tid % kChunksPerRow) * kChunk;
  const int row_local = tid / kChunksPerRow;
  const int step_rows = kRowsPerStep;
  const int64_t batch_q = static_cast<int64_t>(batch) * seq * q_pitch;
  const int64_t batch_kv = static_cast<int64_t>(batch) * seq * kv_pitch;

  const bf16* k_src = kg + batch_kv + static_cast<int64_t>(kv_head) * kDim +
                      row_local * kv_pitch + col;
  const bf16* v_src = vg + batch_kv + static_cast<int64_t>(kv_head) * kDim +
                      row_local * kv_pitch + col;
  bf16* k_dst = k_smem + row_local * kRowStride + col;
  bf16* v_dst = v_smem + row_local * kRowStride + col;

  // --- KV tile 0 prefetch --------------------------------------------------
  stage_tile(k_dst, k_src, kv_pitch, step_rows);
  stage_tile(v_dst, v_src, kv_pitch, step_rows);
  cp_commit();

  // Fragment addressing. Q and K use the non-transposed map, V the transposed
  // one. In each case the lane group picks which half of the 16 wide block the
  // lane supplies, and the lane picks the row inside it. Q feeds the tensor
  // core straight from global memory: an A fragment is four 4-byte row pieces.
  const int row_in_half = lane & 7;
  const int k_half = ((lane >> 3) & 1) * 8;  // selects the K half of the block
  const int n_half = (lane >> 4) * 8;        // selects the N half of the block
  const int q_lane_row = lane >> 2;
  const int q_lane_col = (lane & 3) * 2;
  const bf16* q_frag_src =
      qg + batch_q +
      static_cast<int64_t>(q0 + warp * kRowsPerWarp + q_lane_row) * q_pitch +
      static_cast<int64_t>(head) * kDim + q_lane_col;
  const uint32_t k_frag_addr =
      smem_u32(k_smem + (n_half + row_in_half) * kRowStride + k_half);
  const uint32_t v_frag_addr =
      smem_u32(v_smem + (k_half + row_in_half) * kRowStride + n_half);

  // The Q tile is resident, so its fragments are read once and kept in registers.
  uint32_t q_frag[8][4];
#pragma unroll
  for (int kt = 0; kt < 8; ++kt) {
    const bf16* q_lo = q_frag_src + kt * 16;
    const bf16* q_hi = q_lo + 8 * q_pitch;
    q_frag[kt][0] = *reinterpret_cast<const uint32_t*>(q_lo);
    q_frag[kt][1] = *reinterpret_cast<const uint32_t*>(q_hi);
    q_frag[kt][2] = *reinterpret_cast<const uint32_t*>(q_lo + 8);
    q_frag[kt][3] = *reinterpret_cast<const uint32_t*>(q_hi + 8);
  }

  cp_wait_all();
  __syncthreads();

  const float scaled = kLog2e * rsqrtf(static_cast<float>(kDim));
  const int q_row0 = warp * kRowsPerWarp + (lane >> 2);
  const int q_row1 = q_row0 + 8;
  const int col_pair = 2 * (lane & 3);

  float row_max[2] = {-CUDART_INF_F, -CUDART_INF_F};
  float row_sum[2] = {0.f, 0.f};
  float out[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) out[i][j] = 0.f;

  for (int t = 0; t < n_tiles; ++t) {
    const int stage = t & 1;
    cp_wait_all();
    __syncthreads();

    // Buffer released: prefetch the next tile while this one is consumed.
    if (t + 1 < n_tiles) {
      const bf16* k_next = k_src + static_cast<int64_t>(t + 1) * kCols * kv_pitch;
      const bf16* v_next = v_src + static_cast<int64_t>(t + 1) * kCols * kv_pitch;
      stage_tile(k_smem + (stage ^ 1) * kKvElems + row_local * kRowStride + col,
                 k_next, kv_pitch, step_rows);
      stage_tile(v_smem + (stage ^ 1) * kKvElems + row_local * kRowStride + col,
                 v_next, kv_pitch, step_rows);
      cp_commit();
    }

    // --- scores: S = Q K^T -------------------------------------------------
    float score[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) score[i][j] = 0.f;

    const uint32_t k_stage_addr = k_frag_addr + stage * kKvElems * 2;
#pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
      uint32_t b[8][2];
#pragma unroll
      for (int np = 0; np < 4; ++np) {
        uint32_t r[4];
        ldmatrix_x4(r, k_stage_addr + (np * 16 * kRowStride + kt * 16) * 2);
        b[np * 2][0] = r[0];
        b[np * 2][1] = r[1];
        b[np * 2 + 1][0] = r[2];
        b[np * 2 + 1][1] = r[3];
      }
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) mma_16816(score[nt], q_frag[kt], b[nt]);
    }

    // Causal mask; only the tiles that straddle this warp's rows hold
    // visible/invisible key pairs, and the test is uniform inside the warp.
    if (t * kCols + kCols > q0 + warp * kRowsPerWarp) {
      const int key0 = t * kCols + col_pair;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int key = key0 + nt * 8;
        if (key > q0 + q_row0) score[nt][0] = -CUDART_INF_F;
        if (key + 1 > q0 + q_row0) score[nt][1] = -CUDART_INF_F;
        if (key > q0 + q_row1) score[nt][2] = -CUDART_INF_F;
        if (key + 1 > q0 + q_row1) score[nt][3] = -CUDART_INF_F;
      }
    }

    // --- online softmax ----------------------------------------------------
    float tile_max[2] = {score[0][0], score[0][2]};
#pragma unroll
    for (int nt = 1; nt < 8; ++nt) {
      tile_max[0] = fmaxf(tile_max[0], fmaxf(score[nt][0], score[nt][1]));
      tile_max[1] = fmaxf(tile_max[1], fmaxf(score[nt][2], score[nt][3]));
    }
    const float new_max0 = fmaxf(row_max[0], quad_max(tile_max[0]));
    const float new_max1 = fmaxf(row_max[1], quad_max(tile_max[1]));
    const float alpha0 = exp2f((row_max[0] - new_max0) * scaled);
    const float alpha1 = exp2f((row_max[1] - new_max1) * scaled);
    const bool max_moved = (new_max0 != row_max[0]) | (new_max1 != row_max[1]);
    row_max[0] = new_max0;
    row_max[1] = new_max1;

    // The rescale only bites when the running row maximum moved. Past the
    // first KV tiles it rarely does, so a warp vote skips 64 multiplies.
    if (__any_sync(0xffffffffu, max_moved)) {
#pragma unroll
      for (int nt = 0; nt < 16; ++nt) {
        out[nt][0] *= alpha0;
        out[nt][1] *= alpha0;
        out[nt][2] *= alpha1;
        out[nt][3] *= alpha1;
      }
    }

    const float ref0 = new_max0 * scaled;
    const float ref1 = new_max1 * scaled;
    float partial0 = 0.f;
    float partial1 = 0.f;
    uint32_t prob[4][4];
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      const float p0 = exp2f(score[kt * 2][0] * scaled - ref0);
      const float p1 = exp2f(score[kt * 2][1] * scaled - ref0);
      const float p2 = exp2f(score[kt * 2][2] * scaled - ref1);
      const float p3 = exp2f(score[kt * 2][3] * scaled - ref1);
      const float p4 = exp2f(score[kt * 2 + 1][0] * scaled - ref0);
      const float p5 = exp2f(score[kt * 2 + 1][1] * scaled - ref0);
      const float p6 = exp2f(score[kt * 2 + 1][2] * scaled - ref1);
      const float p7 = exp2f(score[kt * 2 + 1][3] * scaled - ref1);
      partial0 += (p0 + p1) + (p4 + p5);
      partial1 += (p2 + p3) + (p6 + p7);
      prob[kt][0] = pack_bf16(p0, p1);
      prob[kt][1] = pack_bf16(p2, p3);
      prob[kt][2] = pack_bf16(p4, p5);
      prob[kt][3] = pack_bf16(p6, p7);
    }
    row_sum[0] = row_sum[0] * alpha0 + quad_sum(partial0);
    row_sum[1] = row_sum[1] * alpha1 + quad_sum(partial1);

    // --- O += P V ----------------------------------------------------------
    const uint32_t v_stage_addr = v_frag_addr + stage * kKvElems * 2;
#pragma unroll
    for (int kt = 0; kt < 4; ++kt) {
      uint32_t b[16][2];
#pragma unroll
      for (int np = 0; np < 8; ++np) {
        uint32_t r[4];
        ldmatrix_x4_trans(r, v_stage_addr + (kt * 16 * kRowStride + np * 16) * 2);
        b[np * 2][0] = r[0];
        b[np * 2][1] = r[1];
        b[np * 2 + 1][0] = r[2];
        b[np * 2 + 1][1] = r[3];
      }
#pragma unroll
      for (int nt = 0; nt < 16; ++nt) mma_16816(out[nt], prob[kt], b[nt]);
    }
  }

  // --- normalize and store ------------------------------------------------
  const float inv0 = __frcp_rn(row_sum[0]);
  const float inv1 = __frcp_rn(row_sum[1]);
  bf16* o0 = og + batch_q + static_cast<int64_t>(q0 + q_row0) * q_pitch +
             static_cast<int64_t>(head) * kDim + col_pair;
  bf16* o1 = og + batch_q + static_cast<int64_t>(q0 + q_row1) * q_pitch +
             static_cast<int64_t>(head) * kDim + col_pair;
#pragma unroll
  for (int nt = 0; nt < 16; ++nt) {
    *reinterpret_cast<__nv_bfloat162*>(o0 + nt * 8) =
        __floats2bfloat162_rn(out[nt][0] * inv0, out[nt][1] * inv0);
    *reinterpret_cast<__nv_bfloat162*>(o1 + nt * 8) =
        __floats2bfloat162_rn(out[nt][2] * inv1, out[nt][3] * inv1);
  }
}

// ---------------------------------------------------------------------------
// Host wrapper
// ---------------------------------------------------------------------------

void check_input(const torch::Tensor& tensor, const char* name, int64_t batch,
                 int64_t heads) {
  TORCH_CHECK_VALUE(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK_TYPE(tensor.scalar_type() == torch::kBFloat16, name,
                   " must be bfloat16");
  TORCH_CHECK_VALUE(tensor.dim() == 4, name, " must be rank 4");
  TORCH_CHECK_VALUE(tensor.size(0) == batch, name, " batch mismatch");
  TORCH_CHECK_VALUE(tensor.size(2) == heads, name, " head count mismatch");
  TORCH_CHECK_VALUE(tensor.size(3) == kDim, name, " head dim must be 128");
  TORCH_CHECK_VALUE(tensor.is_contiguous(), name, " must be contiguous BSHD");
}

void kernel(const torch::Tensor& q, const torch::Tensor& k,
            const torch::Tensor& v, torch::Tensor& o) {
  const int64_t batch = q.size(0);
  const int64_t seq = q.size(1);
  const int64_t q_heads = q.size(2);
  const int64_t kv_heads = k.size(2);
  check_input(q, "q", batch, q_heads);
  check_input(k, "k", batch, kv_heads);
  check_input(v, "v", batch, kv_heads);
  TORCH_CHECK_VALUE(o.is_cuda() && o.scalar_type() == torch::kBFloat16,
                    "o must be a CUDA bfloat16 tensor");
  TORCH_CHECK_VALUE(o.is_contiguous() && o.sizes() == q.sizes(),
                    "o must be a contiguous q-shaped tensor");
  TORCH_CHECK_VALUE(k.sizes() == v.sizes(), "k and v must share a shape");
  TORCH_CHECK_VALUE(seq % kRows == 0, "sequence must be a multiple of kRows");
  TORCH_CHECK_VALUE(q_heads % kv_heads == 0 && q_heads > 0,
                    "query heads must be a multiple of KV heads");
  if (seq == 0) return;

  const c10::cuda::CUDAGuard guard(q.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(q.get_device()).stream();

  static bool configured = false;
  if (!configured) {
    const cudaError_t status = cudaFuncSetAttribute(
        gqa_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
    TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
    configured = true;
  }

  const int q_tiles = static_cast<int>(seq / kRows);
  const dim3 grid(q_tiles, static_cast<unsigned>(q_heads),
                  static_cast<unsigned>(batch));
  gqa_kernel<<<grid, kThreads, kSmemBytes, stream>>>(
      reinterpret_cast<const bf16*>(q.data_ptr()),
      reinterpret_cast<const bf16*>(k.data_ptr()),
      reinterpret_cast<const bf16*>(v.data_ptr()),
      reinterpret_cast<bf16*>(o.data_ptr()), static_cast<int>(seq),
      static_cast<int>(q_heads * kDim), static_cast<int>(kv_heads * kDim),
      static_cast<int>(q_heads / kv_heads), q_tiles);
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("kernel", &kernel);
}

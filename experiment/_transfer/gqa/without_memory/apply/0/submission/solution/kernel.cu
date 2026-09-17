// Causal grouped-query attention prefill, flash-attention-2 style, raw CUDA.
//
//   per CTA: 256 query rows of one (batch, q head); loop over key blocks 0..m
//   K/V tiles are double buffered through shared memory with cp.async.
//   S = Q K^T and O += P V run on the bf16 tensor cores with fp32 accumulators.
//
// Each warp owns 32 query rows (two m16 tiles) so that a single K/V fragment
// load feeds four mma instead of two; the shared memory pipe is the limiting
// resource, and this halves its traffic per query row.
#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kHeadDim = 128;
constexpr int kBlockM = 256;                       // query rows per CTA
constexpr int kBlockN = 64;                        // key rows per iteration
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kRowsPerWarp = kBlockM / kWarps;     // 32 = two m16 tiles
constexpr int kMTiles = kRowsPerWarp / 16;
constexpr int kPadHalves = 8;                      // 16-byte aligned rows, no bank conflicts
constexpr int kRowStride = kHeadDim + kPadHalves;
constexpr int kChunksPerRow = kHeadDim / 8;        // 16-byte chunks per smem row
constexpr int kRowsPerCopy = kThreads / kChunksPerRow;  // rows staged per copy pass
constexpr int kCopies = kBlockN / kRowsPerCopy;
constexpr int kStages = 2;
constexpr int kNTiles = kBlockN / 8;               // S column tiles
constexpr int kDTiles = kHeadDim / 8;              // O column tiles
constexpr int kKSteps = kHeadDim / 16;             // k steps of Q K^T
constexpr int kKVSteps = kBlockN / 16;             // k steps of P V
constexpr float kLog2e = 1.4426950408889634f;
constexpr int kSmemHalves = kBlockM * kRowStride + kStages * 2 * kBlockN * kRowStride;
constexpr int kSmemBytes = kSmemHalves * 2;

// ---------------------------------------------------------------------------
// Hardware primitives (Ampere and later). Targets below sm_80 are still built
// so the extension loads everywhere, but they never reach this code.
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
#define KLINEAGE_TC 0
#else
#define KLINEAGE_TC 1
#endif

__device__ __forceinline__ uint32_t smem_addr(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
#if KLINEAGE_TC
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
#endif
}

__device__ __forceinline__ void cp_commit() {
#if KLINEAGE_TC
  asm volatile("cp.async.commit_group;");
#endif
}

template <int N>
__device__ __forceinline__ void cp_wait() {
#if KLINEAGE_TC
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
#endif
}

// Load a 16x16 bf16 tile (row major source) as one m16k16 A fragment.
__device__ __forceinline__ void ldsm_a(uint32_t (&r)[4], uint32_t addr) {
#if KLINEAGE_TC
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
#endif
}

// Load two 8-wide n tiles of a row major (k x n) source as one B fragment pair.
__device__ __forceinline__ void ldsm_b(uint32_t (&r)[4], uint32_t addr) {
#if KLINEAGE_TC
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
#endif
}

// Transposed load: row major (k x n) source becomes the (k x n) B operand.
__device__ __forceinline__ void ldsm_b_trans(uint32_t (&r)[4], uint32_t addr) {
#if KLINEAGE_TC
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
#endif
}

__device__ __forceinline__ void mma_16816(float (&c)[4], const uint32_t (&a)[4],
                                          uint32_t b0, uint32_t b1) {
#if KLINEAGE_TC
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

// ex2.approx carries a 2^-22 relative error, far inside the fp32 budget here.
__device__ __forceinline__ float fast_exp2(float x) {
  float y;
#if KLINEAGE_TC
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
#else
  y = exp2f(x);
#endif
  return y;
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 packed = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&packed);
}

// ---------------------------------------------------------------------------
// Kernel.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kThreads, 1)
gqa_prefill_kernel(const __nv_bfloat16* __restrict__ q_ptr,
                   const __nv_bfloat16* __restrict__ k_ptr,
                   const __nv_bfloat16* __restrict__ v_ptr,
                   __nv_bfloat16* __restrict__ o_ptr, int seq_len,
                   int q_row_stride, int kv_row_stride, int group, float scale) {
  extern __shared__ char smem[];
  auto* sq = reinterpret_cast<__nv_bfloat16*>(smem);
  auto* sk = sq + kBlockM * kRowStride;
  auto* sv = sk + kStages * kBlockN * kRowStride;
  constexpr int kTileHalves = kBlockN * kRowStride;

  const int m_block = gridDim.x - 1 - blockIdx.x;
  const int rows0 = m_block * kBlockM;
  const int q_head = blockIdx.y;
  const int kv_head = q_head / group;
  const __nv_bfloat16* qg =
      q_ptr + ((int64_t)blockIdx.z * seq_len + rows0) * q_row_stride + q_head * kHeadDim;
  const __nv_bfloat16* kg =
      k_ptr + (int64_t)blockIdx.z * seq_len * kv_row_stride + kv_head * kHeadDim;
  const __nv_bfloat16* vg =
      v_ptr + (int64_t)blockIdx.z * seq_len * kv_row_stride + kv_head * kHeadDim;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int group_id = lane >> 2;  // row inside the 8-row half of a fragment
  const int tig = lane & 3;        // column pair inside a fragment

  // Q tile -> padded smem (one 16-byte chunk per (row, chunk) pair).
  for (int i = threadIdx.x; i < kBlockM * kChunksPerRow; i += kThreads) {
    const int r = i / kChunksPerRow;
    const int c = i % kChunksPerRow;
    *reinterpret_cast<uint4*>(sq + r * kRowStride + c * 8) =
        *reinterpret_cast<const uint4*>(qg + (int64_t)r * q_row_stride + c * 8);
  }
  __syncthreads();

  // One ldmatrix.x4 covers two 8-key n tiles: bit 4 picks the tile, bit 3 the
  // 8-wide half of the reduction axis.
  const int k_row = (lane & 7) + 8 * ((lane >> 4) & 1);
  const int k_col = 8 * ((lane >> 3) & 1);
  // Transposed load for V: the two matrices of an n tile stack along the
  // reduction (key) axis instead, bit 3 picks the tile and bit 4 the dims.
  const int v_row = (lane & 7) + 8 * ((lane >> 3) & 1);
  const int v_col = (lane >> 4) * 8;
  // Q fragment rows of this warp: two m16 tiles stacked.
  const int q_row = warp * kRowsPerWarp + (lane & 15);
  const int q_col = (lane >> 4) * 8;

  const int n_kblocks = (rows0 + kBlockM + kBlockN - 1) / kBlockN;
  const float scale_log2e = scale * kLog2e;

  // K/V staging. Every thread owns four 16-byte chunks per tile, so the source
  // and destination offsets are loop invariant; only the tile base moves.
  const int lane_row = threadIdx.x / kChunksPerRow;
  const int lane_halves = (threadIdx.x % kChunksPerRow) * 8;
  const int64_t lane_off = (int64_t)lane_row * kv_row_stride + lane_halves;
  const int64_t row_step = (int64_t)kRowsPerCopy * kv_row_stride;
  const uint32_t k_smem0 = smem_addr(sk + lane_row * kRowStride + lane_halves);
  const uint32_t v_smem0 = smem_addr(sv + lane_row * kRowStride + lane_halves);
  constexpr int kChunkStride = kRowsPerCopy * kRowStride * 2;
  constexpr int kBufStride = kTileHalves * 2;

  auto load_kv = [&](const __nv_bfloat16* ksrc, const __nv_bfloat16* vsrc, int buf) {
    const uint32_t kdst = k_smem0 + buf * kBufStride;
    const uint32_t vdst = v_smem0 + buf * kBufStride;
#pragma unroll
    for (int j = 0; j < kCopies; ++j) {
      cp_async16(kdst + j * kChunkStride, ksrc + j * row_step);
      cp_async16(vdst + j * kChunkStride, vsrc + j * row_step);
    }
  };

  float acc_o[kMTiles][kDTiles][4];
#pragma unroll
  for (int m = 0; m < kMTiles; ++m) {
#pragma unroll
    for (int i = 0; i < kDTiles; ++i) {
      acc_o[m][i][0] = 0.f;
      acc_o[m][i][1] = 0.f;
      acc_o[m][i][2] = 0.f;
      acc_o[m][i][3] = 0.f;
    }
  }
  float row_max[kMTiles][2] = {{-1e30f, -1e30f}, {-1e30f, -1e30f}};
  float row_sum[kMTiles][2] = {{0.f, 0.f}, {0.f, 0.f}};

  const __nv_bfloat16* k_tile = kg + lane_off;
  const __nv_bfloat16* v_tile = vg + lane_off;
  const int64_t tile_step = (int64_t)kBlockN * kv_row_stride;
  load_kv(k_tile, v_tile, 0);
  cp_commit();

  for (int kb = 0; kb < n_kblocks; ++kb) {
    const int buf = kb & 1;
    if (kb + 1 < n_kblocks) {
      k_tile += tile_step;
      v_tile += tile_step;
      load_kv(k_tile, v_tile, buf ^ 1);
      cp_commit();
      cp_wait<1>();
    } else {
      cp_wait<0>();
    }
    __syncthreads();

    const __nv_bfloat16* skb = sk + buf * kTileHalves;
    const __nv_bfloat16* svb = sv + buf * kTileHalves;

    // --- S = Q K^T ---------------------------------------------------------
    float acc_s[kMTiles][kNTiles][4];
#pragma unroll
    for (int m = 0; m < kMTiles; ++m) {
#pragma unroll
      for (int j = 0; j < kNTiles; ++j) {
        acc_s[m][j][0] = 0.f;
        acc_s[m][j][1] = 0.f;
        acc_s[m][j][2] = 0.f;
        acc_s[m][j][3] = 0.f;
      }
    }
#pragma unroll
    for (int ks = 0; ks < kKSteps; ++ks) {
      uint32_t qfrag[kMTiles][4];
#pragma unroll
      for (int m = 0; m < kMTiles; ++m) {
        ldsm_a(qfrag[m], smem_addr(sq + (q_row + m * 16) * kRowStride + ks * 16 + q_col));
      }
#pragma unroll
      for (int np = 0; np < kNTiles / 2; ++np) {
        uint32_t kfrag[4];
        ldsm_b(kfrag, smem_addr(skb + (16 * np + k_row) * kRowStride + ks * 16 + k_col));
#pragma unroll
        for (int m = 0; m < kMTiles; ++m) {
          mma_16816(acc_s[m][2 * np], qfrag[m], kfrag[0], kfrag[1]);
          mma_16816(acc_s[m][2 * np + 1], qfrag[m], kfrag[2], kfrag[3]);
        }
      }
    }

    // --- causal mask on the diagonal block ---------------------------------
    // Any block whose last key passes the first query row straddles the diagonal.
    if ((kb + 1) * kBlockN > rows0) {
#pragma unroll
      for (int m = 0; m < kMTiles; ++m) {
        const int row_a = rows0 + warp * kRowsPerWarp + m * 16 + group_id;
        const int row_b = row_a + 8;
#pragma unroll
        for (int j = 0; j < kNTiles; ++j) {
          const int col = kb * kBlockN + j * 8 + 2 * tig;
          if (col > row_a) acc_s[m][j][0] = -INFINITY;
          if (col + 1 > row_a) acc_s[m][j][1] = -INFINITY;
          if (col > row_b) acc_s[m][j][2] = -INFINITY;
          if (col + 1 > row_b) acc_s[m][j][3] = -INFINITY;
        }
      }
    }

    // --- online softmax ----------------------------------------------------
#pragma unroll
    for (int m = 0; m < kMTiles; ++m) {
      float tmax_a = -INFINITY;
      float tmax_b = -INFINITY;
#pragma unroll
      for (int j = 0; j < kNTiles; ++j) {
        tmax_a = fmaxf(tmax_a, fmaxf(acc_s[m][j][0], acc_s[m][j][1]));
        tmax_b = fmaxf(tmax_b, fmaxf(acc_s[m][j][2], acc_s[m][j][3]));
      }
#pragma unroll
      for (int s = 1; s < 4; s <<= 1) {
        tmax_a = fmaxf(tmax_a, __shfl_xor_sync(0xffffffffu, tmax_a, s));
        tmax_b = fmaxf(tmax_b, __shfl_xor_sync(0xffffffffu, tmax_b, s));
      }

      const float m_new_a = fmaxf(row_max[m][0], tmax_a * scale);
      const float m_new_b = fmaxf(row_max[m][1], tmax_b * scale);
      const float alpha_a = fast_exp2((row_max[m][0] - m_new_a) * kLog2e);
      const float alpha_b = fast_exp2((row_max[m][1] - m_new_b) * kLog2e);
      row_max[m][0] = m_new_a;
      row_max[m][1] = m_new_b;

      float tsum_a = 0.f;
      float tsum_b = 0.f;
      const float shift_a = -m_new_a * kLog2e;
      const float shift_b = -m_new_b * kLog2e;
#pragma unroll
      for (int j = 0; j < kNTiles; ++j) {
        acc_s[m][j][0] = fast_exp2(fmaf(acc_s[m][j][0], scale_log2e, shift_a));
        acc_s[m][j][1] = fast_exp2(fmaf(acc_s[m][j][1], scale_log2e, shift_a));
        acc_s[m][j][2] = fast_exp2(fmaf(acc_s[m][j][2], scale_log2e, shift_b));
        acc_s[m][j][3] = fast_exp2(fmaf(acc_s[m][j][3], scale_log2e, shift_b));
        tsum_a += acc_s[m][j][0] + acc_s[m][j][1];
        tsum_b += acc_s[m][j][2] + acc_s[m][j][3];
      }
#pragma unroll
      for (int s = 1; s < 4; s <<= 1) {
        tsum_a += __shfl_xor_sync(0xffffffffu, tsum_a, s);
        tsum_b += __shfl_xor_sync(0xffffffffu, tsum_b, s);
      }
      row_sum[m][0] = row_sum[m][0] * alpha_a + tsum_a;
      row_sum[m][1] = row_sum[m][1] * alpha_b + tsum_b;

      // alpha is exactly 1 whenever the row max held, which is almost always.
#pragma unroll
      for (int i = 0; i < kDTiles; ++i) {
        acc_o[m][i][0] *= alpha_a;
        acc_o[m][i][1] *= alpha_a;
        acc_o[m][i][2] *= alpha_b;
        acc_o[m][i][3] *= alpha_b;
      }
    }

    // --- P V ---------------------------------------------------------------
#pragma unroll
    for (int kk = 0; kk < kKVSteps; ++kk) {
      uint32_t pfrag[kMTiles][4];
#pragma unroll
      for (int m = 0; m < kMTiles; ++m) {
        pfrag[m][0] = pack_bf16(acc_s[m][2 * kk][0], acc_s[m][2 * kk][1]);
        pfrag[m][1] = pack_bf16(acc_s[m][2 * kk][2], acc_s[m][2 * kk][3]);
        pfrag[m][2] = pack_bf16(acc_s[m][2 * kk + 1][0], acc_s[m][2 * kk + 1][1]);
        pfrag[m][3] = pack_bf16(acc_s[m][2 * kk + 1][2], acc_s[m][2 * kk + 1][3]);
      }
#pragma unroll
      for (int np = 0; np < kDTiles / 2; ++np) {
        uint32_t vfrag[4];
        ldsm_b_trans(vfrag, smem_addr(svb + (kk * 16 + v_row) * kRowStride + 16 * np + v_col));
#pragma unroll
        for (int m = 0; m < kMTiles; ++m) {
          mma_16816(acc_o[m][2 * np], pfrag[m], vfrag[0], vfrag[1]);
          mma_16816(acc_o[m][2 * np + 1], pfrag[m], vfrag[2], vfrag[3]);
        }
      }
    }
    __syncthreads();
  }

  // --- epilogue: normalise, round to bf16, store through smem -------------
#pragma unroll
  for (int m = 0; m < kMTiles; ++m) {
    const float inv_a = 1.f / row_sum[m][0];
    const float inv_b = 1.f / row_sum[m][1];
#pragma unroll
    for (int i = 0; i < kDTiles; ++i) {
      __nv_bfloat16* dst = sq + (warp * kRowsPerWarp + m * 16 + group_id) * kRowStride +
                           i * 8 + 2 * tig;
      *reinterpret_cast<uint32_t*>(dst) =
          pack_bf16(acc_o[m][i][0] * inv_a, acc_o[m][i][1] * inv_a);
      *reinterpret_cast<uint32_t*>(dst + 8 * kRowStride) =
          pack_bf16(acc_o[m][i][2] * inv_b, acc_o[m][i][3] * inv_b);
    }
  }
  __syncthreads();

  __nv_bfloat16* og =
      o_ptr + ((int64_t)blockIdx.z * seq_len + rows0) * q_row_stride + q_head * kHeadDim;
  for (int i = threadIdx.x; i < kBlockM * kChunksPerRow; i += kThreads) {
    const int r = i / kChunksPerRow;
    const int c = i % kChunksPerRow;
    *reinterpret_cast<uint4*>(og + (int64_t)r * q_row_stride + c * 8) =
        *reinterpret_cast<const uint4*>(sq + r * kRowStride + c * 8);
  }
}

// ---------------------------------------------------------------------------
// Host wrapper.
// ---------------------------------------------------------------------------
void check_bshd(const torch::Tensor& t, int64_t heads, int64_t head_dim) {
  TORCH_CHECK_VALUE(t.is_cuda(), "tensor must be CUDA");
  TORCH_CHECK_TYPE(t.scalar_type() == torch::kBFloat16, "tensor must be bfloat16");
  TORCH_CHECK_VALUE(t.dim() == 4, "tensor must be rank 4");
  TORCH_CHECK_VALUE(t.is_contiguous(), "tensor must be contiguous BSHD");
  TORCH_CHECK_VALUE(t.size(3) == head_dim, "head dim mismatch");
  TORCH_CHECK_VALUE(t.size(2) == heads, "head count mismatch");
}

void kernel(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
            const torch::Tensor& o) {
  const int64_t batch = q.size(0);
  const int64_t seq_len = q.size(1);
  const int64_t q_heads = q.size(2);
  const int64_t kv_heads = k.size(2);

  check_bshd(q, q_heads, kHeadDim);
  check_bshd(k, kv_heads, kHeadDim);
  check_bshd(v, kv_heads, kHeadDim);
  check_bshd(o, q_heads, kHeadDim);
  TORCH_CHECK_VALUE(k.sizes() == v.sizes(), "k/v shape mismatch");
  TORCH_CHECK_VALUE(o.sizes() == q.sizes(), "q/o shape mismatch");
  TORCH_CHECK_VALUE(q_heads % kv_heads == 0, "query heads must be a multiple of kv heads");
  TORCH_CHECK_VALUE(seq_len % kBlockM == 0, "sequence length must be a multiple of ", kBlockM);
  TORCH_CHECK_VALUE(q.device() == k.device() && q.device() == v.device() &&
                        q.device() == o.device(),
                    "device mismatch");
  if (batch == 0 || seq_len == 0) return;

  c10::cuda::CUDAGuard guard(q.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();

  static const bool configured = [] {
    cudaFuncSetAttribute(gqa_prefill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kSmemBytes);
    return true;
  }();
  TORCH_CHECK(configured, "shared memory configuration failed");

  const dim3 grid(seq_len / kBlockM, q_heads, batch);
  const float scale = 1.0f / sqrtf(static_cast<float>(kHeadDim));
  gqa_prefill_kernel<<<grid, kThreads, kSmemBytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(o.data_ptr<at::BFloat16>()), (int)seq_len,
      (int)q.stride(1), (int)k.stride(1), (int)(q_heads / kv_heads), scale);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "gqa kernel launch failed: ", cudaGetErrorString(err));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("kernel", &kernel);
}

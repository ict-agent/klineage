// Packed-NHD noncausal FMHA for sm90a (H100).
//
// One CTA covers 256 query rows (four 128-thread warpgroups, 64 rows each) of one
// sequence and one head; key/value tiles are 64 rows.  Both matrix products use
// wgmma from shared memory (QK) and from registers (PV), with online softmax kept
// in the log2 domain.
//
//   QK: D = Q * K^T   A = Q (smem, K-major)   B = K (smem, K-major)
//   PV: O = P * V     A = P (registers)       B = V (smem, MN-major)
//
// Operands live in the canonical (unswizzled) shared-memory layout: a tile of R
// rows of 128 halfs is stored as [row/8][16-byte chunk][row%8], so each 8x16-byte
// core matrix is contiguous and the two descriptor strides are plain offsets.
//
//   offset(row, chunk) = (row/8)*1024 + chunk*64 + (row%8)*8      (halfs)
//
#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kDim = 128;    // head dimension (fixed)
constexpr int kN = 64;       // key/value rows per tile
constexpr int kM = 256;      // query rows per CTA
constexpr int kStages = 2;   // key/value pipeline stages
constexpr int kThreads = 512;
constexpr int kRowStride = kDim;                // halfs per token row
constexpr int kTileHalfs = kN * kDim;           // halfs in one key or value tile
constexpr int kGroupHalfs = (kDim / 8) * 8 * 8; // halfs per 8-row group
constexpr int kRowsPerWg = 64;                  // query rows per warpgroup

constexpr int kQkRegs = 32;   // m64n64 accumulator, fp32
constexpr int kOutRegs = 64;  // m64n128 accumulator, fp32
constexpr int kProbRegs = 16; // m64k64 A fragment, packed fp16

// Descriptor fields: byte offsets expressed in 16-byte units.
constexpr uint32_t kKmajLbo = 8;     // k-major: 128 B between k chunks
constexpr uint32_t kKmajSbo = 128;   // k-major: 2048 B between 8-row groups
constexpr uint32_t kKmajStep = 16;   // 256 B per k16 step (two chunks)
constexpr uint32_t kMnmajLbo = 128;  // m/n-major: 2048 B between 8-row groups
constexpr uint32_t kMnmajSbo = 8;    // m/n-major: 128 B between n chunks
constexpr uint32_t kMnmajStep = 256; // 4096 B per k16 step (two key groups)

// log2(e)/sqrt(128) and 1/sqrt(128) in fp32.
constexpr float kLog2Scale = 0.127517424f;

// ---------------------------------------------------------------- descriptors

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Matrix descriptor for the unswizzled canonical layout.
__device__ __forceinline__ uint64_t desc(const void* p, uint32_t lbo16, uint32_t sbo16) {
  return static_cast<uint64_t>((smem_u32(p) >> 4) & 0x3FFF) |
         (static_cast<uint64_t>(lbo16) << 16) | (static_cast<uint64_t>(sbo16) << 32);
}

__device__ __forceinline__ __half* tile_slot(__half* tile, int row, int chunk) {
  return tile + (row >> 3) * kGroupHalfs + chunk * 64 + (row & 7) * 8;
}

// ------------------------------------------------------------------- wgmma

#define WGMMA_ACC32                                                          \
  "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"                  \
  "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"

#define WGMMA_QK_DEF(NAME, ACC)                                              \
  __device__ __forceinline__ void NAME(float (&d)[kQkRegs], uint64_t a,      \
                                       uint64_t b) {                         \
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 " WGMMA_ACC32 \
                 ", %32, %33, " #ACC ", 1, 1, 0, 0;\n"                       \
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), \
                   "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), \
                   "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), \
                   "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), \
                   "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), \
                   "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) \
                 : "l"(a), "l"(b)                                            \
                 : "memory");                                                \
  }

WGMMA_QK_DEF(wgmma_qk_clear, 0)
WGMMA_QK_DEF(wgmma_qk_acc, 1)

// PV: A in registers (packed fp16, k-major fragment), B = V as an MN-major operand.
__device__ __forceinline__ void wgmma_pv(float (&d)[kOutRegs], uint32_t a0, uint32_t a1,
                                         uint32_t a2, uint32_t a3, uint64_t b) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
      "{%64,%65,%66,%67}, %68, 1, 1, 1, 1;\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]),
        "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),
        "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
        "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]),
        "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]),
        "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),
        "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(b)
      : "memory");
}

__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_wait_all() {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

// -------------------------------------------------------------- async copies

__device__ __forceinline__ void cp_async16(__half* dst, const __half* src) {
  const uint32_t d = smem_u32(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
}

__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_wait_all() {
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

__device__ __forceinline__ void async_fence() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// ---------------------------------------------------------------- kernel body

// Load one 64-row key tile and the matching value tile into the given stages.
// Rows past the end of the sequence are clamped: their columns are masked out of
// the softmax, so the duplicated data never contributes.
__device__ __forceinline__ void load_kv(const __half* gk, const __half* gv, __half* dk,
                                        __half* dv, int row0, int last_row,
                                        size_t row_stride, int tid) {
  const int ck0 = tid;
#pragma unroll
  for (int i = 0; i < (kTileHalfs / 8) / kThreads; ++i) {
    const int ck = ck0 + i * kThreads;
    const int row = ck >> 4, chunk = ck & 15;
    const int grow = min(row0 + row, last_row);
    cp_async16(tile_slot(dk, row, chunk), gk + static_cast<size_t>(grow) * row_stride + chunk * 8);
    cp_async16(tile_slot(dv, row, chunk), gv + static_cast<size_t>(grow) * row_stride + chunk * 8);
  }
}

__global__ void __launch_bounds__(kThreads, 1)
    fmha_kernel(const __half* __restrict__ gq, const __half* __restrict__ gk,
                const __half* __restrict__ gv, const int* __restrict__ cuq,
                const int* __restrict__ cuk, __half* __restrict__ go, int heads,
                int num_seq) {
  extern __shared__ __align__(1024) __half smem[];
  __half* sq = smem;
  __half* sk = sq + kM * kDim;
  __half* sv = sk + kStages * kTileHalfs;

  // Map this CTA onto one (sequence, query tile) pair.
  int q_start = 0, q_end = 0, k_start = 0, k_end = 0;
  int tile_in_seq = 0;
  {
    const int block = blockIdx.x;
    int acc = 0;
    for (int s = 0; s < num_seq; ++s) {
      const int qs = cuq[s], qe = cuq[s + 1];
      const int tiles = (qe - qs + kM - 1) / kM;
      if (block < acc + tiles) {
        tile_in_seq = block - acc;
        q_start = qs;
        q_end = qe;
        k_start = cuk[s];
        k_end = cuk[s + 1];
        break;
      }
      acc += tiles;
    }
  }
  if (q_start >= q_end) return;  // padding block

  const int tid = threadIdx.x;
  const int wg = tid >> 7;
  const int lane = tid & 127;
  const int warp = lane >> 5;
  const int sub = lane & 31;
  const int row = warp * 16 + (sub >> 2);  // 0..15 within the warpgroup tile
  const int col = (sub & 3) * 2;           // 0,2,4,6

  const int head = blockIdx.y;
  const size_t row_stride = static_cast<size_t>(heads) * kDim;
  const int q0 = q_start + tile_in_seq * kM;
  const int q_rows = min(kM, q_end - q0);
  const int kv_len = k_end - k_start;
  const int kv_tiles = (kv_len + kN - 1) / kN;

  // Query tile: 2048 16-byte chunks, 8 per thread.
  {
    const __half* src = gq + static_cast<size_t>(head) * kDim;
#pragma unroll
    for (int i = 0; i < (kM * kDim / 8) / kThreads; ++i) {
      const int ck = tid + i * kThreads;
      const int r = ck >> 4, c = ck & 15;
      const int gr = min(q0 + r, q_end - 1);
      cp_async16(tile_slot(sq, r, c), src + static_cast<size_t>(gr) * row_stride + c * 8);
    }
  }
  load_kv(gk + head * kDim, gv + head * kDim, sk, sv, k_start, k_end - 1, row_stride, tid);
  cp_commit();

  const uint64_t q_desc = desc(sq + wg * (kRowsPerWg / 8) * kGroupHalfs, kKmajLbo, kKmajSbo);

  float out[kOutRegs];
#pragma unroll
  for (int i = 0; i < kOutRegs; ++i) out[i] = 0.f;
  float row_max[2] = {-INFINITY, -INFINITY};
  float row_sum[2] = {0.f, 0.f};

  for (int t = 0; t < kv_tiles; ++t) {
    const int stage = t & (kStages - 1);
    __half* k_tile = sk + stage * kTileHalfs;
    __half* v_tile = sv + stage * kTileHalfs;

    cp_wait_all();
    __syncthreads();
    async_fence();

    // S = Q * K^T
    float s[kQkRegs];
    const uint64_t k_desc = desc(k_tile, kKmajLbo, kKmajSbo);
    wgmma_qk_clear(s, q_desc, k_desc);
#pragma unroll
    for (int k = 1; k < kDim / 16; ++k) wgmma_qk_acc(s, q_desc + k * kKmajStep, k_desc + k * kKmajStep);
    wgmma_commit();
    wgmma_wait_all();

    // Mask key columns past the end of the sequence (last tile only).
    if (t + 1 == kv_tiles && kv_len < kv_tiles * kN) {
      const int valid = kv_len - t * kN;
#pragma unroll
      for (int i = 0; i < kQkRegs; ++i) {
        if ((i >> 2) * 8 + col + (i & 1) >= valid) s[i] = -INFINITY;
      }
    }

    // Release the previous stage (both warpgroups have finished their wgmma reads).
    __syncthreads();
    if (t + 1 < kv_tiles) {
      const int stage_next = (t + 1) & (kStages - 1);
      load_kv(gk + head * kDim, gv + head * kDim, sk + stage_next * kTileHalfs,
              sv + stage_next * kTileHalfs, k_start + (t + 1) * kN, k_end - 1, row_stride,
              tid);
    }
    cp_commit();

    // Online softmax: row max, exponentials, row sum.
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int i = 0; i < kQkRegs; i += 4) {
      mx0 = fmaxf(mx0, fmaxf(s[i], s[i + 1]));
      mx1 = fmaxf(mx1, fmaxf(s[i + 2], s[i + 3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));

    const float new_max0 = mx0 * kLog2Scale;
    const float new_max1 = mx1 * kLog2Scale;
    const float alpha0 = exp2f(row_max[0] - new_max0);
    const float alpha1 = exp2f(row_max[1] - new_max1);
    row_max[0] = new_max0;
    row_max[1] = new_max1;

    float p[kQkRegs];
#pragma unroll
    for (int i = 0; i < kQkRegs; ++i) {
      const float bias = (i & 2) ? new_max1 : new_max0;
      p[i] = exp2f(fmaf(s[i], kLog2Scale, -bias));
    }
    float part0 = 0.f, part1 = 0.f;
#pragma unroll
    for (int i = 0; i < kQkRegs; i += 4) {
      part0 += p[i] + p[i + 1];
      part1 += p[i + 2] + p[i + 3];
    }
    part0 += __shfl_xor_sync(0xffffffffu, part0, 1);
    part0 += __shfl_xor_sync(0xffffffffu, part0, 2);
    part1 += __shfl_xor_sync(0xffffffffu, part1, 1);
    part1 += __shfl_xor_sync(0xffffffffu, part1, 2);
    row_sum[0] = row_sum[0] * alpha0 + part0;
    row_sum[1] = row_sum[1] * alpha1 + part1;

    // Rescale the accumulator when this tile moved any row maximum.
    if (__any_sync(0xffffffffu, alpha0 != 1.f || alpha1 != 1.f)) {
#pragma unroll
      for (int i = 0; i < kOutRegs; ++i) out[i] *= (i & 2) ? alpha1 : alpha0;
    }

    // Pack the probabilities into the k-major fp16 A fragment.
    uint32_t pf[kProbRegs];
#pragma unroll
    for (int j = 0; j < kQkRegs / 8; ++j) {
      pf[4 * j + 0] = __half_as_ushort(__float2half_rn(p[8 * j + 0])) |
                      (uint32_t(__half_as_ushort(__float2half_rn(p[8 * j + 1]))) << 16);
      pf[4 * j + 1] = __half_as_ushort(__float2half_rn(p[8 * j + 2])) |
                      (uint32_t(__half_as_ushort(__float2half_rn(p[8 * j + 3]))) << 16);
      pf[4 * j + 2] = __half_as_ushort(__float2half_rn(p[8 * j + 4])) |
                      (uint32_t(__half_as_ushort(__float2half_rn(p[8 * j + 5]))) << 16);
      pf[4 * j + 3] = __half_as_ushort(__float2half_rn(p[8 * j + 6])) |
                      (uint32_t(__half_as_ushort(__float2half_rn(p[8 * j + 7]))) << 16);
    }

    // O += P * V
    const uint64_t v_desc = desc(v_tile, kMnmajLbo, kMnmajSbo);
#pragma unroll
    for (int k = 0; k < kN / 16; ++k) {
      wgmma_pv(out, pf[4 * k + 0], pf[4 * k + 1], pf[4 * k + 2], pf[4 * k + 3],
               v_desc + k * kMnmajStep);
    }
    wgmma_commit();
  }
  wgmma_wait_all();

  // Epilogue: normalise and store the two rows this thread owns.
  const float inv0 = 1.f / row_sum[0];
  const float inv1 = 1.f / row_sum[1];
  const int gl_row = q0 + wg * kRowsPerWg + row;
  const int rows_valid = q0 + q_rows;
  __half* dst = go + static_cast<size_t>(head) * kDim;
#pragma unroll
  for (int i = 0; i < kOutRegs / 4; ++i) {
    const int c = i * 8 + col;
    const int r0 = gl_row, r1 = gl_row + 8;
    const uint32_t v0 = __half_as_ushort(__float2half_rn(out[4 * i + 0] * inv0)) |
                        (uint32_t(__half_as_ushort(__float2half_rn(out[4 * i + 1] * inv0))) << 16);
    const uint32_t v1 = __half_as_ushort(__float2half_rn(out[4 * i + 2] * inv1)) |
                        (uint32_t(__half_as_ushort(__float2half_rn(out[4 * i + 3] * inv1))) << 16);
    if (r0 < rows_valid) *reinterpret_cast<uint32_t*>(dst + static_cast<size_t>(r0) * row_stride + c) = v0;
    if (r1 < rows_valid) *reinterpret_cast<uint32_t*>(dst + static_cast<size_t>(r1) * row_stride + c) = v1;
  }
}

// -------------------------------------------------------------- host wrapper

void kernel(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
            const torch::Tensor& cu_seqlens_q, const torch::Tensor& cu_seqlens_k,
            const torch::Tensor& output) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && output.is_cuda(),
              "all tensors must be CUDA");
  TORCH_CHECK(q.scalar_type() == torch::kFloat16 && k.scalar_type() == torch::kFloat16 &&
                  v.scalar_type() == torch::kFloat16 && output.scalar_type() == torch::kFloat16,
              "q, k, v and output must be float16");
  TORCH_CHECK(cu_seqlens_q.scalar_type() == torch::kInt32 &&
                  cu_seqlens_k.scalar_type() == torch::kInt32,
              "cumulative lengths must be int32");
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q, k, v must be rank 3");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                  output.is_contiguous(),
              "tensors must be contiguous");
  TORCH_CHECK(q.size(2) == kDim, "head dimension must be 128");
  TORCH_CHECK(cu_seqlens_q.dim() == 1 && cu_seqlens_k.dim() == 1, "offsets must be rank 1");
  TORCH_CHECK(q.size(0) == output.size(0) && q.size(1) == output.size(1) &&
                  q.size(2) == output.size(2),
              "output shape must match q");
  const int num_seq = static_cast<int>(cu_seqlens_q.numel()) - 1;
  const int heads = static_cast<int>(q.size(1));
  const int64_t tokens = q.size(0);
  TORCH_CHECK(num_seq >= 1, "need at least one sequence");
  if (tokens == 0) return;

  c10::cuda::CUDAGuard guard(q.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();

  constexpr int kSmemBytes = (kM * kDim + 2 * kStages * kTileHalfs) * sizeof(__half);
  static bool configured = false;
  if (!configured) {
    cudaError_t err = cudaFuncSetAttribute(fmha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           kSmemBytes);
    TORCH_CHECK(err == cudaSuccess, "smem attribute: ", cudaGetErrorString(err));
    configured = true;
  }

  // Tiles never span sequences, so the padding waste is below one tile per sequence.
  const int tiles_max = static_cast<int>((tokens + kM - 1) / kM) + num_seq;
  const dim3 grid(tiles_max, heads);
  fmha_kernel<<<grid, kThreads, kSmemBytes, stream>>>(
      reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
      cu_seqlens_q.data_ptr<int>(), cu_seqlens_k.data_ptr<int>(),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()), heads, num_seq);
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "fmha launch: ", cudaGetErrorString(error));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("kernel", &kernel);
}

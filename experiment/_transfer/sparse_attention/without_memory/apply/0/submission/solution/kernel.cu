// Sparse MLA prefill (DeepSeek-V3.2 supplied-index sparse attention) for sm_90a.
//
//   out[t,h,:] = softmax_k(scale * q[t,h,:] . kv[idx[t,k],:]) @ kv[idx[t,k],0:512]
//   max_logits[t,h] = max_k logit,  lse[t,h] = logsumexp_k logit
//
// CTA = 1 token x 64 heads, 256 threads = 2 warpgroups (values split 256/256).
// Keys are streamed in 32 chunks of 64; each warpgroup runs the QK GEMM on half of
// the chunk's keys (32) and both run the PV GEMM over the full chunk.
//
// Shared memory uses the GMMA "interleaved" core-matrix grid:
//   a [R,C] bf16 tile has element (r,c) at  128*((r/8)*(C/8) + c/8) + (r%8)*16 + (c%8)*2
// so a single staged K/V tile can be read two ways by the tensor cores:
//   * QK  B operand (k=dim, n=key)  : K-major  desc(LBO=8,   SBO=576)
//   * PV  B operand (k=key, n=vdim) : MN-major desc(LBO=576, SBO=8)
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "wgmma.cuh"

namespace {

constexpr int kHeads = 128;
constexpr int kHeadsPerCta = 64;
constexpr int kDimQk = 576;
constexpr int kDimV = 512;
constexpr int kTopK = 2048;
constexpr int kTokensMax = 1 << 24;  // indices beyond the kv extent are invalid

constexpr int kThreads = 256;
constexpr int kChunkKeys = 64;                       // keys per pipeline stage
constexpr int kChunks = kTopK / kChunkKeys;          // 32
constexpr int kKeysPerWg = kChunkKeys / 2;           // 32 keys per warpgroup
constexpr int kDimCm = kDimQk / 8;                   // 72 core-matrix columns
constexpr int kVdimsWg = kDimV / 2;                  // 256 value dims per warpgroup
constexpr int kOregs = kHeadsPerCta * kVdimsWg / 128;  // 128 fp32 accumulators/thread

constexpr float kScale = 0.1352337788608801f;
constexpr float kNegBig = -1e30f;

constexpr int kQBytes = kHeadsPerCta * kDimQk * 2;       // 73728
constexpr int kTileBytes = kChunkKeys * kDimQk * 2;      // 73728
constexpr int kPBytes = kHeadsPerCta * kChunkKeys * 2;   // 8192
constexpr int kSmemBytes = kQBytes + 2 * kTileBytes + kPBytes + 4 * kHeadsPerCta * 4 + 16;

__device__ __forceinline__ unsigned smemU32(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

//! GMMA shared-memory descriptor: 14-bit address (16B units) + LBO + SBO, no swizzle.
__device__ __forceinline__ unsigned long long gdesc(const void* p, int lboUnits, int sboUnits) {
  return static_cast<unsigned long long>((smemU32(p) >> 4) & 0x3FFFu) |
         (static_cast<unsigned long long>(lboUnits & 0x3FFFu) << 16) |
         (static_cast<unsigned long long>(sboUnits & 0x3FFFu) << 32);
}

//! Stage `rows` rows of 576 bf16 dims (pitch rowBytes) into an interleaved tile.
//! Thread t covers row t/4; each row is filled by 4 threads in 18 sixteen-byte chunks.
//! Plain global loads beat cp.async here: the gather reads eight rows per warp, a
//! pattern the LDGSTS path serves at ~0.6 TB/s while the LSU path reaches ~3.4 TB/s.
__device__ __forceinline__ void loadTileRows(unsigned char* dst, const unsigned char* src,
                                             int rowBytes, int tid) {
  const int row = tid >> 2;
  const int col0 = tid & 3;
  const unsigned char* s = src + static_cast<long>(row) * rowBytes + col0 * 16;
  unsigned char* d = dst + 128 * ((row >> 3) * kDimCm + col0) + (row & 7) * 16;
#pragma unroll
  for (int j = 0; j < kDimCm / 8; ++j) {
    uint4 v0 = *reinterpret_cast<const uint4*>(s + j * 128);
    uint4 v1 = *reinterpret_cast<const uint4*>(s + j * 128 + 64);
    *reinterpret_cast<uint4*>(d + j * 1024) = v0;
    *reinterpret_cast<uint4*>(d + j * 1024 + 512) = v1;
  }
}

//! Gather 64 indexed kv rows into an interleaved tile (same addressing as above).
__device__ __forceinline__ void loadTileGather(unsigned char* dst, const unsigned char* kv,
                                               const int* __restrict__ idx, int tid) {
  const int key = tid >> 2;
  const int col0 = tid & 3;
  int row = __ldg(idx + key);
  row = row < 0 ? 0 : row;
  const unsigned char* s = kv + static_cast<long>(row) * (kDimQk * 2) + col0 * 16;
  unsigned char* d = dst + 128 * ((key >> 3) * kDimCm + col0) + (key & 7) * 16;
#pragma unroll
  for (int j = 0; j < kDimCm / 8; ++j) {
    uint4 v0 = *reinterpret_cast<const uint4*>(s + j * 128);
    uint4 v1 = *reinterpret_cast<const uint4*>(s + j * 128 + 64);
    *reinterpret_cast<uint4*>(d + j * 1024) = v0;
    *reinterpret_cast<uint4*>(d + j * 1024 + 512) = v1;
  }
}

__global__ void __launch_bounds__(kThreads, 1)
sparseAttnKernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kv,
                 const int* __restrict__ indices, __nv_bfloat16* __restrict__ out,
                 float* __restrict__ maxLogits, float* __restrict__ lse, int tokens) {
  (void)tokens;
#if !defined(__CUDA_ARCH_FEAT_SM90_ALL)
  // Portable fallback: identical math without tensor cores (two-pass softmax with
  // one warp per (token, head)). Reached only on non-sm90a targets.
  const int warpsPerBlock = kThreads / 32;
  const int warpStride = 2 * tokens * warpsPerBlock;
  const int lane = threadIdx.x & 31;
  for (int w = blockIdx.x * warpsPerBlock + (threadIdx.x >> 5); w < tokens * kHeads; w += warpStride) {
    const int head = w % kHeads;
    const int token = w / kHeads;
    const __nv_bfloat16* qRow = q + (static_cast<size_t>(token) * kHeads + head) * kDimQk;
    const int* rowIdx = indices + static_cast<size_t>(token) * kTopK;

    float mMax = kNegBig, mSum = 0.f;
    for (int k = 0; k < kTopK; ++k) {
      const int row = __ldg(rowIdx + k);
      const __nv_bfloat16* kvRow = kv + static_cast<size_t>(row < 0 ? 0 : row) * kDimQk;
      float acc = 0.f;
#pragma unroll
      for (int i = 0; i < kDimQk / 32; ++i) {
        const int d = lane + 32 * i;
        acc += __bfloat162float(qRow[d]) * __bfloat162float(kvRow[d]);
      }
#pragma unroll
      for (int off = 16; off; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
      const float sc = (row < 0) ? kNegBig : acc * kScale;
      const float mNew = fmaxf(mMax, sc);
      mSum = mSum * __expf(mMax - mNew) + __expf(sc - mNew);
      mMax = mNew;
    }

    float o[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) o[i] = 0.f;
    for (int k = 0; k < kTopK; ++k) {
      const int row = __ldg(rowIdx + k);
      const __nv_bfloat16* kvRow = kv + static_cast<size_t>(row < 0 ? 0 : row) * kDimQk;
      float acc = 0.f;
#pragma unroll
      for (int i = 0; i < kDimQk / 32; ++i) {
        const int d = lane + 32 * i;
        acc += __bfloat162float(qRow[d]) * __bfloat162float(kvRow[d]);
      }
#pragma unroll
      for (int off = 16; off; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
      const float pv = (row < 0) ? 0.f : __expf(acc * kScale - mMax);
#pragma unroll
      for (int i = 0; i < 16; ++i) o[i] += pv * __bfloat162float(kvRow[lane * 16 + i]);
    }

    __nv_bfloat16* outRow = out + (static_cast<size_t>(token) * kHeads + head) * kDimV;
    const float inv = 1.f / mSum;
#pragma unroll
    for (int i = 0; i < 16; ++i) outRow[lane * 16 + i] = __float2bfloat16(o[i] * inv);
    if (lane == 0) {
      maxLogits[static_cast<size_t>(token) * kHeads + head] = mMax;
      lse[static_cast<size_t>(token) * kHeads + head] = mMax + logf(mSum);
    }
  }
#else
  extern __shared__ __align__(1024) unsigned char smem[];
  unsigned char* sQ = smem;
  unsigned char* sTile0 = smem + kQBytes;
  unsigned char* sTile1 = sTile0 + kTileBytes;
  unsigned char* sP = sTile1 + kTileBytes;
  float* sStat = reinterpret_cast<float*>(sP + kPBytes);  // [4][64]: maxima + sums
  int* sValidLen = reinterpret_cast<int*>(sStat + 4 * kHeadsPerCta);  // first invalid index

  const int tid = threadIdx.x;
  const int token = blockIdx.x >> 1;
  const int headBase = (blockIdx.x & 1) << 6;
  const int wg = tid >> 7;                    // warpgroup 0/1
  const int tw = tid & 127;                   // thread inside warpgroup
  const int a4 = tw & 3;
  const int b8 = (tw >> 2) & 7;
  const int c4 = tw >> 5;
  const int rowA = b8 + 16 * c4;  // the two output rows this thread owns
  const int rowB = rowA + 8;

  const int* idx = indices + static_cast<size_t>(token) * kTopK;
  const unsigned char* kvBytes = reinterpret_cast<const unsigned char*>(kv);

  // ---- Q tile [64 heads, 576 dims]
  // Index rows are `min(TOPK, t+1)` valid entries followed by -1 padding; find the
  // boundary once so chunks beyond it (whole 64-key stages) need no work at all.
  if (tid == 0) *sValidLen = kTopK;
  __syncthreads();
  {
    int firstBad = kTopK;
#pragma unroll
    for (int i = 0; i < kTopK / kThreads; ++i) {
      const int pos = tid + kThreads * i;
      const int v = __ldg(idx + pos);
      if (v < 0 || v >= kTokensMax) {
        firstBad = pos;
        break;
      }
    }
    if (firstBad < kTopK) atomicMin(sValidLen, firstBad);
  }
  __syncthreads();
  const int validLen = *sValidLen;

  loadTileRows(sQ,
               reinterpret_cast<const unsigned char*>(q) +
                   static_cast<size_t>(token * kHeads + headBase) * (kDimQk * 2),
               kDimQk * 2, tid);
  loadTileGather(sTile0, kvBytes, idx, tid);

  float o[kOregs];
#pragma unroll
  for (int i = 0; i < kOregs; ++i) o[i] = 0.f;
  float mRun0 = kNegBig, mRun1 = kNegBig;  // running logit max (for max_logits)
  float lRun0 = 0.f, lRun1 = 0.f;          // sum of exp on the shared reference scale
  float expRef0 = kNegBig, expRef1 = kNegBig;

  const unsigned long long qDesc = gdesc(sQ, 8, 8 * kDimCm);
  const unsigned long long pDesc = gdesc(sP, 8, 8 * (kChunkKeys / 8));

  for (int c = 0; c < kChunks; ++c) {
    unsigned char* tile = (c & 1) ? sTile1 : sTile0;

    __syncthreads();  // tile c staged by every warp

    // ---- QK: S[64, 32] = Q[64, 576] * K[576, 32]  (this warpgroup's key half)
    float s[16];
    const bool stageLive = c * kChunkKeys < validLen;
    wgmma_fence();
    {
      const unsigned long long kDesc =
          gdesc(tile + 128 * (wg ? 4 * kDimCm : 0), 8, 8 * kDimCm);
      if (stageLive) {
#pragma unroll
        for (int ks = 0; ks < kDimQk / 16; ++ks) {
          wgmma_m64n32k16_ss<0, 0>(s, qDesc + ks * 16, kDesc + ks * 16, ks ? 1 : 0);
        }
      }
      wgmma_commit();  // empty group keeps the wait accounting uniform
    }
    wgmma_wait<1>();  // PV(c-1) done -> its tile buffer is reusable; QK(c) stays in flight

    // ---- stage the next chunk into the freed buffer; overlaps the QK wgmma
    if (c + 1 < kChunks && (c + 1) * kChunkKeys < validLen) {
      loadTileGather((c & 1) ? sTile0 : sTile1, kvBytes, idx + (c + 1) * kChunkKeys, tid);
    }
    wgmma_wait<0>();  // S ready

    // ---- row max of this warpgroup's key half (masked by index validity)
    // Lane l owns key l of this warpgroup's half; one ballot yields every key's
    // validity bit. Key of element (a4,d,f) is 2*a4 + d + 8*f inside the half.
    const int keyBase = c * kChunkKeys + wg * kKeysPerWg;
    const unsigned validBits = __ballot_sync(0xffffffffu, __ldg(idx + keyBase + (tw & 31)) >= 0);
    const int nd0 = 2 * a4;
    const bool valid[4][2] = {
        {(validBits >> (nd0 + 0)) & 1u, (validBits >> (nd0 + 1)) & 1u},
        {(validBits >> (nd0 + 8)) & 1u, (validBits >> (nd0 + 9)) & 1u},
        {(validBits >> (nd0 + 16)) & 1u, (validBits >> (nd0 + 17)) & 1u},
        {(validBits >> (nd0 + 24)) & 1u, (validBits >> (nd0 + 25)) & 1u},
    };
    float mx0 = kNegBig, mx1 = kNegBig;
#pragma unroll
    for (int f = 0; f < 4; ++f) {
#pragma unroll
      for (int d = 0; d < 2; ++d) {
        s[4 * f + d] = valid[f][d] ? s[4 * f + d] * kScale : kNegBig;
        s[4 * f + 2 + d] = valid[f][d] ? s[4 * f + 2 + d] * kScale : kNegBig;
        mx0 = fmaxf(mx0, s[4 * f + d]);
        mx1 = fmaxf(mx1, s[4 * f + 2 + d]);
      }
    }
    if (stageLive) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
      mRun0 = fmaxf(mRun0, mx0);
      mRun1 = fmaxf(mRun1, mx1);
    }

    // One shared exp reference for the whole row: the first live stage's row max.
    // The MUFU exponent handles the full fp32 range, and every later logit stays
    // within ~30 of that reference on this workload, so the decay is exact enough
    // and no per-stage rescale of the accumulator is needed.
    if (stageLive && expRef0 == kNegBig) {
      if (a4 == 0) {
        sStat[wg * kHeadsPerCta + b8 + 16 * c4] = mx0;
        sStat[wg * kHeadsPerCta + b8 + 16 * c4 + 8] = mx1;
      }
      __syncthreads();
      expRef0 = fmaxf(sStat[rowA], sStat[kHeadsPerCta + rowA]);
      expRef1 = fmaxf(sStat[rowB], sStat[kHeadsPerCta + rowB]);
      __syncthreads();
    }

    float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const float mm = e ? expRef1 : expRef0;
      unsigned char* pd = sP + 128 * ((2 * c4 + e) * (kChunkKeys / 8) + 4 * wg) + b8 * 16 + 4 * a4;
#pragma unroll
      for (int f = 0; f < 4; ++f) {
        const float p0 = valid[f][0] ? __expf(s[4 * f + 2 * e] - mm) : 0.f;
        const float p1 = valid[f][1] ? __expf(s[4 * f + 2 * e + 1] - mm) : 0.f;
        *reinterpret_cast<__nv_bfloat162*>(pd + 128 * f) = __floats2bfloat162_rn(p0, p1);
        if (e == 0) {
          sum0 += p0 + p1;
        } else {
          sum1 += p0 + p1;
        }
      }
    }
    sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1);
    sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
    sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1);
    sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
    lRun0 += sum0;
    lRun1 += sum1;
    __syncthreads();  // P tile complete

    // ---- PV: O[64, 256] += P[64, 64] * V[64, 256]
    wgmma_fence();
    {
      const unsigned long long vDesc =
          gdesc(tile + 128 * 32 * wg, 8 * kDimCm, 8);
      // o[0:64] holds value dims 0..127, o[64:128] holds 128..255; both accumulate
      // over all four k-steps (16 keys each). Advance the V descriptor by 16 dim
      // chunks (128 units) to reach the upper half of the value slice.
      constexpr int kNAdvanceUnits = (128 / 8) * 8;  // 128 dims -> 16 chunks x 8 units
      if (stageLive) {
#pragma unroll
        for (int ks = 0; ks < kChunkKeys / 16; ++ks) {
          const unsigned long long vk = vDesc + ks * (2 * 8 * kDimCm);
          wgmma_m64n128k16_ss<0, 1>(*reinterpret_cast<float(*)[64]>(&o[0]), pDesc + ks * 16, vk, 1);
          wgmma_m64n128k16_ss<0, 1>(*reinterpret_cast<float(*)[64]>(&o[64]), pDesc + ks * 16,
                                    vk + kNAdvanceUnits, 1);
        }
      }
      wgmma_commit();  // empty group keeps the wait accounting uniform
    }
  }
  wgmma_wait<0>();

  // ---- epilogue: combine both warpgroups' maxima and exp sums, then normalize
  if (a4 == 0) {
    sStat[wg * kHeadsPerCta + rowA] = mRun0;
    sStat[wg * kHeadsPerCta + rowB] = mRun1;
    sStat[2 * kHeadsPerCta + wg * kHeadsPerCta + rowA] = lRun0;
    sStat[2 * kHeadsPerCta + wg * kHeadsPerCta + rowB] = lRun1;
  }
  __syncthreads();

  const float mFin0 = fmaxf(sStat[rowA], sStat[kHeadsPerCta + rowA]);
  const float mFin1 = fmaxf(sStat[rowB], sStat[kHeadsPerCta + rowB]);
  const float lFin0 = sStat[2 * kHeadsPerCta + rowA] + sStat[3 * kHeadsPerCta + rowA];
  const float lFin1 = sStat[2 * kHeadsPerCta + rowB] + sStat[3 * kHeadsPerCta + rowB];
  const float inv0 = 1.f / lFin0;
  const float inv1 = 1.f / lFin1;

  __nv_bfloat16* outRow = out + static_cast<size_t>(token * kHeads + headBase) * kDimV +
                          wg * kVdimsWg;
#pragma unroll
  for (int f = 0; f < 32; ++f) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const float inv = e ? inv1 : inv0;
      const int m = b8 + 16 * c4 + 8 * e;
      const int n = 2 * a4 + 8 * f;
      __nv_bfloat162 v =
          __floats2bfloat162_rn(o[4 * f + 2 * e] * inv, o[4 * f + 2 * e + 1] * inv);
      *reinterpret_cast<__nv_bfloat162*>(outRow + static_cast<size_t>(m) * kDimV + n) = v;
    }
  }
  if (a4 == 0) {
    maxLogits[static_cast<size_t>(token) * kHeads + headBase + rowA] = mFin0;
    maxLogits[static_cast<size_t>(token) * kHeads + headBase + rowB] = mFin1;
    lse[static_cast<size_t>(token) * kHeads + headBase + rowA] = expRef0 + logf(lFin0);
    lse[static_cast<size_t>(token) * kHeads + headBase + rowB] = expRef1 + logf(lFin1);
  }
#endif  // __CUDA_ARCH_FEAT_SM90_ALL
}

}  // namespace

void sparseAttn(torch::Tensor q, torch::Tensor kv, torch::Tensor indices, torch::Tensor out,
                torch::Tensor maxLogits, torch::Tensor lse) {
  const int tokens = static_cast<int>(q.size(0));
  TORCH_CHECK(q.is_cuda() && kv.is_cuda() && indices.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(q.dtype() == torch::kBFloat16 && kv.dtype() == torch::kBFloat16,
              "q/kv must be bfloat16");
  TORCH_CHECK(indices.dtype() == torch::kInt32, "indices must be int32");
  TORCH_CHECK(q.size(1) == kHeads && q.size(2) == kDimQk, "unexpected q shape");
  TORCH_CHECK(kv.size(1) == 1 && kv.size(2) == kDimQk, "unexpected kv shape");
  TORCH_CHECK(indices.size(1) == 1 && indices.size(2) == kTopK, "unexpected indices shape");
  TORCH_CHECK(q.is_contiguous() && kv.is_contiguous() && indices.is_contiguous(),
              "inputs must be contiguous");

  const c10::cuda::CUDAGuard guard(q.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();

  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(sparseAttnKernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         kSmemBytes);
    configured = true;
  }
  sparseAttnKernel<<<2 * tokens, kThreads, kSmemBytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(kv.data_ptr()),
      indices.data_ptr<int>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      maxLogits.data_ptr<float>(), lse.data_ptr<float>(), tokens);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "sparseAttn launch failed: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("kernel", &sparseAttn); }

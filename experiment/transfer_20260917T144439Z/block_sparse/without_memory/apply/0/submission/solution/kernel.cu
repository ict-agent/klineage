// Causal block-sparse attention, one CTA per (query block, query head) on sm_90.
//
//   out[b,h,qb*128 + r, :] = sum_slots softmax_j( q.k_j / sqrt(D) ) v_j
//
// Slot s of (b,h,qb) names key block ids[b,h,qb,s]; its keys sit at global token
// id*128 + c and count iff s < block_counts and token <= qb*128 + r.  Only the
// diagonal slot (id == qb) truncates, every other live id is fully open, and ids
// outside [0, qb] are padding that never becomes live.
//
// 256 threads = 2 warpgroups of 4 warps; warpgroup w owns query rows 64w..64w+64.
// Both products run on warpgroup mma, whose B operand comes straight from shared
// memory descriptors:
//
//   S = Q K^T     m64n64k16, A and B from shared memory, one chunk per 64 keys
//   O += P V      m64n128k16, A = P (registers), B = V (shared, transposed)
//
// Shared tiles follow the canonical no-swizzle K-major GMMA layout
//
//   unit(m, c/8) = (c>>6)*16384 + (m>>3)*1024 + (m&7)*128 + (((c>>3)&7)^(m&7))*16
//
// so that 8-element runs lie along k inside 128-byte core matrices.  Descriptors
// use LBO = 8 and SBO = 128 (16-byte units): +256 B per k step, +2048 B per
// M/N block.  The second product needs V transposed, so each slot stages V in
// its natural [key][dim] order and flips it in shared memory with
// ldmatrix.x4.trans / stmatrix.x4.
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace {

using bf16 = __nv_bfloat16;

constexpr int kBlock = 128;                    // tokens per key block
constexpr int kDim = 128;                      // head dimension
constexpr int kSlots = 16;                     // block_ids capacity
constexpr int kThr = 256;                      // two warpgroups
constexpr int kTile = kBlock * kDim;           // bf16 elements in one 128x128 tile
constexpr int kChunkKeys = 64;                 // keys per S chunk
constexpr int kChunkTiles = kChunkKeys / 8;    // n8 tiles per chunk
constexpr int kSRegs = kChunkTiles * 4;        // fp32 S accumulators per thread
constexpr int kORegs = kDim / 8 * 4;           // fp32 O accumulators per thread
constexpr int kChunks = kBlock / kChunkKeys;
constexpr int kSteps = kDim / 16;              // k steps over the head dimension
constexpr int kNTiles = kDim / 8;              // n8 tiles over the head dimension
constexpr int kVTiles = kTile * 7;             // Q + K x2 + V x2 + Vt x2
// 128B-swizzled K-major layout: 8-row x 64-element atoms of 1024B with the 16B
// unit inside a 128B row permuted by the row index.  Descriptor steps: LBO = one
// 16B unit (next core matrix along k), SBO = 1024B (next 8-row group).
constexpr uint32_t kLbo = 1;
constexpr uint32_t kSbo = 64;
constexpr uint32_t kAtomBytes = 16384;         // one 8-row atom of a 128-column tile
constexpr uint32_t kRowGrpBytes = 1024;        // 8 rows of the tile
constexpr uint32_t kKStepBytes = 32;           // 16 columns = 2 units
constexpr uint32_t kUnitBytes = 16;
constexpr float kNegBig = -1.0e30f;
constexpr float kScaleLog2e = 1.4426950408889634f / 11.313708498984761f;

__device__ __forceinline__ uint32_t saddr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// swizzle 128B, base offset 0, LBO = k-core-matrix step, SBO = M/N block step
__device__ __forceinline__ uint64_t wdesc(uint32_t addr) {
    return (uint64_t)((addr >> 4) & 0x3FFF) | ((uint64_t)kLbo << 16) |
           ((uint64_t)kSbo << 32) | ((uint64_t)1 << 46) | ((uint64_t)1 << 62);
}

// Byte offset of the 16B unit holding row m, column unit cu (element 8*cu) of a
// 128B-swizzled K-major tile.
__device__ __forceinline__ uint32_t unit_off(int m, int cu) {
    return (uint32_t)(cu >> 3) * kAtomBytes + (uint32_t)(m >> 3) * kRowGrpBytes +
           (uint32_t)((m & 7) << 7) + (uint32_t)(((cu & 7) ^ (m & 7)) * kUnitBytes);
}

// Descriptor start address of k-step kk (16 columns) for a tile starting at
// column 0.
__device__ __forceinline__ uint32_t kstep(uint32_t base, int kk) {
    return base + (uint32_t)(kk >> 2) * kAtomBytes + (uint32_t)(kk & 3) * kKStepBytes;
}

__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void cp_wait1() { asm volatile("cp.async.wait_group 1;"); }

__device__ __forceinline__ void ldsm_x4_t(uint32_t a, uint32_t* r) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void stsm_x4(uint32_t a, const uint32_t* r) {
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};"
                 :: "r"(a), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]));
}
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;"); }
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;"); }
__device__ __forceinline__ void wg_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;"); }
__device__ __forceinline__ void wg_wait1() { asm volatile("wgmma.wait_group.sync.aligned 1;"); }

__device__ __forceinline__ void wg_ss64(float (&d)[kSRegs], uint64_t da, uint64_t db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 0, 0;\n}\n"
        : "+f"(d[0]),
            "+f"(d[1]),
            "+f"(d[2]),
            "+f"(d[3]),
            "+f"(d[4]),
            "+f"(d[5]),
            "+f"(d[6]),
            "+f"(d[7]),
            "+f"(d[8]),
            "+f"(d[9]),
            "+f"(d[10]),
            "+f"(d[11]),
            "+f"(d[12]),
            "+f"(d[13]),
            "+f"(d[14]),
            "+f"(d[15]),
            "+f"(d[16]),
            "+f"(d[17]),
            "+f"(d[18]),
            "+f"(d[19]),
            "+f"(d[20]),
            "+f"(d[21]),
            "+f"(d[22]),
            "+f"(d[23]),
            "+f"(d[24]),
            "+f"(d[25]),
            "+f"(d[26]),
            "+f"(d[27]),
            "+f"(d[28]),
            "+f"(d[29]),
            "+f"(d[30]),
            "+f"(d[31])
        : "l"(da), "l"(db), "r"(1));
}

__device__ __forceinline__ void wg_rs128(float (&d)[kORegs], const uint32_t (&a)[4], uint64_t db) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %68, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]),
            "+f"(d[1]),
            "+f"(d[2]),
            "+f"(d[3]),
            "+f"(d[4]),
            "+f"(d[5]),
            "+f"(d[6]),
            "+f"(d[7]),
            "+f"(d[8]),
            "+f"(d[9]),
            "+f"(d[10]),
            "+f"(d[11]),
            "+f"(d[12]),
            "+f"(d[13]),
            "+f"(d[14]),
            "+f"(d[15]),
            "+f"(d[16]),
            "+f"(d[17]),
            "+f"(d[18]),
            "+f"(d[19]),
            "+f"(d[20]),
            "+f"(d[21]),
            "+f"(d[22]),
            "+f"(d[23]),
            "+f"(d[24]),
            "+f"(d[25]),
            "+f"(d[26]),
            "+f"(d[27]),
            "+f"(d[28]),
            "+f"(d[29]),
            "+f"(d[30]),
            "+f"(d[31]),
            "+f"(d[32]),
            "+f"(d[33]),
            "+f"(d[34]),
            "+f"(d[35]),
            "+f"(d[36]),
            "+f"(d[37]),
            "+f"(d[38]),
            "+f"(d[39]),
            "+f"(d[40]),
            "+f"(d[41]),
            "+f"(d[42]),
            "+f"(d[43]),
            "+f"(d[44]),
            "+f"(d[45]),
            "+f"(d[46]),
            "+f"(d[47]),
            "+f"(d[48]),
            "+f"(d[49]),
            "+f"(d[50]),
            "+f"(d[51]),
            "+f"(d[52]),
            "+f"(d[53]),
            "+f"(d[54]),
            "+f"(d[55]),
            "+f"(d[56]),
            "+f"(d[57]),
            "+f"(d[58]),
            "+f"(d[59]),
            "+f"(d[60]),
            "+f"(d[61]),
            "+f"(d[62]),
            "+f"(d[63])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(1));
}

__device__ __forceinline__ float ex2f(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ uint32_t pack2(float x, float y) {
    __nv_bfloat162 h = __floats2bfloat162_rn(x, y);
    return *reinterpret_cast<uint32_t*>(&h);
}

// 128x128 global block -> shared tile in the interleaved GMMA layout, 16B at a time.
// Thread tid owns column unit (tid & 15) and rows (tid >> 4) + 16j: both addresses
// advance by 4096 B per step.
__device__ __forceinline__ void stage_tile(uint32_t tile, const bf16* src, int tid) {
    const int row = tid >> 4, cu = tid & 15;
    uint32_t dst = tile + unit_off(row, cu);
    const char* p = reinterpret_cast<const char*>(src) + ((row << 7) + (cu << 3)) * 2;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        cp16(dst, p);
        dst += 2 * kRowGrpBytes;   // +16 rows, same column half
        p += 4096;
    }
}

// V[key][dim] -> Vt[dim][key], one 2x2 grid of 8x8 blocks per warp instruction.
__device__ __forceinline__ void transpose_v(uint32_t src, uint32_t dst, int tid) {
    const int warp = tid >> 5, lane = tid & 31, which = lane >> 3, row = lane & 7;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int grid = warp * 8 + i, a = grid >> 3, b = grid & 7;
        const int kb = 2 * a + ((which >> 1) & 1), db = 2 * b + (which & 1);
        uint32_t r[4];
        ldsm_x4_t(src + unit_off(8 * kb + row, db), r);
        stsm_x4(dst + unit_off(8 * db + row, kb), r);
    }
}

// P = exp2(scale * S) for one 64-key chunk; DIAG cuts the causal diagonal block.
// Masked lanes underflow to zero, so the normaliser is a plain row sum.
template <bool DIAG>
__device__ __forceinline__ void chunk_softmax(float (&s)[kSRegs], uint32_t (&ap)[kChunkTiles / 2][4],
                                              int key0, int row0, int g, int lc, float (&l)[2]) {
#pragma unroll
    for (int j = 0; j < kChunkTiles; ++j) {
        const int c0 = key0 + 8 * j + 2 * lc;
        const int r0 = row0 + g, r1 = r0 + 8;
        if (DIAG && c0 > r0) s[4 * j] = kNegBig;
        if (DIAG && c0 + 1 > r0) s[4 * j + 1] = kNegBig;
        if (DIAG && c0 > r1) s[4 * j + 2] = kNegBig;
        if (DIAG && c0 + 1 > r1) s[4 * j + 3] = kNegBig;

        s[4 * j] = ex2f(s[4 * j] * kScaleLog2e);
        s[4 * j + 1] = ex2f(s[4 * j + 1] * kScaleLog2e);
        s[4 * j + 2] = ex2f(s[4 * j + 2] * kScaleLog2e);
        s[4 * j + 3] = ex2f(s[4 * j + 3] * kScaleLog2e);
        l[0] += s[4 * j] + s[4 * j + 1];
        l[1] += s[4 * j + 2] + s[4 * j + 3];
    }
#pragma unroll
    for (int j = 0; j < kChunkTiles / 2; ++j) {
        ap[j][0] = pack2(s[8 * j], s[8 * j + 1]);
        ap[j][1] = pack2(s[8 * j + 2], s[8 * j + 3]);
        ap[j][2] = pack2(s[8 * j + 4], s[8 * j + 5]);
        ap[j][3] = pack2(s[8 * j + 6], s[8 * j + 7]);
    }
}

__global__ void __launch_bounds__(kThr, 1) block_sparse_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
    const int* __restrict__ block_ids, const int* __restrict__ block_counts,
    bf16* __restrict__ out, int S, int NB, int HQ, int HKV, int groups) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
    (void)q; (void)k; (void)v; (void)block_ids; (void)block_counts; (void)out;
    (void)S; (void)NB; (void)HQ; (void)HKV; (void)groups;
#else
    const int h = blockIdx.x, qb = blockIdx.y, b = blockIdx.z;
    const int kvh = h / groups;
    const int t = threadIdx.x;
    const int wg = t >> 7, wrp = (t >> 5) & 3, lane = t & 31;
    const int g = lane >> 2, lc = lane & 3;
    const int row0 = 64 * wg + 16 * wrp;

    extern __shared__ __align__(128) bf16 smem[];

    // Double-buffered tile bases. Kept as scalar shared addresses: a `base[phase]`
    // pointer array would spill to local memory, whose per-lane stride makes every
    // reload an uncoalesced 32-sector L2 access.
    const uint32_t qt = saddr(smem);
    const uint32_t kt0 = qt + kTile * 2, kt1 = kt0 + kTile * 2;
    const uint32_t vn0 = kt1 + kTile * 2, vn1 = vn0 + kTile * 2;
    const uint32_t vt0 = vn1 + kTile * 2, vt1 = vt0 + kTile * 2;

    const int count = __ldg(block_counts + (b * HQ + h) * NB + qb);
    const int* ids = block_ids + ((b * HQ + h) * NB + qb) * kSlots;
    const long kvrow = (long)(b * HKV + kvh) * S * kDim;

    stage_tile(qt, q + ((long)(b * HQ + h) * S + qb * kBlock) * kDim, t);

    int live = 0;
    for (int s = 0; s < count; ++s) {
        const int id = __ldg(ids + s);
        if (id >= 0 && id <= qb) live |= 1 << s;
    }
    int cur = (live == 0) ? -1 : __ffs(live) - 1;
    if (cur >= 0) {
        const int id = __ldg(ids + cur);
        stage_tile(kt0, k + kvrow + (long)id * kBlock * kDim, t);
        stage_tile(vn0, v + kvrow + (long)id * kBlock * kDim, t);
    }
    cp_commit();

    float o[kORegs];
#pragma unroll
    for (int j = 0; j < kORegs; ++j) o[j] = 0.f;
    float l[2] = {0.f, 0.f};

    const uint32_t qa = qt + wg * 8 * kRowGrpBytes;
    int phase = 0;
    while (cur >= 0) {
        const int rest = live >> (cur + 1);
        const int nxt = rest ? __ffs(rest) + cur : -1;
        const int id = __ldg(ids + cur);

        if (nxt >= 0) {
            const int nid = __ldg(ids + nxt);
            stage_tile(phase ? kt0 : kt1, k + kvrow + (long)nid * kBlock * kDim, t);
            stage_tile(phase ? vn0 : vn1, v + kvrow + (long)nid * kBlock * kDim, t);
        }
        cp_commit();
        cp_wait1();
        __syncthreads();

        const uint32_t ka = phase ? kt1 : kt0;
        const uint32_t vta = phase ? vt1 : vt0;
        float s[kChunks][kSRegs];
#pragma unroll
        for (int ch = 0; ch < kChunks; ++ch)
#pragma unroll
            for (int j = 0; j < kSRegs; ++j) s[ch][j] = 0.f;

        // S = Q K^T, both operands straight from shared memory. One commit per chunk:
        // waiting for chunk 0 lets chunk 1 run while its softmax and PV proceed.
        wg_fence();
#pragma unroll
        for (int ch = 0; ch < kChunks; ++ch) {
#pragma unroll
            for (int kk = 0; kk < kSteps; ++kk)
                wg_ss64(s[ch], wdesc(kstep(qa, kk)),
                        wdesc(kstep(ka + ch * 8 * kRowGrpBytes, kk)));
            wg_commit();
        }

        // V transpose overlaps the asynchronous S.
        transpose_v(phase ? vn1 : vn0, vta, t);
        __syncthreads();
        wg_fence();

        uint32_t ap[kChunkTiles / 2][4];
        float lc_acc[2] = {0.f, 0.f};
        const bool diag = (id == qb);
#pragma unroll
        for (int ch = 0; ch < kChunks; ++ch) {
            wg_wait1();
            if (diag) {
                chunk_softmax<true>(s[ch], ap, ch * kChunkKeys, row0, g, lc, lc_acc);
            } else {
                chunk_softmax<false>(s[ch], ap, ch * kChunkKeys, row0, g, lc, lc_acc);
            }
#pragma unroll
            for (int ks = 0; ks < kChunkTiles / 2; ++ks)
                wg_rs128(o, ap[ks], wdesc(kstep(vta, ch * 4 + ks)));
            wg_commit();
        }
        l[0] += lc_acc[0];
        l[1] += lc_acc[1];

        if (nxt < 0) break;
        cur = nxt;
        phase ^= 1;
    }

#pragma unroll
    for (int d = 1; d <= 2; d <<= 1) {
        l[0] += __shfl_xor_sync(0xffffffffu, l[0], d);
        l[1] += __shfl_xor_sync(0xffffffffu, l[1], d);
    }
    const float rinv[2] = {1.f / l[0], 1.f / l[1]};
    bf16* dst = out + ((long)(b * HQ + h) * S + qb * kBlock) * kDim;
#pragma unroll
    for (int j = 0; j < kNTiles; ++j) {
        bf16* p0 = dst + (row0 + g) * kDim + 8 * j + 2 * lc;
        bf16* p1 = p0 + 8 * kDim;
        *reinterpret_cast<uint32_t*>(p0) = pack2(o[4 * j] * rinv[0], o[4 * j + 1] * rinv[0]);
        *reinterpret_cast<uint32_t*>(p1) = pack2(o[4 * j + 2] * rinv[1], o[4 * j + 3] * rinv[1]);
    }
#endif
}

void launch(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
            const torch::Tensor& block_ids, const torch::Tensor& block_counts, torch::Tensor& out,
            cudaStream_t stream) {
    const int S = q.size(2), NB = block_ids.size(2), HQ = q.size(1), HKV = k.size(1);
    constexpr int smem_bytes = kVTiles * 2;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(block_sparse_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes);
        configured = true;
    }
    dim3 grid(HQ, NB, q.size(0));
    block_sparse_kernel<<<grid, kThr, smem_bytes, stream>>>(
        reinterpret_cast<const bf16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16*>(v.data_ptr<at::BFloat16>()),
        block_ids.data_ptr<int>(), block_counts.data_ptr<int>(),
        reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>()), S, NB, HQ, HKV, HQ / HKV);
    C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void kernel(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
            const torch::Tensor& block_ids, const torch::Tensor& block_counts, torch::Tensor& out) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && block_ids.is_cuda() && block_counts.is_cuda() && out.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                    v.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16,
                "q/k/v/out must be bfloat16");
    TORCH_CHECK(block_ids.scalar_type() == at::kInt && block_counts.scalar_type() == at::kInt,
                "block_ids/block_counts must be int32");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                    block_ids.is_contiguous() && block_counts.is_contiguous() && out.is_contiguous(),
                "tensors must be contiguous");
    TORCH_CHECK(q.dim() == 4 && q.size(3) == kDim, "q must be [B,HQ,S,128]");
    TORCH_CHECK(block_ids.dim() == 4 && block_ids.size(3) == kSlots, "block_ids must be [B,HQ,NB,16]");
    TORCH_CHECK(q.size(2) % block_ids.size(2) == 0 && q.size(2) / block_ids.size(2) == kBlock,
                "query block size must be 128");
    TORCH_CHECK(q.size(1) % k.size(1) == 0, "query heads must be a multiple of kv heads");

    const at::cuda::CUDAGuard guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
    launch(q, k, v, block_ids, block_counts, out, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("kernel", &kernel); }

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define DEV __device__ __forceinline__

namespace {

constexpr int kDim     = 128;
constexpr int kHeadsQ  = 32;
constexpr int kHeadsKV = 8;
constexpr int kGroup   = kHeadsQ / kHeadsKV;
constexpr int kRowsHead = 32;                 // q rows per head per CTA
constexpr int kBlockN   = 128;                // kv columns per tile
constexpr int kThreads  = 256;                // two warpgroups
constexpr int kStages   = 3;
constexpr int kQRowStride  = kHeadsQ * kDim;
constexpr int kKVRowStride = kHeadsKV * kDim;
constexpr unsigned kFull = 0xffffffffu;

constexpr float kScale    = 0.08838834764831845f;   // 1/sqrt(128)
constexpr float kLog2e    = 1.4426950408889634f;
constexpr float kScaleL2e = kScale * kLog2e;

// ---- wgmma core matrix layout: 8 kv rows x 8 dims, 128 B, d contiguous ----
constexpr int kLboB  = 144;                   // dim block (8 d) stride, 16 B pad
constexpr int kSboB  = (kDim / 8) * kLboB;    // kv block (8 rows) stride
constexpr int kTileB = (kBlockN / 8) * kSboB;
constexpr int kStageB = 2 * kTileB;
constexpr int kSmemB = kStages * kStageB;
constexpr int kDRegs = kBlockN / 2;           // 64 accumulator floats per thread
constexpr int kKStep = 18;                    // 16 dims = 2 dim blocks = 288 B / 16
constexpr int kVStep = 288;                   // 16 kv = 2 kv blocks = 4608 B / 16

// The tile is consumed once, so a 16 B asynchronous copy is the cheapest way to
// move it: no register round trip and no warp stall on the global latency.
DEV void cpa16(void* dst, const void* src) {
    uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
}

DEV void sts16(void* dst, uint4 r) {
    uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};\n" ::"r"(d), "r"(r.x), "r"(r.y),
                 "r"(r.z), "r"(r.w));
}

DEV float ex2(float x) { float y; asm("ex2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(x)); return y; }
DEV uint32_t pack_bf16(float lo, float hi) {
    __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}

DEV uint64_t core_desc(const void* p, int lbo_b, int sbo_b) {
    const uint64_t a = (uint64_t)(uint32_t)__cvta_generic_to_shared(p);
    return ((a >> 4) & 0x3FFFULL) | ((uint64_t)(lbo_b >> 4) << 16) |
           ((uint64_t)(sbo_b >> 4) << 32);
}

// 128 kv x 128 dim tile split into 16 x 16 core matrices of 128 B.
DEV void load_tile(const __nv_bfloat16* __restrict__ src, char* dst, int tid) {
    const char* g = reinterpret_cast<const char*>(src);
    uint4 r[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int c = tid + i * kThreads;      // 0 .. 2047
        r[i] = *reinterpret_cast<const uint4*>(g + (size_t)(c >> 4) * (kKVRowStride * 2) +
                                               (c & 15) * 16);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int c = tid + i * kThreads;
        sts16(dst + ((c >> 4) & 7) * 16 + (c >> 7) * kSboB + (c & 15) * kLboB, r[i]);
    }
}

DEV void cpa_tile(const __nv_bfloat16* __restrict__ src, char* dst, int tid) {
    // Thread t owns dim chunk (t & 15) of kv block (t >> 4). Its eight copies
    // walk the eight rows of that block, so both addresses advance by a
    // constant: +16 B in shared, +one kv row in global.
    const int blk = tid >> 4, chunk = tid & 15;
    const char* gb = reinterpret_cast<const char*>(src) + (size_t)blk * (8 * kKVRowStride * 2) +
                     chunk * 16;
    char* db = dst + blk * kSboB + chunk * kLboB;
#pragma unroll
    for (int r = 0; r < 8; ++r)
        cpa16(db + r * 16, gb + (size_t)r * (kKVRowStride * 2));
    asm volatile("cp.async.commit_group;\n" ::);
}

// imm-trans-b selects the majorness of the shared memory operand:
//   0 = K-major (Q*K^T),  1 = N-major (P*V).
template <int kTransB>
DEV void wgmma(float (&d)[kDRegs], const uint32_t (&a)[4], uint64_t bd) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, "
        "{%64,%65,%66,%67}, %68, 1, 1, 1, %69;\n"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),"+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(bd), "n"(kTransB));
}

// ------------------------------------------------------------------ kernel ---
__global__ void __launch_bounds__(kThreads, 1)
gqa_wg2(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
          const __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ o,
          int seq, int qblocks) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    extern __shared__ __nv_bfloat16 smem[];
    char* st = reinterpret_cast<char*>(smem);

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int wg   = (tid >> 7) & 1;      // warpgroup: heads 0,1 or 2,3
    const int w    = (tid >> 5) & 3;      // warp inside the warpgroup

    const int grp = blockIdx.x / qblocks;                 // kv head
    const int qb  = qblocks - 1 - (blockIdx.x % qblocks); // long CTAs first
    const int q0  = qb * kRowsHead;
    const int kvl = (q0 + kRowsHead - 1) >> 7;

    // ---- A fragments of Q, straight from global memory ----
    // wgmma row r = 16*w + lane/4 (+8); rows 0-31 are head 2*wg, 32-63 head 2*wg+1.
    const int r0  = 16 * w + (lane >> 2);
    const int r1  = r0 + 8;
    const int c2  = (lane & 3) * 2;
    const __nv_bfloat16* q0p = q + ((size_t)(q0 + (r0 & 31)) * kHeadsQ + grp * kGroup + 2 * wg + (r0 >> 5)) * kDim;
    const __nv_bfloat16* q1p = q + ((size_t)(q0 + (r1 & 31)) * kHeadsQ + grp * kGroup + 2 * wg + (r1 >> 5)) * kDim;
    uint32_t aq[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        aq[i][0] = *reinterpret_cast<const uint32_t*>(q0p + 16 * i + c2);
        aq[i][1] = *reinterpret_cast<const uint32_t*>(q1p + 16 * i + c2);
        aq[i][2] = *reinterpret_cast<const uint32_t*>(q0p + 16 * i + 8 + c2);
        aq[i][3] = *reinterpret_cast<const uint32_t*>(q1p + 16 * i + 8 + c2);
    }

    float acc[kDRegs];
#pragma unroll
    for (int i = 0; i < kDRegs; ++i) acc[i] = 0.f;
    float mstat[2] = {-INFINITY, -INFINITY};
    float lstat[2] = {0.f, 0.f};

    const __nv_bfloat16* kg = k + grp * kDim;
    const __nv_bfloat16* vg = v + grp * kDim;

    // ---- prologue: fill every stage but the last ----
#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s <= kvl) {
            char* pd = st + s * kStageB;
            cpa_tile(kg + (size_t)s * kBlockN * kKVRowStride, pd, tid);
            cpa_tile(vg + (size_t)s * kBlockN * kKVRowStride, pd + kTileB, tid);
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.commit_group;\n" ::);
        }
    }
    __syncthreads();

    for (int j = 0; j <= kvl; ++j) {
        // Own cp.async groups for tile j landed. The barrier publishes them to
        // the whole CTA: every thread reads the tile through wgmma, not only
        // the bytes it copied itself.
        asm volatile("cp.async.wait_group 2;\n" ::);
        __syncthreads();
        const char* kt = st + (j % kStages) * kStageB;
        const uint64_t kd = core_desc(kt, kLboB, kSboB);
        const uint64_t vd = core_desc(kt + kTileB, kSboB, kLboB);

        // ---- S = Q * K^T ----
        float sc[kDRegs];
#pragma unroll
        for (int i = 0; i < kDRegs; ++i) sc[i] = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) wgmma<0>(sc, aq[i], kd + kKStep * i);
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::);
        asm volatile("wgmma.fence.sync.aligned;\n" ::);

        // The tile two iterations ahead lands in the buffer the previous
        // iteration's P*V has just released; its global latency hides behind
        // the Q*K^T wgmma still running on the tensor pipe.
        // O_{j-1} is done, so its V buffer is free to be refilled asynchronously.
        asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::);
        // Every thread's in-flight wgmma on the stage below is now retired, so
        // the stage is free to be refilled; the trailing barrier of the loop
        // no longer carries that duty.
        __syncthreads();
        const int pf = j + kStages - 1;
        if (pf <= kvl) {
            char* pd = st + (pf % kStages) * kStageB;
            cpa_tile(kg + (size_t)pf * kBlockN * kKVRowStride, pd, tid);
            cpa_tile(vg + (size_t)pf * kBlockN * kKVRowStride, pd + kTileB, tid);
        } else {
            asm volatile("cp.async.commit_group;\n" ::);
            asm volatile("cp.async.commit_group;\n" ::);
        }
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::);

        // ---- causal mask, only on the tile that straddles the diagonal ----
        if (j == kvl) {
            const int lim0 = q0 + (r0 & 31);
            const int lim1 = q0 + (r1 & 31);
            const int base = j * kBlockN + c2;
#pragma unroll
            for (int g = 0; g < 16; ++g) {
                const int c = base + 8 * g;
                sc[4 * g + 0] = (c <= lim0) ? sc[4 * g + 0] : -INFINITY;
                sc[4 * g + 1] = (c + 1 <= lim0) ? sc[4 * g + 1] : -INFINITY;
                sc[4 * g + 2] = (c <= lim1) ? sc[4 * g + 2] : -INFINITY;
                sc[4 * g + 3] = (c + 1 <= lim1) ? sc[4 * g + 3] : -INFINITY;
            }
        }

        // ---- online softmax statistics ----
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int g = 0; g < 16; ++g) {
            mx0 = fmaxf(mx0, fmaxf(sc[4 * g + 0], sc[4 * g + 1]));
            mx1 = fmaxf(mx1, fmaxf(sc[4 * g + 2], sc[4 * g + 3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(kFull, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(kFull, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(kFull, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(kFull, mx1, 2));
        const float nm0 = fmaxf(mstat[0], mx0 * kScaleL2e);
        const float nm1 = fmaxf(mstat[1], mx1 * kScaleL2e);
        const float alpha0 = ex2(mstat[0] - nm0);
        const float alpha1 = ex2(mstat[1] - nm1);
        mstat[0] = nm0;
        mstat[1] = nm1;

        // ---- weights and row sums ----
        float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
        for (int g = 0; g < 16; ++g) {
            const float p0 = ex2(fmaf(sc[4 * g + 0], kScaleL2e, -nm0));
            const float p1 = ex2(fmaf(sc[4 * g + 1], kScaleL2e, -nm0));
            const float p2 = ex2(fmaf(sc[4 * g + 2], kScaleL2e, -nm1));
            const float p3 = ex2(fmaf(sc[4 * g + 3], kScaleL2e, -nm1));
            sc[4 * g + 0] = p0; sc[4 * g + 1] = p1;
            sc[4 * g + 2] = p2; sc[4 * g + 3] = p3;
            sum0 += p0 + p1;
            sum1 += p2 + p3;
        }
        sum0 += __shfl_xor_sync(kFull, sum0, 1);
        sum0 += __shfl_xor_sync(kFull, sum0, 2);
        sum1 += __shfl_xor_sync(kFull, sum1, 1);
        sum1 += __shfl_xor_sync(kFull, sum1, 2);
        lstat[0] = lstat[0] * alpha0 + sum0;
        lstat[1] = lstat[1] * alpha1 + sum1;

        // Rescaling is a no-op while the running maximum stays put.
        if (alpha0 != 1.f) {
#pragma unroll
            for (int g = 0; g < 16; ++g) { acc[4 * g + 0] *= alpha0; acc[4 * g + 1] *= alpha0; }
        }
        if (alpha1 != 1.f) {
#pragma unroll
            for (int g = 0; g < 16; ++g) { acc[4 * g + 2] *= alpha1; acc[4 * g + 3] *= alpha1; }
        }

        // ---- O += P * V ----
        asm volatile("wgmma.fence.sync.aligned;\n" ::);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t pf4[4] = {pack_bf16(sc[8 * i + 0], sc[8 * i + 1]),
                                     pack_bf16(sc[8 * i + 2], sc[8 * i + 3]),
                                     pack_bf16(sc[8 * i + 4], sc[8 * i + 5]),
                                     pack_bf16(sc[8 * i + 6], sc[8 * i + 7])};
            wgmma<1>(acc, pf4, vd + kVStep * i);
        }
        asm volatile("wgmma.commit_group.sync.aligned;\n" ::);
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::);

    // ---- normalise and store ----
    const float inv0 = 1.f / lstat[0];
    const float inv1 = 1.f / lstat[1];
    __nv_bfloat16* o0p = o + ((size_t)(q0 + (r0 & 31)) * kHeadsQ + grp * kGroup + 2 * wg + (r0 >> 5)) * kDim;
    __nv_bfloat16* o1p = o + ((size_t)(q0 + (r1 & 31)) * kHeadsQ + grp * kGroup + 2 * wg + (r1 >> 5)) * kDim;
#pragma unroll
    for (int g = 0; g < 16; ++g) {
        const __nv_bfloat162 lo = __floats2bfloat162_rn(acc[4 * g + 0] * inv0, acc[4 * g + 1] * inv0);
        const __nv_bfloat162 hi = __floats2bfloat162_rn(acc[4 * g + 2] * inv1, acc[4 * g + 3] * inv1);
        *reinterpret_cast<__nv_bfloat162*>(o0p + 8 * g + c2) = lo;
        *reinterpret_cast<__nv_bfloat162*>(o1p + 8 * g + c2) = hi;
    }
#endif
}

// ------------------------------------------------------------------- host ---
void launch(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
            torch::Tensor& o) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && o.is_cuda(),
                "gqa: inputs must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                    v.scalar_type() == at::kBFloat16 && o.scalar_type() == at::kBFloat16,
                "gqa: expected bfloat16 tensors");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                    o.is_contiguous(),
                "gqa: expected contiguous tensors");

    const int64_t seq = q.size(1);
    TORCH_CHECK(q.dim() == 4 && q.size(0) == 1 && q.size(2) == kHeadsQ &&
                    q.size(3) == kDim,
                "gqa: unexpected q shape");
    TORCH_CHECK(k.sizes() == v.sizes() && k.size(0) == 1 && k.size(1) == seq &&
                    k.size(2) == kHeadsKV && k.size(3) == kDim,
                "gqa: unexpected k/v shape");
    TORCH_CHECK(o.sizes() == q.sizes(), "gqa: unexpected o shape");
    TORCH_CHECK(seq % kRowsHead == 0, "gqa: sequence must be a multiple of 32");

    const at::cuda::CUDAGuard guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();

    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(gqa_wg2, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemB);
        configured = true;
    }

    const int qblocks = (int)(seq / kRowsHead);
    gqa_wg2<<<qblocks * kHeadsKV, kThreads, kSmemB, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(o.data_ptr<at::BFloat16>()), (int)seq, qblocks);
    C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void kernel(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
            torch::Tensor& o) {
    launch(q, k, v, o);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("kernel", &kernel); }

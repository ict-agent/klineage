#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "mma.cuh"

namespace {
constexpr int kDim = 4096;
constexpr int kBlockK = 64;
constexpr int kWarpgroup = 128;
constexpr int kThreads = G_MATH_WGS * kWarpgroup + G_PROD_WARPS * 32;
constexpr int kProducerGroup = G_PROD_LAST ? G_MATH_WGS : 0;
constexpr int kFirstMathThread = G_PROD_LAST ? 0 : kWarpgroup;
constexpr int kRowsPerGroup = G_BM / G_MATH_WGS;
constexpr int kRowTiles = kRowsPerGroup / 64;
constexpr int kStageBytes = (G_BM + G_BN) * kBlockK * 2;
constexpr int kDataBytes = G_STAGES * kStageBytes;
constexpr int kOutputBytes = G_BM * G_OUT_COLS * 2;
constexpr int kSharedBytes = kDataBytes + kOutputBytes + G_STAGES * 16;
constexpr int kTilesM = (kDim + G_BM - 1) / G_BM;
constexpr int kTilesN = kDim / G_BN;
constexpr int kTiles = kTilesM * kTilesN;
constexpr int kGrid = G_GRID == 0 ? kTiles : G_GRID;
static_assert(G_BM % (G_MATH_WGS * 64) == 0);
static_assert(G_GRID > 0);
static_assert(G_FULL_TILES >= 0 && G_FULL_TILES <= kTiles);
static_assert(G_BN % 128 == 0);
static_assert(G_OUT_COLS % 64 == 0 && G_OUT_COLS <= G_BN / 2);
static_assert(G_WAIT < G_STAGES && G_WAIT < 8);
static_assert(kSharedBytes <= 232448);
static_assert(G_PROD_LAST || G_PROD_WARPS == 4);
static_assert(kDim % G_BM == 0);
static_assert(G_BN == 384 && G_MATH_WGS == 2 && G_PROD_WARPS == 4);
constexpr int kRegistersPerSm = 65536;
constexpr int kRegisterStep = 8;
constexpr int kInitialRegisters = (kRegistersPerSm / kThreads / kRegisterStep) * kRegisterStep;
static_assert((G_REG_MATH * G_MATH_WGS + G_REG_LOAD) * kWarpgroup <= kInitialRegisters * kThreads);
constexpr int kEdgeWidth = kDim % G_BN;
constexpr int kPanelRows = G_BN / 2;
constexpr int kEdgeHeavyCtas = 10;
constexpr int kLightCtas = 2;
constexpr int kTwoFullLimit = kGrid - kEdgeHeavyCtas;
constexpr int kThreeFullCtas = kTwoFullLimit - kLightCtas;
constexpr int kFirstEdges = kGrid - kThreeFullCtas;
static_assert(kEdgeWidth == 256);
static_assert(G_BN % G_OUT_COLS == 0 && kEdgeWidth % G_OUT_COLS == 0);
static_assert(kGrid + kTwoFullLimit + kThreeFullCtas == kTiles);
static_assert(kFirstEdges + 2 * kEdgeHeavyCtas == kTilesM);
static_assert(G_BM * G_BN * 2 <= kDataBytes);

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void init_bar(uint32_t bar, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void wait_bar(uint32_t bar, int phase) {
    asm volatile("{ .reg .pred p; wait_loop: "
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; "
                 "@!p bra.uni wait_loop; }" :: "r"(bar), "r"(phase) : "memory");
}

__device__ __forceinline__ void arrive(uint32_t bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar) : "memory");
}

template<int Bytes>
__device__ __forceinline__ void expect(uint32_t bar) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(bar), "n"(Bytes) : "memory");
}

template<int Mode, int Fraction, int Fallback>
__device__ __forceinline__ uint64_t cache_policy() {
    static_assert(Fraction > 0 && Fraction <= 100);
    static_assert(Fallback == 0 || Fallback == 1);
    constexpr float kFraction = Fraction / 100.0f;
    uint64_t policy = 0;
    if constexpr (Mode == 1) {
        if constexpr (Fallback == 0) {
            asm("createpolicy.fractional.L2::evict_first.b64 %0, %1;"
                : "=l"(policy) : "f"(kFraction));
        }
        if constexpr (Fallback == 1) {
            asm("createpolicy.fractional.L2::evict_first.L2::evict_first.b64 %0, %1;"
                : "=l"(policy) : "f"(kFraction));
        }
    }
    if constexpr (Mode == 2) {
        if constexpr (Fallback == 0) {
            asm("createpolicy.fractional.L2::evict_last.b64 %0, %1;"
                : "=l"(policy) : "f"(kFraction));
        }
        if constexpr (Fallback == 1) {
            asm("createpolicy.fractional.L2::evict_last.L2::evict_first.b64 %0, %1;"
                : "=l"(policy) : "f"(kFraction));
        }
    }
    if constexpr (Mode == 3) {
        if constexpr (Fallback == 0) {
            asm("createpolicy.fractional.L2::evict_normal.b64 %0, %1;"
                : "=l"(policy) : "f"(kFraction));
        }
        if constexpr (Fallback == 1) {
            asm("createpolicy.fractional.L2::evict_normal.L2::evict_first.b64 %0, %1;"
                : "=l"(policy) : "f"(kFraction));
        }
    }
    if constexpr (Mode == 4) {
        if constexpr (Fallback == 0) {
            asm("createpolicy.fractional.L2::evict_unchanged.b64 %0, %1;"
                : "=l"(policy) : "f"(kFraction));
        }
        if constexpr (Fallback == 1) {
            asm("createpolicy.fractional.L2::evict_unchanged.L2::evict_first.b64 %0, %1;"
                : "=l"(policy) : "f"(kFraction));
        }
    }
    return policy;
}

template<int Mode, int Fraction, int Fallback>
__device__ __forceinline__ void load_tile(uint32_t dst, const CUtensorMap* map,
                                         int col, int row, uint32_t bar) {
    if constexpr (Mode == 0) {
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
                     "[%0], [%1, {%2, %3}], [%4];"
                     :: "r"(dst), "l"(map), "r"(col), "r"(row), "r"(bar) : "memory");
    } else {
        const uint64_t policy = cache_policy<Mode, Fraction, Fallback>();
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
                     "[%0], [%1, {%2, %3}], [%4], %5;"
                     :: "r"(dst), "l"(map), "r"(col), "r"(row), "r"(bar), "l"(policy) : "memory");
    }
}

template<int Mode, int Fraction, int Fallback>
__device__ __forceinline__ void load_wide(uint32_t dst, const CUtensorMap* map,
                                         int col, int row, uint32_t bar) {
    if constexpr (Mode == 0) {
        asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
                     "[%0], [%1, {%2, %3, 0}], [%4];"
                     :: "r"(dst), "l"(map), "r"(col), "r"(row), "r"(bar) : "memory");
    } else {
        const uint64_t policy = cache_policy<Mode, Fraction, Fallback>();
        asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
                     "[%0], [%1, {%2, %3, 0}], [%4], %5;"
                     :: "r"(dst), "l"(map), "r"(col), "r"(row), "r"(bar), "l"(policy) : "memory");
    }
}

__device__ __forceinline__ void store_tile(uint32_t src, const CUtensorMap* map,
                                          int col, int row) {
    const unsigned tid = threadIdx.x % kWarpgroup;
    if constexpr (G_CACHE_C == 0) {
        asm volatile("{ .reg .pred p; setp.eq.u32 p, %4, %5; "
                     "@p cp.async.bulk.tensor.3d.global.shared::cta.bulk_group "
                     "[%0, {0, %2, %1}], [%3]; }"
                     :: "l"(map), "r"(col), "r"(row), "r"(src),
                        "r"(tid), "n"(0) : "memory");
    } else {
        const uint64_t policy = cache_policy<G_CACHE_C, G_FRACTION_C, G_FALLBACK_C>();
        asm volatile("{ .reg .pred p; setp.eq.u32 p, %5, %6; "
                     "@p cp.async.bulk.tensor.3d.global.shared::cta.bulk_group.L2::cache_hint "
                     "[%0, {0, %2, %1}], [%3], %4; }"
                     :: "l"(map), "r"(col), "r"(row), "r"(src), "l"(policy),
                        "r"(tid), "n"(0) : "memory");
    }
}

__device__ __forceinline__ uint32_t pack_bf16(float a, float b) {
    uint32_t result;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(result) : "f"(a), "f"(b));
    return result;
}

__device__ __forceinline__ void store_matrix(uint32_t dst, uint32_t v0,
                                            uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
                 :: "r"(dst), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

__device__ __forceinline__ void math_barrier() {
    const unsigned group = __shfl_sync(0xffffffffu, threadIdx.x / kWarpgroup, 0);
    const unsigned math = group - (G_PROD_LAST ? 0 : 1);
    asm volatile("bar.sync %0, %1;" :: "r"(1 + math), "n"(kWarpgroup) : "memory");
}

__device__ __forceinline__ uint64_t matrix_desc(uint32_t ptr) {
    constexpr uint64_t kSwizzle128 = 1ull << 62;
    constexpr uint64_t kStrideEightRows = 64ull << 32;
    constexpr uint64_t kLeadingOffset = 1ull << 16;
    return kSwizzle128 | kStrideEightRows | kLeadingOffset | (ptr >> 4);
}

__device__ __forceinline__ void tile_coords(int tile, int& pm, int& pn) {
    const unsigned group = tile / (G_GROUP * kTilesN);
    const int first = group * G_GROUP;
    const int size = min(G_GROUP, kTilesM - first);
    const int inner = tile % (G_GROUP * kTilesN);
    pm = first + inner % size;
    pn = inner / size;
}

struct Ring {
    unsigned stage;
    unsigned phase;
    unsigned count;
};

__device__ __forceinline__ void advance_ring(Ring& state) {
    ++state.count;
    ++state.stage;
    if (state.stage == G_STAGES) {
        state.stage = 0;
        state.phase ^= 1;
    }
}

template<int Width>
__device__ __forceinline__ void load_worker(const CUtensorMap* aMap,
                                           const CUtensorMap* bMap, int m0, int n0,
                                           Ring& state, uint32_t base,
                                           uint32_t ready, uint32_t free) {
    #pragma unroll 1
    for (int kb = 0; kb < kDim / kBlockK; ++kb, advance_ring(state)) {
        const int s = state.stage;
        if (state.count >= G_STAGES) wait_bar(free + s * 8, state.phase ^ 1);
        const uint32_t bar = ready + s * 8;
        const uint32_t dst = base + s * kStageBytes;
        expect<(G_BM + Width) * kBlockK * 2>(bar);
        load_tile<G_CACHE_A, G_FRACTION_A, G_FALLBACK_A>(dst, aMap, kb * kBlockK, m0, bar);
        if constexpr (Width == G_BN) {
            load_wide<G_CACHE_B, G_FRACTION_B, G_FALLBACK_B>(dst + G_BM * kBlockK * 2, bMap, kb * kBlockK, n0, bar);
        } else {
            load_tile<G_CACHE_B, G_FRACTION_B, G_FALLBACK_B>(dst + G_BM * kBlockK * 2, bMap, kb * kBlockK, n0, bar);
        }
    }
}

template<int Width>
__device__ __forceinline__ void math_worker(const CUtensorMap* cMap, int m0, int n0,
                                           Ring& state, uint32_t base, uint32_t output,
                                           uint32_t ready, uint32_t free, unsigned group) {
    const unsigned tid = threadIdx.x;
    const unsigned math = group - (G_PROD_LAST ? 0 : 1);
    const unsigned lane = tid % 32;
    const unsigned warp = (tid % kWarpgroup) / 32;
    unsigned release = state.stage;
    float d[kRowTiles][Width / 2] = {};
    #pragma unroll
    for (int r = 0; r < kRowTiles; ++r) mma_fence(d[r]);
    #pragma unroll 1
    for (int kb = 0; kb < kDim / kBlockK; ++kb, advance_ring(state)) {
        const int s = state.stage;
        wait_bar(ready + s * 8, state.phase);
        const uint32_t ptr = base + s * kStageBytes;
        const uint64_t b_base = matrix_desc(ptr + G_BM * kBlockK * 2);
        uint64_t a_base[kRowTiles];
        #pragma unroll
        for (int r = 0; r < kRowTiles; ++r) {
            a_base[r] = matrix_desc(ptr + (math * kRowsPerGroup + r * 64) * kBlockK * 2);
        }
        #pragma unroll
        for (int kk = 0; kk < kBlockK; kk += 16) {
            constexpr int kDescriptorUnit = 16;
            const uint64_t offset = kk * 2 / kDescriptorUnit;
            #pragma unroll
            for (int r = 0; r < kRowTiles; ++r) mma(d[r], a_base[r] + offset, b_base + offset);
        }
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        #pragma unroll
        for (int r = 0; r < kRowTiles; ++r) mma_wait<G_WAIT>(d[r]);
        if (kb >= G_WAIT) {
            if (tid % kWarpgroup == 0) arrive(free + release * 8);
            ++release;
            if (release == G_STAGES) release = 0;
        }
    }
    #pragma unroll
    for (int r = 0; r < kRowTiles; ++r) mma_wait<0>(d[r]);
    #pragma unroll
    for (int i = 0; i < G_WAIT; ++i) {
        if (tid % kWarpgroup == 0) arrive(free + release * 8);
        ++release;
        if (release == G_STAGES) release = 0;
    }
    #pragma unroll
    for (int segment = 0; segment < Width / G_OUT_COLS; ++segment) {
        asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
        math_barrier();
        #pragma unroll
        for (int r = 0; r < kRowTiles; ++r) {
            const unsigned row = warp * 16 + (lane % 16) + r * 64;
            #pragma unroll
            for (int j = 0; j < G_OUT_COLS / 16; ++j) {
                const unsigned col = (lane / 16) * 8 + j * 16;
                const unsigned panel = col / 64;
                const unsigned swizzled = ((col % 64) * 2) ^ ((row % 8) * 16);
                const uint32_t dst = output + math * kRowsPerGroup * G_OUT_COLS * 2
                                     + panel * kRowsPerGroup * 128 + row * 128 + swizzled;
                const unsigned index = (segment * (G_OUT_COLS / 16) + j) * 8;
                store_matrix(dst, pack_bf16(d[r][index], d[r][index + 1]),
                             pack_bf16(d[r][index + 2], d[r][index + 3]),
                             pack_bf16(d[r][index + 4], d[r][index + 5]),
                             pack_bf16(d[r][index + 6], d[r][index + 7]));
            }
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        math_barrier();
        store_tile(output + math * kRowsPerGroup * G_OUT_COLS * 2, cMap,
                   (n0 + segment * G_OUT_COLS) / 64, m0 + math * kRowsPerGroup);
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    }

}

enum class Role { Load, Math };

template<int Width, Role Task>
__device__ __forceinline__ void run_tile(const CUtensorMap* aMap, const CUtensorMap* bMap,
                                        const CUtensorMap* cMap, int m0, int n0, Ring& state,
                                        uint32_t base, uint32_t output, uint32_t ready,
                                        uint32_t free, unsigned group) {
    if constexpr (Task == Role::Load) {
        load_worker<Width>(aMap, bMap, m0, n0, state, base, ready, free);
    } else {
        math_worker<Width>(cMap, m0, n0, state, base, output, ready, free, group);
    }
}

template<Role Task>
__device__ __forceinline__ void work(const CUtensorMap* aMap, const CUtensorMap* bMap,
                                    const CUtensorMap* sMap, const CUtensorMap* cMap,
                                    const CUtensorMap* tMap, uint32_t base, uint32_t output,
                                    uint32_t ready, uint32_t free, unsigned group) {
    const unsigned cta = blockIdx.x;
    Ring state = {};
    #pragma unroll 1
    for (int ordinal = 0; ordinal < 3; ++ordinal) {
        if (ordinal == 1 && cta >= kTwoFullLimit) continue;
        if (ordinal == 2 && cta >= kThreeFullCtas) continue;
        const unsigned tile = cta + (ordinal == 0 ? 0 : ordinal == 1 ? kGrid : kGrid + kTwoFullLimit);
        int pm, pn;
        tile_coords(tile, pm, pn);
        run_tile<G_BN, Task>(aMap, bMap, cMap, pm * G_BM, pn * G_BN,
                             state, base, output, ready, free, group);
    }
    if (cta < kThreeFullCtas) return;
    run_tile<kEdgeWidth, Task>(aMap, sMap, tMap, (cta - kThreeFullCtas) * G_BM, kTilesN * G_BN,
                              state, base, output, ready, free, group);
    if (cta < kTwoFullLimit) return;
    #pragma unroll 1
    for (int ordinal = 0; ordinal < 2; ++ordinal) {
        const unsigned row = kFirstEdges + ordinal * kEdgeHeavyCtas + cta - kTwoFullLimit;
        run_tile<kEdgeWidth, Task>(aMap, sMap, tMap, row * G_BM, kTilesN * G_BN,
                                  state, base, output, ready, free, group);
    }
}

__global__ __launch_bounds__(kThreads, 1)
void raw_gemm(const __grid_constant__ CUtensorMap aMap,
              const __grid_constant__ CUtensorMap bMap,
              const __grid_constant__ CUtensorMap sMap,
              const __grid_constant__ CUtensorMap cMap,
              const __grid_constant__ CUtensorMap tMap) {
    extern __shared__ __align__(1024) unsigned char storage[];
    const uint32_t base = smem_addr(storage);
    const uint32_t output = base + kDataBytes;
    const uint32_t ready = output + kOutputBytes;
    const uint32_t free = ready + G_STAGES * 8;
    const unsigned tid = threadIdx.x;
    const unsigned group = __shfl_sync(0xffffffffu, tid / kWarpgroup, 0);
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < G_STAGES; ++s) {
            init_bar(ready + s * 8, 1);
            init_bar(free + s * 8, G_MATH_WGS);
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    __syncthreads();

    if (group == kProducerGroup) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" :: "n"(G_REG_LOAD) : "memory");
        if (tid != kProducerGroup * kWarpgroup) return;
        work<Role::Load>(&aMap, &bMap, &sMap, &cMap, &tMap, base, output, ready, free, group);
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(G_REG_MATH) : "memory");
    work<Role::Math>(&aMap, &bMap, &sMap, &cMap, &tMap, base, output, ready, free, group);
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

CUresult encode(CUtensorMap& map, void* ptr, uint32_t rows) {
    const uint64_t dims[2] = {kDim, kDim};
    const uint64_t strides[1] = {kDim * 2};
    const uint32_t box[2] = {kBlockK, rows};
    const uint32_t steps[2] = {1, 1};
    return cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr,
        dims, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, static_cast<CUtensorMapL2promotion>(G_L2_PROMO),
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
CUresult encode_output(CUtensorMap& map, void* ptr, uint32_t width, uint64_t row_stride) {
    const uint64_t dims[3] = {64, kDim, kDim / 64};
    const uint64_t strides[2] = {row_stride * 2, 128};
    const uint32_t box[3] = {64, kRowsPerGroup, min(width, static_cast<uint32_t>(G_OUT_COLS)) / 64};
    const uint32_t steps[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, ptr,
        dims, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, static_cast<CUtensorMapL2promotion>(G_L2_PROMO),
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
CUresult encode_wide(CUtensorMap& map, void* ptr) {
    const uint64_t dims[3] = {kDim, kDim - kPanelRows, 2};
    const uint64_t strides[2] = {kDim * 2, kPanelRows * kDim * 2};
    const uint32_t box[3] = {kBlockK, kPanelRows, 2};
    const uint32_t steps[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, ptr,
        dims, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, static_cast<CUtensorMapL2promotion>(G_L2_PROMO),
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}
} // namespace

extern "C" int launch(void* x, void* w, void* y, void* stream, uint64_t row_stride) {
    CUtensorMap aMap, bMap, sMap, cMap, tMap;
    CUresult result = encode(aMap, x, G_BM);
    if (result != CUDA_SUCCESS) return -static_cast<int>(result);
    result = encode_wide(bMap, w);
    if (result != CUDA_SUCCESS) return -static_cast<int>(result);
    result = encode(sMap, w, kEdgeWidth);
    if (result != CUDA_SUCCESS) return -static_cast<int>(result);
    result = encode_output(cMap, y, G_BN, row_stride);
    if (result != CUDA_SUCCESS) return -static_cast<int>(result);
    result = encode_output(tMap, y, kEdgeWidth, row_stride);
    if (result != CUDA_SUCCESS) return -static_cast<int>(result);
    cudaError_t error = cudaFuncSetAttribute(raw_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
    if (error != cudaSuccess) return static_cast<int>(error);
    constexpr int grid = kGrid < kTiles ? kGrid : kTiles;
    raw_gemm<<<grid, kThreads, kSharedBytes, static_cast<cudaStream_t>(stream)>>>(
        aMap, bMap, sMap, cMap, tMap);
    return static_cast<int>(cudaGetLastError());
}

extern "C" const char* error_text(int code) {
    if (code >= 0) return cudaGetErrorString(static_cast<cudaError_t>(code));
    const char* message = nullptr;
    cuGetErrorString(static_cast<CUresult>(-code), &message);
    return message;
}

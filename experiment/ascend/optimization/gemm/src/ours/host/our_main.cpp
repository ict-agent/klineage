// Standalone runner for the hand-written GEMM kernel (BF16).
// Usage: ./our_matmul M N K [deviceId] [launches]
// Env: SKIP_VERIFY=1 skip CPU golden, DUMP_C=<path> write C, OUR_SWZ / OUR_DIR
//      tune the block rasterization.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "kernel_operator.h"
#include "tiling/platform/platform_ascendc.h"
#include "our_matmul_tiling.h"

extern "C" __global__ __aicore__ void our_matmul(GM_ADDR a, GM_ADDR b, GM_ADDR c, OurTiling tiling);

#define ACL_CHECK(expr)                                                       \
    do {                                                                      \
        aclError _ret = (expr);                                               \
        if (_ret != ACL_SUCCESS) {                                            \
            printf("ACL error %d at %s:%d\n", (int)_ret, __FILE__, __LINE__); \
            return -1;                                                        \
        }                                                                     \
    } while (0)

static uint32_t LcgNext(uint32_t &state)
{
    state = state * 1664525u + 1013904223u;
    return state;
}

static uint16_t F2B(float v)
{
    uint32_t b = 0;
    __builtin_memcpy(&b, &v, 4);
    uint32_t rounding = (b & 0xffffu) + ((b >> 16) & 1u);
    return static_cast<uint16_t>((b + rounding) >> 16);
}

static float B2F(uint16_t v)
{
    uint32_t b = static_cast<uint32_t>(v) << 16;
    float f = 0.0f;
    __builtin_memcpy(&f, &b, 4);
    return f;
}

int main(int argc, const char **argv)
{
    if (argc < 4) {
        printf("Usage: %s M N K [deviceId] [launches]\n", argv[0]);
        return -1;
    }
    uint32_t m = (uint32_t)atoi(argv[1]);
    uint32_t n = (uint32_t)atoi(argv[2]);
    uint32_t k = (uint32_t)atoi(argv[3]);
    int32_t deviceId = argc > 4 ? atoi(argv[4]) : 0;
    uint32_t launches = argc > 5 ? (uint32_t)atoi(argv[5]) : 1;
    if (launches < 1) {
        printf("launches must be >= 1\n");
        return -1;
    }

    // The kernel copies whole 16-element rows/columns and K blocks.
    if (m % 16 != 0 || n % 16 != 0 || k % 16 != 0) {
        printf("M, N, K must be multiples of 16\n");
        return -1;
    }
    if (k >= 65536 || n >= 65536) {
        printf("K and N must stay below 65536\n");
        return -1;
    }

    OurTiling tiling;
    tiling.m = m;
    tiling.n = n;
    tiling.k = k;
    const char *swz = getenv("OUR_SWZ");
    const char *dir = getenv("OUR_DIR");
    if (swz != nullptr) {
        tiling.swizzle = (uint32_t)atoi(swz);
    }
    if (dir != nullptr) {
        tiling.swizzleDir = (uint32_t)atoi(dir);
    }

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    aclrtStream stream{nullptr};
    ACL_CHECK(aclrtCreateStream(&stream));

    size_t lenA = (size_t)m * k;
    size_t lenB = (size_t)k * n;
    size_t lenC = (size_t)m * n;

    std::vector<uint16_t> hostA(lenA);
    std::vector<uint16_t> hostB(lenB);
    uint32_t rng = 12345u;
    for (size_t i = 0; i < lenA; ++i) {
        float v = ((float)(LcgNext(rng) % 4000) / 1000.0f) - 2.0f;
        hostA[i] = F2B(v);
    }
    for (size_t i = 0; i < lenB; ++i) {
        float v = ((float)(LcgNext(rng) % 4000) / 1000.0f) - 2.0f;
        hostB[i] = F2B(v);
    }
    // Debug patterns: make B (or A) identity so C must reproduce A (or B),
    // which exposes any wrong fractal layout mapping.
    const char *pattern = getenv("PATTERN");
    if (pattern != nullptr && strcmp(pattern, "idB") == 0) {
        for (size_t i = 0; i < lenB; ++i) hostB[i] = F2B(0.0f);
        for (uint32_t i = 0; i < (k < n ? k : n); ++i) hostB[(size_t)i * n + i] = F2B(1.0f);
        printf("pattern: B = identity\n");
    } else if (pattern != nullptr && strcmp(pattern, "idA") == 0) {
        for (size_t i = 0; i < lenA; ++i) hostA[i] = F2B(0.0f);
        for (uint32_t i = 0; i < (m < k ? m : k); ++i) hostA[(size_t)i * k + i] = F2B(1.0f);
        printf("pattern: A = identity\n");
    }

    uint8_t *deviceA = nullptr;
    uint8_t *deviceB = nullptr;
    uint8_t *deviceC = nullptr;
    ACL_CHECK(aclrtMalloc((void **)&deviceA, lenA * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&deviceB, lenB * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)&deviceC, lenC * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(deviceA, lenA * 2, hostA.data(), lenA * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(deviceB, lenB * 2, hostB.data(), lenB * 2, ACL_MEMCPY_HOST_TO_DEVICE));

    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        printf("Get PlatformAscendC failed\n");
        return -1;
    }
    uint32_t aicNum = platform->GetCoreNumAic();
    uint32_t blocks = ((m + 127) / 128) * ((n + 255) / 256);
    uint32_t blockDim = aicNum < blocks ? aicNum : blocks;

    printf("grid: %ux%u blocks, blockDim %u, swizzle %u/%u\n", (m + 127) / 128, (n + 255) / 256,
           blockDim, tiling.swizzle, tiling.swizzleDir);

    auto launch = [&]() {
        our_matmul<<<blockDim, nullptr, stream>>>(deviceA, deviceB, deviceC, tiling);
    };

    // Launch back to back. msprof picks its window from --warm-up and
    // --launch-count, so a batch of warmup + count launches feeds it; the
    // printed number is only a host-side sanity check.
    aclrtEvent ev0 = nullptr;
    aclrtEvent ev1 = nullptr;
    ACL_CHECK(aclrtCreateEvent(&ev0));
    ACL_CHECK(aclrtCreateEvent(&ev1));
    ACL_CHECK(aclrtRecordEvent(ev0, stream));
    for (uint32_t i = 0; i < launches; ++i) {
        launch();
    }
    ACL_CHECK(aclrtRecordEvent(ev1, stream));
    ACL_CHECK(aclrtSynchronizeStream(stream));
    float ms = 0.0f;
    ACL_CHECK(aclrtEventElapsedTime(&ms, ev0, ev1));
    printf("host_batch_us: %.2f for %u launches (%.2f us/launch)\n", ms * 1000.0f, launches,
           ms * 1000.0f / launches);

    std::vector<uint16_t> hostC(lenC);
    ACL_CHECK(aclrtMemcpy(hostC.data(), lenC * 2, deviceC, lenC * 2, ACL_MEMCPY_DEVICE_TO_HOST));

    const char *dump = getenv("DUMP_C");
    if (dump != nullptr) {
        FILE *f = fopen(dump, "wb");
        if (f != nullptr) {
            fwrite(hostC.data(), 2, lenC, f);
            fclose(f);
            printf("dumped C to %s\n", dump);
        }
        char path[512];
        snprintf(path, sizeof(path), "%s.A", dump);
        f = fopen(path, "wb");
        if (f != nullptr) {
            fwrite(hostA.data(), 2, lenA, f);
            fclose(f);
        }
        snprintf(path, sizeof(path), "%s.B", dump);
        f = fopen(path, "wb");
        if (f != nullptr) {
            fwrite(hostB.data(), 2, lenB, f);
            fclose(f);
        }
    }

    const char *skipEnv = getenv("SKIP_VERIFY");
    if (skipEnv != nullptr && skipEnv[0] == '1') {
        printf("Verify skipped.\n");
    } else {
        float maxRel = 0.0f;
        size_t errCnt = 0;
        for (uint32_t i = 0; i < m; ++i) {
            for (uint32_t j = 0; j < n; ++j) {
                float acc = 0.0f;
                for (uint32_t p = 0; p < k; ++p) {
                    acc += B2F(hostA[(size_t)i * k + p]) * B2F(hostB[(size_t)p * n + j]);
                }
                float got = B2F(hostC[(size_t)i * n + j]);
                float denom = fabsf(acc) > 1.0f ? fabsf(acc) : 1.0f;
                float rel = fabsf(got - acc) / denom;
                if (rel > maxRel) {
                    maxRel = rel;
                }
                if (rel > 0.01f) {
                    ++errCnt;
                }
            }
        }
        printf("max_rel_err: %f err_cnt: %zu/%zu\n", maxRel, errCnt, lenC);
        printf(errCnt == 0 ? "Compare success.\n" : "Compare failed.\n");
    }

    ACL_CHECK(aclrtDestroyEvent(ev0));
    ACL_CHECK(aclrtDestroyEvent(ev1));
    ACL_CHECK(aclrtFree(deviceA));
    ACL_CHECK(aclrtFree(deviceB));
    ACL_CHECK(aclrtFree(deviceC));
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());
    return 0;
}

#ifndef OUR_MATMUL_TILING_H
#define OUR_MATMUL_TILING_H

#include <cstdint>

// Host-computed launch parameters, passed to the kernel by value
// (direct-launch mode cannot use REGISTER_TILING_DEFAULT).
struct OurTiling {
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    // Block rasterization window (blocks along the inner axis) and direction:
    // 0 = walk M inside a window of N, 1 = walk N inside a window of M.
    // A window of one M row keeps the whole A panel hot in L2 for its N row.
    uint32_t swizzle = 1;
    uint32_t swizzleDir = 0;
};

#endif

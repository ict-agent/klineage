#ifndef FMHA_FIXED_UB_LAYOUT_HPP
#define FMHA_FIXED_UB_LAYOUT_HPP

#include "../../../fmha_tile_config.hpp"

// Shared contract between online softmax and the resident-output epilogue.
// All offsets are bytes. This layout is specific to Q256/KV512/D128 on 910B.
namespace NpuArch::Epilogue::Block {
struct FmhaFixedUbLayout {
    static constexpr uint32_t KIB = 1024;
    static constexpr uint32_t Q_ROWS = KernelCommon::FmhaTileConfig::Q_TILE;
    static constexpr uint32_t ROWS_PER_AIV = Q_ROWS / 2;
    static constexpr uint32_t HEAD_DIM = KernelCommon::FmhaTileConfig::DIM;
    static constexpr uint32_t STORE_ROWS = 64;
    static constexpr uint32_t STATS_ROWS = ROWS_PER_AIV;
    static constexpr uint32_t PIPE_SLOTS = KernelCommon::FmhaTileConfig::SLOTS;

    // S and P retain their original double buffers. They are live while
    // running O survives across all KV iterations of this Q task.
    static constexpr uint32_t SCORE = 0;
    static constexpr uint32_t PROBABILITY = 64 * KIB;
    static constexpr uint32_t OUTPUT_STAGING = 96 * KIB;
    static constexpr uint32_t RUNNING_OUTPUT = 112 * KIB;
    static constexpr uint32_t BROADCAST = 176 * KIB;
    static constexpr uint32_t LOCAL_MAX = BROADCAST + 8 * KIB;
    static constexpr uint32_t HISTORY_MAX = BROADCAST + 9 * KIB;
    static constexpr uint32_t GLOBAL_MAX = BROADCAST + 10 * KIB;
    static constexpr uint32_t LOCAL_SUM = BROADCAST + 11 * KIB;
    static constexpr uint32_t GLOBAL_SUM = BROADCAST + 12 * KIB;
    static constexpr uint32_t DELTA_MAX = BROADCAST + 13 * KIB;
    static constexpr uint32_t END = DELTA_MAX + PIPE_SLOTS * STATS_ROWS * sizeof(float);

    static_assert(OUTPUT_STAGING + STORE_ROWS * HEAD_DIM * sizeof(half) == RUNNING_OUTPUT,
                  "FP16 staging must not overlap persistent O");
    static_assert(RUNNING_OUTPUT + ROWS_PER_AIV * HEAD_DIM * sizeof(float) == BROADCAST,
                  "Persistent O must not overlap softmax statistics");
    static_assert(END <= 192 * KIB, "The shared layout must fit the 910B UB");
};
} // namespace NpuArch::Epilogue::Block
#endif

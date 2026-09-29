#ifndef FMHA_TILE_CONFIG_HPP
#define FMHA_TILE_CONFIG_HPP

namespace KernelCommon {
// The wrapper validates the resident-path ABI. Hardware tile geometry and
// workspace strides stay together; batch, heads and sequence are runtime data.
struct FmhaTileConfig {
    static constexpr uint32_t DIM = 128;
    static constexpr uint32_t Q_TILE = 256;
    static constexpr uint32_t KV_TILE = 512;
    static constexpr uint32_t AIC_COUNT = 24;
    static constexpr uint32_t LOOKAHEAD = 2;
    static constexpr uint32_t SLOTS = LOOKAHEAD + 1;
    static constexpr uint64_t SCORE_SLOT_ELEMENTS = Q_TILE * KV_TILE;
    static constexpr uint64_t OUTPUT_SLOT_ELEMENTS = Q_TILE * DIM;
    static constexpr uint64_t SCORE_BYTES = AIC_COUNT * SLOTS * SCORE_SLOT_ELEMENTS * sizeof(float);
    static constexpr uint64_t PROBABILITY_BYTES = AIC_COUNT * SLOTS * SCORE_SLOT_ELEMENTS * sizeof(half);
    static constexpr uint64_t PARTIAL_OUTPUT_BYTES = AIC_COUNT * SLOTS * OUTPUT_SLOT_ELEMENTS * sizeof(float);
    static constexpr uint64_t WORKSPACE_BYTES = SCORE_BYTES + PROBABILITY_BYTES + PARTIAL_OUTPUT_BYTES;

    struct Task {
        uint32_t batch;
        uint32_t head;
        uint32_t queryBlock;
    };

    __aicore__ static inline Task decode(uint32_t index, uint32_t heads, uint32_t tasksPerBatch)
    {
        // Preserve the accepted head-fast ordering, without scanning ragged
        // sequence boundaries or reading cumulative lengths for each task.
        return {index / tasksPerBatch, index % heads,
                (index % tasksPerBatch) / heads};
    }

    static_assert(WORKSPACE_BYTES <= 88ULL * 1024 * 1024,
                  "Must fit the official host-allocated workspace");
};
} // namespace KernelCommon
#endif

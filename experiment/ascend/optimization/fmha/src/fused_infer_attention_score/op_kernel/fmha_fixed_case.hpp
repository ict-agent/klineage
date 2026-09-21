#ifndef FMHA_FIXED_CASE_HPP
#define FMHA_FIXED_CASE_HPP

namespace KernelCommon {
// The Python entry validates this ABI before dispatch. Keeping geometry,
// task mapping and workspace strides together prevents independently changing
// one tile dimension while leaving an old task count or buffer stride behind.
struct FmhaFixedCase {
    static constexpr uint32_t BATCH = 8;
    static constexpr uint32_t HEADS = 64;
    static constexpr uint32_t SEQUENCE = 2048;
    static constexpr uint32_t DIM = 128;
    static constexpr uint32_t Q_TILE = 256;
    static constexpr uint32_t KV_TILE = 512;
    static constexpr uint32_t AIC_COUNT = 24;
    static constexpr uint32_t LOOKAHEAD = 2;
    static constexpr uint32_t SLOTS = LOOKAHEAD + 1;
    static constexpr uint32_t Q_BLOCKS = SEQUENCE / Q_TILE;
    static constexpr uint32_t KV_BLOCKS = SEQUENCE / KV_TILE;
    static constexpr uint32_t TASKS_PER_BATCH = HEADS * Q_BLOCKS;
    static constexpr uint32_t TASKS = BATCH * TASKS_PER_BATCH;
    static constexpr uint32_t TOKENS = BATCH * SEQUENCE;
    static constexpr uint32_t TOKEN_STRIDE = HEADS * DIM;
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

    __aicore__ static inline Task decode(uint32_t index)
    {
        // Preserve the accepted head-fast ordering, without scanning ragged
        // sequence boundaries or reading cumulative lengths for each task.
        return {index / TASKS_PER_BATCH, index % HEADS,
                (index % TASKS_PER_BATCH) / HEADS};
    }

    static_assert(SEQUENCE % Q_TILE == 0 && SEQUENCE % KV_TILE == 0,
                  "This specialization has no tail blocks");
    static_assert(LOOKAHEAD > 0 && LOOKAHEAD < KV_BLOCKS,
                  "The fixed pipeline must have a nonempty steady state");
    static_assert(WORKSPACE_BYTES <= 88ULL * 1024 * 1024,
                  "Must fit the official host-allocated workspace");
};
} // namespace KernelCommon
#endif

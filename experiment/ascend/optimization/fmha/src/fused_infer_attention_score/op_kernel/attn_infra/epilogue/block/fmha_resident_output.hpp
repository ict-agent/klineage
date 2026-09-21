#ifndef FMHA_RESIDENT_OUTPUT_HPP
#define FMHA_RESIDENT_OUTPUT_HPP

#include "fmha_fixed_ub_layout.hpp"

namespace NpuArch::Epilogue::Block {

// Tile-local online-attention output state. The 128 rows owned by this AIV
// remain in UB from the first PV tile until the final normalization/store.
// The update is vectorized over all rows; only final FP16 stores are chunked.
template<class ArchTag>
class FmhaResidentOutput {
    using UB = FmhaFixedUbLayout;
    static constexpr uint32_t DIM = UB::HEAD_DIM;
    static constexpr uint32_t ROWS = UB::ROWS_PER_AIV;
    static constexpr uint32_t CHUNK = ROWS;
    static constexpr uint32_t STORE_ROWS = UB::STORE_ROWS;
    static constexpr uint32_t VEC = 64;
    static constexpr uint32_t BLOCK_FLOATS = 8;
    static constexpr uint32_t CHUNK_ELEMENTS = CHUNK * DIM;

public:
    __aicore__ inline void init(Arch::Resource<ArchTag>& resource)
    {
        static_assert(UB::END <= ArchTag::UB_SIZE, "Resident output exceeds UB");
        // Between softmax invocations, its 64 KiB score buffer is dead and can
        // receive the entire per-AIV PV result. Final stores use a separate
        // buffer so a following Q task may start softmax without aliasing DMA.
        partial = resource.ubBuf.template GetBufferByByte<float>(UB::SCORE);
        packedOutput = resource.ubBuf.template GetBufferByByte<half>(UB::OUTPUT_STAGING);
        running = resource.ubBuf.template GetBufferByByte<float>(UB::RUNNING_OUTPUT);
        broadcast = resource.ubBuf.template GetBufferByByte<float>(UB::BROADCAST);
        deltaMax = resource.ubBuf.template GetBufferByByte<float>(UB::DELTA_MAX);
        globalSum = resource.ubBuf.template GetBufferByByte<float>(UB::GLOBAL_SUM);
    }

    __aicore__ inline void operator()(
        AscendC::GlobalTensor<half> output,
        AscendC::GlobalTensor<float> input,
        uint32_t outputStride, bool first, bool last, uint32_t slot)
    {
        const uint32_t subRow = AscendC::GetSubBlockIdx() * ROWS;
        auto localInput = input[subRow * DIM];
        auto localOutput = output[subRow * outputStride];
        if (first) {
            initialize(localInput);
            return; // Four nonempty KV512 blocks are guaranteed by the entry ABI.
        }
        // Every softmax invocation drains its score-row loop before this
        // epilogue runs. S has no live values across invocations, so borrow
        // it between stages and fence both boundaries before returning it.
        scoreScratchFence();
        acquirePartial(localInput);
        update(running, 0, slot);
        if (last) {
            normalize(running, 0);
            for (uint32_t row = 0; row < ROWS; row += STORE_ROWS) {
                // Protect the compact output buffer between its two stores.
                if (row != 0) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
                }
                store(localOutput[row * outputStride], running[row * DIM], outputStride);
            }
        }
        releasePartial();
        scoreScratchFence(); // Return S only after the last vector read.
    }

private:
    // Event ownership matches the outer kernel's initial/final tokens:
    // V_MTE2:3 protects partial/input storage; MTE3_MTE2:6 protects the
    // final FP16 store buffer. MTE2_V:7 is a local handoff; V_MTE2:5
    // transfers ownership of score scratch between softmax and output update.
    __aicore__ inline void scoreScratchFence()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID5);
    }

    __aicore__ inline void acquireStorage()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
    }

    __aicore__ inline void finishLoad()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID7);
    }

    __aicore__ inline void releasePartial()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
    }

    __aicore__ inline void initialize(AscendC::GlobalTensor<float> input)
    {
        acquireStorage();
        AscendC::DataCopy(running, input,
            AscendC::DataCopyParams(1, ROWS * DIM / BLOCK_FLOATS, 0, 0));
        finishLoad();
        releasePartial();
    }

    __aicore__ inline void acquirePartial(AscendC::GlobalTensor<float> input)
    {
        acquireStorage();
        AscendC::DataCopy(partial, input,
            AscendC::DataCopyParams(1, CHUNK_ELEMENTS / BLOCK_FLOATS, 0, 0));
        finishLoad();
    }

    __aicore__ inline void broadcastRows(AscendC::LocalTensor<float> scalars)
    {
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        AscendC::Brcb(broadcast.template ReinterpretCast<uint32_t>(),
            scalars.template ReinterpretCast<uint32_t>(), CHUNK / BLOCK_FLOATS,
            AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void update(AscendC::LocalTensor<float> accumulator,
                                 uint32_t row, uint32_t slot)
    {
        broadcastRows(deltaMax[slot * UB::STATS_ROWS + row]);
        // Preserve the official FP32 multiply-then-add ordering, including
        // separate instructions: do not replace this pair with fused FMA.
        for (uint32_t col = 0; col < DIM; col += VEC) {
            AscendC::Mul<float, false>(accumulator[col], accumulator[col], broadcast,
                (uint64_t)0, CHUNK,
                AscendC::BinaryRepeatParams(1, 1, 0, DIM / BLOCK_FLOATS, DIM / BLOCK_FLOATS, 1));
        }
        AscendC::PipeBarrier<PIPE_V>();
        // Vector repeat is uint8: split 256 repeats into two 128-repeat
        // instructions, without changing each element's addition order.
        for (uint32_t offset = 0; offset < CHUNK_ELEMENTS; offset += 8192) {
            AscendC::Add<float, false>(accumulator[offset], accumulator[offset], partial[offset],
                (uint64_t)0, 8192 / VEC,
                AscendC::BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void normalize(AscendC::LocalTensor<float> accumulator, uint32_t row)
    {
        broadcastRows(globalSum[row]);
        for (uint32_t col = 0; col < DIM; col += VEC) {
            AscendC::Div<float, false>(accumulator[col], accumulator[col], broadcast,
                (uint64_t)0, CHUNK,
                AscendC::BinaryRepeatParams(1, 1, 0, DIM / BLOCK_FLOATS, DIM / BLOCK_FLOATS, 1));
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void store(AscendC::GlobalTensor<half> output,
                                AscendC::LocalTensor<float> accumulator, uint32_t outputStride)
    {
        // This 16 KiB buffer is disjoint from S and the persistent O state.
        auto packed = packedOutput;
        AscendC::Cast<half, float, false>(packed, accumulator, AscendC::RoundMode::CAST_NONE,
            (uint64_t)0, STORE_ROWS * DIM / VEC, AscendC::UnaryRepeatParams(1, 1, 4, 8));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID5);
        AscendC::DataCopyPad(output, packed,
            AscendC::DataCopyExtParams(STORE_ROWS, DIM * sizeof(half), 0,
                                      (outputStride - DIM) * sizeof(half), 0));
    }

    AscendC::LocalTensor<float> partial;
    AscendC::LocalTensor<half> packedOutput;
    AscendC::LocalTensor<float> running;
    AscendC::LocalTensor<float> broadcast;
    AscendC::LocalTensor<float> deltaMax;
    AscendC::LocalTensor<float> globalSum;
};
} // namespace NpuArch::Epilogue::Block
#endif

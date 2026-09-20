#ifndef OUR_MATMUL_KERNEL_H
#define OUR_MATMUL_KERNEL_H

#include "kernel_operator.h"
#include "our_matmul_tiling.h"

// Hand-written cube-only BF16 GEMM: C = A * B, row-major, FP32 accumulate.
//
// Pipeline: one K tile is prefetched ahead into L1, each L1 tile is consumed
// in L0_K steps, and every level runs double buffered:
//
//   GM(A) --MTE2--> L1 A [2 stages] --MTE1--> L0A [2 stages] --\
//   GM(B) --MTE2--> L1 B [2 stages] --MTE1--> L0B [2 stages] ---M--> L0C --FIX--> GM(C)
//
// One output block is BLK_M x BLK_N; its K axis is walked as L1_K tiles.
// The next K tile is prefetched into the other L1 stage while the current tile
// is computed, and the first K tile of the next output block is prefetched
// during the last K iteration (cross-block preload). Each L1 tile is consumed as
// L0_K-sized L0 steps, again double buffered. The final Mmad of a block carries
// unit flag 0b11, which drains L0C to GM through fixpipe.

namespace OurMatmul {

using Bf16 = bfloat16_t;

// ---- Tile geometry (bf16) ----
constexpr uint32_t BLK_M = 128;  // output rows per block
constexpr uint32_t BLK_N = 256;  // output cols per block
constexpr uint32_t L1_K = 256;   // K elements per L1 fill
constexpr uint32_t L0_K = 64;    // K elements per L0 fill
constexpr uint32_t STAGES = 2;   // L1 / L0 buffer depth

constexpr uint32_t C0 = 16;         // bf16 elements per 32B block
constexpr uint32_t FRAC = C0 * C0;  // elements per 16x16 fractal

constexpr uint32_t L1A_BYTES = BLK_M * L1_K * 2;  // 64 KB
constexpr uint32_t L1B_BYTES = L1_K * BLK_N * 2;  // 128 KB
constexpr uint32_t L0A_BYTES = BLK_M * L0_K * 2;  // 16 KB
constexpr uint32_t L0B_BYTES = L0_K * BLK_N * 2;  // 32 KB

// L1 fractal strides (bf16), both operands as zN (16x16 fractals):
//   A: off(m,kk) = (m/16)*256   + (kk/16)*2048 + (m%16)*16 + (kk%16)
//   B: off(kk,n) = (kk/16)*256  + (n/16)*4096  + (kk%16)*16 + (n%16)
constexpr uint32_t L1A_C0_STRIDE = BLK_M;      // C0 blocks between A 16-col groups
constexpr uint32_t L1A_ROWGAP = FRAC;          // elements between A 16-row groups
constexpr uint32_t L1B_KGAP = FRAC;            // elements between B 16-row groups
constexpr uint32_t L1B_NGAP = L1_K * C0;       // elements between B 16-col groups

// Hardware event slots; an id is only waited after it was set for the same pair.
constexpr int32_t EVT_L1A = 0;  // + stage
constexpr int32_t EVT_L1B = 2;  // + stage
constexpr int32_t EVT_L0A = 0;  // + buffer
constexpr int32_t EVT_L0B = 2;  // + buffer
constexpr int32_t EVT_MTE1_M = 0;
constexpr int32_t EVT_FIX_M = 0;

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

class Kernel {
public:
    __aicore__ inline Kernel() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, const OurTiling &t)
    {
        tiling = t;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ Bf16 *>(a), (uint64_t)t.m * t.k);
        gmB.SetGlobalBuffer(reinterpret_cast<__gm__ Bf16 *>(b), (uint64_t)t.k * t.n);
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ Bf16 *>(c), (uint64_t)t.m * t.n);

        // L1 holds the A stages then the B stages; L0A/L0B hold their stages.
        {
            AscendC::TBuf<AscendC::TPosition::A1> buf;
            GetTPipePtr()->InitBuffer(buf, STAGES * L1A_BYTES);
            auto base = buf.Get<uint8_t>();
            for (uint32_t s = 0; s < STAGES; s++) {
                l1A[s] = base[s * L1A_BYTES].ReinterpretCast<Bf16>();
            }
        }
        {
            AscendC::TBuf<AscendC::TPosition::B1> buf;
            GetTPipePtr()->InitBuffer(buf, STAGES * L1B_BYTES);
            auto base = buf.Get<uint8_t>();
            for (uint32_t s = 0; s < STAGES; s++) {
                l1B[s] = base[s * L1B_BYTES].ReinterpretCast<Bf16>();
            }
        }
        {
            AscendC::TBuf<AscendC::TPosition::A2> buf;
            GetTPipePtr()->InitBuffer(buf, STAGES * L0A_BYTES);
            auto base = buf.Get<uint8_t>();
            for (uint32_t s = 0; s < STAGES; s++) {
                l0A[s] = base[s * L0A_BYTES].ReinterpretCast<Bf16>();
            }
        }
        {
            AscendC::TBuf<AscendC::TPosition::B2> buf;
            GetTPipePtr()->InitBuffer(buf, STAGES * L0B_BYTES);
            auto base = buf.Get<uint8_t>();
            for (uint32_t s = 0; s < STAGES; s++) {
                l0B[s] = base[s * L0B_BYTES].ReinterpretCast<Bf16>();
            }
        }
        {
            AscendC::TBuf<AscendC::TPosition::CO1> buf;
            GetTPipePtr()->InitBuffer(buf, BLK_M * BLK_N * sizeof(float));
            l0C = buf.Get<float>();
        }
    }

    __aicore__ inline void Process()
    {
        // Everything starts free; the L0C accumulator starts drained.
        for (uint32_t s = 0; s < STAGES; s++) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVT_L1A + s);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVT_L1B + s);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVT_L0A + s);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVT_L0B + s);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVT_FIX_M);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t mBlocks = CeilDivU32(tiling.m, BLK_M);
        uint32_t nBlocks = CeilDivU32(tiling.n, BLK_N);
        uint32_t blocks = mBlocks * nBlocks;
        uint32_t kTiles = CeilDivU32(tiling.k, L1_K);
        uint32_t startK = coreIdx % kTiles;  // shuffleK: rotate the K start per core

        uint32_t l1Id = 0;
        for (uint32_t task = coreIdx; task < blocks; task += coreNum) {
            l1Id = RunBlock(task, coreNum, blocks, kTiles, startK, l1Id, task == coreIdx);
        }

        for (uint32_t s = 0; s < STAGES; s++) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVT_L1A + s);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVT_L1B + s);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVT_L0A + s);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVT_L0B + s);
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVT_FIX_M);
    }

private:
    // Block rasterization. swizzleDir 0 walks the N blocks of a window of
    // `swizzle` M rows, dir 1 the M blocks of a window of `swizzle` N columns;
    // every other window runs backwards so the walk stays local in L2.
    __aicore__ inline void Swizzle(uint32_t task, uint32_t &mIdx, uint32_t &nIdx)
    {
        uint32_t mBlocks = CeilDivU32(tiling.m, BLK_M);
        uint32_t nBlocks = CeilDivU32(tiling.n, BLK_N);
        if (tiling.swizzleDir == 0) {
            uint32_t winLoop = CeilDivU32(mBlocks, tiling.swizzle);
            uint32_t winIdx = task / (tiling.swizzle * nBlocks);
            uint32_t inWin = task % (tiling.swizzle * nBlocks);
            uint32_t winRows = tiling.swizzle;
            if (winIdx == winLoop - 1) {
                winRows = mBlocks - tiling.swizzle * winIdx;
            }
            mIdx = winIdx * tiling.swizzle + inWin % winRows;
            nIdx = inWin / winRows;
            if (winIdx % 2 == 1) {
                nIdx = nBlocks - nIdx - 1;
            }
            return;
        }
        uint32_t winLoop = CeilDivU32(nBlocks, tiling.swizzle);
        uint32_t winIdx = task / (tiling.swizzle * mBlocks);
        uint32_t inWin = task % (tiling.swizzle * mBlocks);
        uint32_t winCols = tiling.swizzle;
        if (winIdx == winLoop - 1) {
            winCols = nBlocks - tiling.swizzle * winIdx;
        }
        mIdx = inWin / winCols;
        nIdx = winIdx * tiling.swizzle + inWin % winCols;
        if (winIdx % 2 == 1) {
            mIdx = mBlocks - mIdx - 1;
        }
    }

    __aicore__ inline uint32_t TileK(uint32_t kIdx, uint32_t kTiles)
    {
        if (kIdx < kTiles - 1) {
            return L1_K;
        }
        return tiling.k - kIdx * L1_K;
    }

    // GM -> L1 for one K tile. A is row-major (mAct x kAct) with row stride k,
    // B is row-major (kAct x nAct) with row stride n.
    __aicore__ inline void LoadL1(uint32_t stage, uint64_t offA, uint64_t offB, uint32_t mAct,
                                  uint32_t nAct, uint32_t kIdx, uint32_t kAct)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVT_L1A + stage);
        AscendC::Nd2NzParams pa;
        pa.ndNum = 1;
        pa.nValue = (uint16_t)mAct;
        pa.dValue = (uint16_t)kAct;
        pa.srcNdMatrixStride = 0;
        pa.srcDValue = (uint16_t)tiling.k;
        pa.dstNzC0Stride = (uint16_t)L1A_C0_STRIDE;
        pa.dstNzNStride = 1;
        pa.dstNzMatrixStride = 0;
        AscendC::DataCopy(l1A[stage], gmA[offA + (uint64_t)kIdx * L1_K], pa);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVT_L1A + stage);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVT_L1B + stage);
        // B lands as one ND matrix of kAct rows: 16x16 fractals, 16 K rows
        // per fractal row, n above k. One call per tile; splitting it per 16 K
        // rows issued 16x the nd2nz instructions and stalled MTE2.
        AscendC::Nd2NzParams pb;
        pb.ndNum = 1;
        pb.nValue = (uint16_t)kAct;
        pb.dValue = (uint16_t)nAct;
        pb.srcNdMatrixStride = 0;
        pb.srcDValue = (uint16_t)tiling.n;
        pb.dstNzC0Stride = (uint16_t)(L1B_NGAP / C0);
        pb.dstNzNStride = (uint16_t)(L1B_KGAP / FRAC);
        pb.dstNzMatrixStride = 0;
        AscendC::DataCopy(l1B[stage], gmB[offB + (uint64_t)kIdx * L1_K * tiling.n], pb);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVT_L1B + stage);
    }

    // L1 -> L0A: zN(BLK_M x L1_K) -> zZ(BLK_M x kPartAct), one call per 16 rows.
    __aicore__ inline void LoadL0A(uint32_t buf, uint32_t l1Stage, uint32_t kPart, uint32_t kPartAct)
    {
        AscendC::LoadData2DParams p;
        p.startIndex = 0;
        p.repeatTimes = (uint8_t)(kPartAct / C0);
        p.srcStride = (uint16_t)(L1A_C0_STRIDE / C0);  // 2048 elements = 8 fractals
        p.sid = 0;
        p.dstGap = 0;
        p.ifTranspose = false;
        p.addrMode = 0;

        uint32_t srcBase = (kPart * L0_K / C0) * (BLK_M * C0);
        uint32_t dstStride = kPartAct * C0;
        for (uint32_t i = 0; i < BLK_M / C0; i++) {
            AscendC::LoadData(l0A[buf][i * dstStride], l1A[l1Stage][srcBase + i * L1A_ROWGAP], p);
        }
    }

    // L1 -> L0B: zN(L1_K x BLK_N) -> nZ(kPartAct x nAct), transposed per K block.
    __aicore__ inline void LoadL0B(uint32_t buf, uint32_t l1Stage, uint32_t kPart, uint32_t kPartAct,
                                   uint32_t nAct)
    {
        AscendC::LoadData2DParams p;
        p.startIndex = 0;
        p.repeatTimes = (uint8_t)(BLK_N / C0);
        p.srcStride = (uint16_t)(L1B_NGAP / FRAC);  // n groups are NGAP apart
        p.sid = 0;
        p.dstGap = 0;
        p.ifTranspose = true;
        p.addrMode = 0;

        uint32_t srcBase = (kPart * L0_K / C0) * L1B_KGAP;
        uint32_t dstStride = nAct * C0;
        for (uint32_t i = 0; i < kPartAct / C0; i++) {
            AscendC::LoadData(l0B[buf][i * dstStride], l1B[l1Stage][srcBase + i * L1B_KGAP], p);
        }
    }

    // L0C -> GM for the finished block.
    __aicore__ inline void StoreC(uint64_t offC, uint32_t mAct, uint32_t nAct)
    {
        AscendC::FixpipeParamsV220 p;
        p.nSize = (uint16_t)nAct;
        p.mSize = (uint16_t)mAct;
        p.srcStride = (uint16_t)mAct;
        p.dstStride = tiling.n;
        p.quantPre = QuantMode_t::F322BF16;
        p.reluEn = false;
        p.unitFlag = 0b11;
        AscendC::Fixpipe<Bf16, float, AscendC::CFG_ROW_MAJOR>(gmC[offC], l0C, p);
    }

    // Consume one L1 K tile through L0 steps of L0_K, L0A/L0B double buffered.
    __aicore__ inline void ComputeTile(uint32_t l1Stage, uint32_t kAct, uint32_t mAct, uint32_t nAct,
                                       uint64_t offC, bool firstK, bool lastK)
    {
        uint32_t kPartLoop = CeilDivU32(kAct, L0_K);
        uint32_t l0aId = 0;
        uint32_t l0bId = 0;
        for (uint32_t kPart = 0; kPart < kPartLoop; kPart++) {
            uint32_t kPartAct = (kPart < kPartLoop - 1) ? L0_K : (kAct - kPart * L0_K);

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVT_L0A + l0aId);
            if (kPart == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVT_L1A + l1Stage);
            }
            LoadL0A(l0aId, l1Stage, kPart, kPartAct);
            if (kPart == kPartLoop - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVT_L1A + l1Stage);
            }

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVT_L0B + l0bId);
            if (kPart == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVT_L1B + l1Stage);
            }
            LoadL0B(l0bId, l1Stage, kPart, kPartAct, nAct);
            if (kPart == kPartLoop - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVT_L1B + l1Stage);
            }

            // Both L0 loads are on MTE1; one MTE1->M flag covers them.
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVT_MTE1_M);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVT_MTE1_M);

            bool initC = firstK && (kPart == 0);

            AscendC::MmadParams mp;
            mp.m = (uint16_t)mAct;
            mp.n = (uint16_t)nAct;
            mp.k = (uint16_t)kPartAct;
            mp.unitFlag = (lastK && (kPart == kPartLoop - 1)) ? 0b11 : 0b10;
            mp.cmatrixInitVal = initC;
            AscendC::Mmad(l0C, l0A[l0aId], l0B[l0bId], mp);

            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVT_L0B + l0bId);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVT_L0A + l0aId);
            l0bId ^= 1;
            l0aId ^= 1;
        }

        if (lastK) {
            // Unit flag 0b11 on the final Mmad triggers this drain; the unit
            // flag also keeps the next block's Mmad off L0C until it is read.
            StoreC(offC, mAct, nAct);
        }
    }

    // One output block: K tiles with preload, including the cross-block preload
    // of the next block's first K tile on this core's schedule.
    __aicore__ inline uint32_t RunBlock(uint32_t task, uint32_t coreNum, uint32_t blocks,
                                        uint32_t kTiles, uint32_t startK, uint32_t l1Id, bool isFirst)
    {
        uint32_t mIdx = 0;
        uint32_t nIdx = 0;
        Swizzle(task, mIdx, nIdx);
        uint32_t mAct = MinU32(BLK_M, tiling.m - mIdx * BLK_M);
        uint32_t nAct = MinU32(BLK_N, tiling.n - nIdx * BLK_N);
        uint64_t offA = (uint64_t)mIdx * BLK_M * tiling.k;
        uint64_t offB = (uint64_t)nIdx * BLK_N;
        uint64_t offC = (uint64_t)mIdx * BLK_M * tiling.n + nIdx * BLK_N;

        if (isFirst) {
            LoadL1(l1Id, offA, offB, mAct, nAct, startK, TileK(startK, kTiles));
        }

        uint32_t nextTask = task + coreNum;
        bool hasNext = nextTask < blocks;
        uint32_t mNextAct = 0;
        uint32_t nNextAct = 0;
        uint64_t offANext = 0;
        uint64_t offBNext = 0;
        if (hasNext) {
            uint32_t mNext = 0;
            uint32_t nNext = 0;
            Swizzle(nextTask, mNext, nNext);
            mNextAct = MinU32(BLK_M, tiling.m - mNext * BLK_M);
            nNextAct = MinU32(BLK_N, tiling.n - nNext * BLK_N);
            offANext = (uint64_t)mNext * BLK_M * tiling.k;
            offBNext = (uint64_t)nNext * BLK_N;
        }

        for (uint32_t kLoop = 0; kLoop < kTiles; kLoop++) {
            uint32_t kIdx = (startK + kLoop) % kTiles;
            uint32_t nextId = l1Id ^ 1;
            bool lastK = (kLoop + 1 == kTiles);

            if (!lastK) {
                uint32_t kIdxNext = (startK + kLoop + 1) % kTiles;
                LoadL1(nextId, offA, offB, mAct, nAct, kIdxNext, TileK(kIdxNext, kTiles));
            } else if (hasNext) {
                LoadL1(nextId, offANext, offBNext, mNextAct, nNextAct, startK, TileK(startK, kTiles));
            }

            ComputeTile(l1Id, TileK(kIdx, kTiles), mAct, nAct, offC, kLoop == 0, lastK);
            l1Id = nextId;
        }
        return l1Id;
    }

    AscendC::GlobalTensor<Bf16> gmA;
    AscendC::GlobalTensor<Bf16> gmB;
    AscendC::GlobalTensor<Bf16> gmC;
    AscendC::LocalTensor<Bf16> l1A[STAGES];
    AscendC::LocalTensor<Bf16> l1B[STAGES];
    AscendC::LocalTensor<Bf16> l0A[STAGES];
    AscendC::LocalTensor<Bf16> l0B[STAGES];
    AscendC::LocalTensor<float> l0C;
    OurTiling tiling;
};

}  // namespace OurMatmul

#endif

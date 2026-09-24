#include "kernel_operator.h"

using namespace AscendC;

namespace kda {

constexpr int32_t kDim = 128;
constexpr int32_t kHeads = 96;
constexpr int32_t kTokens = 4096;
constexpr int32_t kVSplit = 2;
constexpr int32_t kRows = kDim / kVSplit;
constexpr int32_t kTB = 16;
constexpr int32_t kPerTok = kHeads * kDim;
constexpr int32_t kPerTokBlocks = kPerTok * 2 / 32;  // 32B datablock units
constexpr int32_t kStateElems = kDim * kRows;
constexpr float kNormEps = 1e-6f;

constexpr int32_t kInBf16 = 3 * kTB * kDim + kTB * kRows;
constexpr int32_t kInBytes = kInBf16 * 2;
constexpr int32_t kBetaOff = kInBytes / 4;
constexpr int32_t kBetaFloats = kTB * 8;
constexpr int32_t kInBufferBytes = kInBytes + kBetaFloats * 4 + 32;

constexpr int32_t kQF = 0;
constexpr int32_t kKF = 128;
constexpr int32_t kDF = 256;
constexpr int32_t kVF = 384;
constexpr int32_t kBF = 448;
constexpr int32_t kPF = 576;
constexpr int32_t kOF = 640;
constexpr int32_t kLF = 704;
constexpr int32_t kA0 = 768;
constexpr int32_t kA1 = 776;
constexpr int32_t kA2 = 784;
constexpr int32_t kAC = 792;
constexpr int32_t kTF = 800;
constexpr int32_t kBA = 928;
constexpr int32_t kRW = 1056;

constexpr int32_t kChunk = 128;
constexpr int32_t kChunkLanes = kChunk * kRows;
constexpr int32_t kBP = 1600;
constexpr int32_t kBT = 2624;
constexpr int32_t kWK = 10816;
constexpr int32_t kWQ = 10944;
constexpr int32_t kBC = 11072;
constexpr int32_t kWorkFloats = 19264;

__aicore__ inline void syncS2V(TPipe& pipe) {
    const event_t event = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(event);
    WaitFlag<HardEvent::S_V>(event);
}

__aicore__ inline void syncV2S(TPipe& pipe) {
    const event_t event = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(event);
    WaitFlag<HardEvent::V_S>(event);
}

__aicore__ inline void syncMte2ToS(TPipe& pipe) {
    const event_t event = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(event);
    WaitFlag<HardEvent::MTE2_S>(event);
}

__aicore__ inline void syncMte2ToV(TPipe& pipe) {
    const event_t event = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(event);
    WaitFlag<HardEvent::MTE2_V>(event);
}

__aicore__ inline void syncS2Mte3(TPipe& pipe) {
    const event_t event = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(event);
    WaitFlag<HardEvent::S_MTE3>(event);
}

}  // namespace kda

extern "C" __global__ __aicore__ void chunk_kda_fwd(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR scale, GM_ADDR a_log, GM_ADDR dt_bias, GM_ADDR lower_bound,
    GM_ADDR initial_state, GM_ADDR output, GM_ADDR final_state,
    GM_ADDR workspace, GM_ADDR tiling) {
    using namespace kda;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workspace;
    (void)tiling;

    const int32_t bid = static_cast<int32_t>(GetBlockIdx());
    const int32_t head = bid / kVSplit;
    const int32_t slice = bid - head * kVSplit;
    if (head >= kHeads) {
        return;
    }

    GlobalTensor<bfloat16_t> qGm;
    GlobalTensor<bfloat16_t> kGm;
    GlobalTensor<bfloat16_t> vGm;
    GlobalTensor<bfloat16_t> gGm;
    GlobalTensor<bfloat16_t> outGm;
    GlobalTensor<float> betaGm;
    GlobalTensor<float> scaleGm;
    GlobalTensor<float> aLogGm;
    GlobalTensor<float> biasGm;
    GlobalTensor<float> lowerGm;
    GlobalTensor<float> initGm;
    GlobalTensor<float> finalGm;
    qGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(q));
    kGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(k));
    vGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(v));
    gGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(g));
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output));
    betaGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(beta));
    scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale));
    aLogGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a_log));
    biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dt_bias));
    lowerGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lower_bound));
    initGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(initial_state));
    finalGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(final_state));

    TPipe pipe;
    TBuf<TPosition::VECCALC> stateBuf;
    TBuf<TPosition::VECCALC> stageBuf;
    TBuf<TPosition::VECCALC> workBuf;
    pipe.InitBuffer(stateBuf, kStateElems * sizeof(float));
    pipe.InitBuffer(stageBuf, kStateElems * sizeof(float));
    pipe.InitBuffer(workBuf, kWorkFloats * sizeof(float));
    TQue<QuePosition::VECIN, 2> inQue;
    TQue<QuePosition::VECOUT, 2> outQue;
    pipe.InitBuffer(inQue, 2, kInBufferBytes);
    pipe.InitBuffer(outQue, 2, kTB * kRows * sizeof(bfloat16_t));

    LocalTensor<float> state = stateBuf.Get<float>();
    LocalTensor<float> stage = stageBuf.Get<float>();
    LocalTensor<float> work = workBuf.Get<float>();

    LocalTensor<float> qf = work[kQF];
    LocalTensor<float> kf = work[kKF];
    LocalTensor<float> df = work[kDF];
    LocalTensor<float> vf = work[kVF];
    LocalTensor<float> bf = work[kBF];
    LocalTensor<float> pred = work[kPF];
    LocalTensor<float> outv = work[kOF];
    LocalTensor<float> delta = work[kLF];
    LocalTensor<float> a0 = work[kA0];
    LocalTensor<float> a1 = work[kA1];
    LocalTensor<float> a2 = work[kA2];
    LocalTensor<float> ac = work[kAC];
    LocalTensor<float> tmp = work[kTF];
    LocalTensor<float> betaAux = work[kBA];
    LocalTensor<float> rw = work[kRW];
    LocalTensor<float> bP = work[kBP];
    LocalTensor<float> bT = work[kBT];
    LocalTensor<float> bC = work[kBC];
    LocalTensor<float> bX = stage;
    LocalTensor<float> wk = work[kWK];
    LocalTensor<float> wq = work[kWQ];
    BrcbRepeatParams brcb;
    brcb.dstBlkStride = 1;
    brcb.dstRepStride = 8;

    const float scaleVal = scaleGm.GetValue(0);
    const float lowerVal = lowerGm.GetValue(0);
    const float aLogVal = aLogGm.GetValue(head);

    DataCopy(bf, biasGm[head * kDim], kDim);
    DataCopy(stage, initGm[(head * kDim + slice * kRows) * kDim], kStateElems);

    work.SetValue(kAC, aLogVal);
    syncMte2ToS(pipe);
    syncMte2ToV(pipe);
    syncS2V(pipe);
    Exp(work[kAC], work[kAC], 8);
    syncV2S(pipe);
    const float logGain = work.GetValue(kAC);

    for (int32_t kk = 0; kk < kDim; ++kk) {
        for (int32_t r = 0; r < kRows; ++r) {
            state.SetValue(kk * kRows + r, stage.GetValue(r * kDim + kk));
        }
    }
    syncS2V(pipe);

    DataCopyParams pq;
    pq.blockCount = static_cast<uint16_t>(kTB);
    pq.blockLen = 8;
    pq.srcStride = static_cast<uint16_t>(kPerTokBlocks - 8);
    pq.dstStride = 0;
    DataCopyParams pv;
    pv.blockCount = static_cast<uint16_t>(kTB);
    pv.blockLen = 4;
    pv.srcStride = static_cast<uint16_t>(kPerTokBlocks - 4);
    pv.dstStride = 0;
    DataCopyParams pb;
    pb.blockCount = static_cast<uint16_t>(kTB);
    pb.blockLen = 1;
    pb.srcStride = static_cast<uint16_t>(kHeads / 8 - 1);
    pb.dstStride = 0;
    DataCopyParams po;
    po.blockCount = static_cast<uint16_t>(kTB);
    po.blockLen = 4;
    po.srcStride = 0;
    po.dstStride = static_cast<uint16_t>(kPerTokBlocks - 4);

    const int32_t headGroup = head & ~7;
    const int32_t headLane = head - headGroup;
    const int32_t qkBytes = 3 * kTB * kDim;

    for (int32_t tb = 0; tb < kTokens; tb += kTB) {
        const int32_t gBase = tb * kPerTok + head * kDim;
        auto inT = inQue.AllocTensor<bfloat16_t>();
        DataCopy(inT[0], qGm[gBase], pq);
        DataCopy(inT[kTB * kDim], kGm[gBase], pq);
        DataCopy(inT[2 * kTB * kDim], gGm[gBase], pq);
        DataCopy(inT[qkBytes], vGm[gBase + slice * kRows], pv);
        LocalTensor<float> betaF = inT.ReinterpretCast<float>();
        DataCopy(betaF[kBetaOff], betaGm[tb * kHeads + headGroup], pb);
        inQue.EnQue(inT);
        auto inR = inQue.DeQue<bfloat16_t>();
        auto outT = outQue.AllocTensor<bfloat16_t>();

        syncV2S(pipe);
        syncMte2ToS(pipe);
        for (int32_t i = 0; i < kTB; ++i) {
            betaAux.SetValue(i * 8, betaF.GetValue(kBetaOff + i * 8 + headLane));
        }
        syncS2V(pipe);

        for (int32_t i = 0; i < kTB; ++i) {
            Cast(qf, inR[i * kDim], RoundMode::CAST_NONE, kDim);
            Cast(kf, inR[kTB * kDim + i * kDim], RoundMode::CAST_NONE, kDim);
            Cast(df, inR[2 * kTB * kDim + i * kDim], RoundMode::CAST_NONE, kDim);
            Cast(vf, inR[qkBytes + i * kRows], RoundMode::CAST_NONE, kRows);

            Mul(tmp, qf, qf, kDim);
            ReduceSum(a0, tmp, rw, kDim);
            Mul(tmp, kf, kf, kDim);
            ReduceSum(a1, tmp, rw, kDim);
            Mul(tmp, qf, kf, kDim);
            ReduceSum(a2, tmp, rw, kDim);

            Muls(ac, betaAux[i * 8], 1.0f, 8);
            Muls(ac, ac, -1.0f, 8);
            Exp(ac, ac, 8);
            Adds(ac, ac, 1.0f, 8);
            Reciprocal(ac, ac, 8);

            Add(df, df, bf, kDim);
            Muls(df, df, logGain, kDim);
            Muls(df, df, -1.0f, kDim);
            Exp(df, df, kDim);
            Adds(df, df, 1.0f, kDim);
            Reciprocal(df, df, kDim);
            Muls(df, df, lowerVal, kDim);
            Exp(df, df, kDim);

            Adds(a0, a0, kNormEps, 8);
            Rsqrt(a0, a0, 8);
            Adds(a1, a1, kNormEps, 8);
            Rsqrt(a1, a1, 8);
            syncV2S(pipe);

            const float qFac = a0.GetValue(0) * scaleVal;
            const float kFac = a1.GetValue(0);
            const float kqDot = a2.GetValue(0) * qFac * kFac;
            const float gain = ac.GetValue(0);

            Muls(qf, qf, qFac, kDim);
            Muls(kf, kf, kFac, kDim);

            Brcb(bP, df, static_cast<uint8_t>(kChunk / 8), brcb);
            Brcb(bT, bP, static_cast<uint8_t>(kChunk), brcb);
            Mul(state, state, bT, kChunkLanes);
            Brcb(bP, kf, static_cast<uint8_t>(kChunk / 8), brcb);
            Brcb(bC, bP, static_cast<uint8_t>(kChunk), brcb);
            Mul(bX, state, bC, kChunkLanes);
            for (int32_t s = kChunkLanes / 2; s >= kRows; s >>= 1) {
                Add(bX, bX, bX[s], s);
            }
            Adds(pred, bX, 0.0f, kRows);
            Sub(delta, vf, pred, kRows);
            Muls(delta, delta, gain, kRows);
            Brcb(bP, qf, static_cast<uint8_t>(kChunk / 8), brcb);
            Brcb(bT, bP, static_cast<uint8_t>(kChunk), brcb);
            Mul(bX, state, bT, kChunkLanes);
            for (int32_t s = kChunkLanes / 2; s >= kRows; s >>= 1) {
                Add(bX, bX, bX[s], s);
            }
            Adds(outv, bX, 0.0f, kRows);
            Axpy(outv, delta, kqDot, kRows);
            Adds(bX, delta, 0.0f, kRows);
            for (int32_t n = kRows; n < kChunkLanes; n <<= 1) {
                Adds(bX[n], bX, 0.0f, n);
            }
            Mul(bT, bC, bX, kChunkLanes);
            Add(state, state, bT, kChunkLanes);
            Cast(outT[i * kRows], outv, RoundMode::CAST_RINT, kRows);
        }

        outQue.EnQue(outT);
        auto outR = outQue.DeQue<bfloat16_t>();
        DataCopy(outGm[gBase + slice * kRows], outR, po);
        outQue.FreeTensor(outR);
        inQue.FreeTensor(inR);
    }

    syncV2S(pipe);
    for (int32_t kk = 0; kk < kDim; ++kk) {
        for (int32_t r = 0; r < kRows; ++r) {
            stage.SetValue(r * kDim + kk, state.GetValue(kk * kRows + r));
        }
    }
    syncS2Mte3(pipe);
    DataCopy(finalGm[(head * kDim + slice * kRows) * kDim], stage, kStateElems);
}

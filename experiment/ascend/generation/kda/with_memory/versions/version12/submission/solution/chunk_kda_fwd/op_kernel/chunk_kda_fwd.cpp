#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr int32_t kDim = 128;
constexpr int32_t kHeads = 96;
constexpr int32_t kTokens = 4096;
constexpr int32_t kVBlock = 64;
constexpr int32_t kVSplit = 2;
constexpr int32_t kRows = kDim;
constexpr int32_t kTile = kRows * kVBlock;
constexpr int32_t kBcElems = kDim * 8;
constexpr float kNormEps = 1e-6f;

template <HardEvent kEvent>
__aicore__ inline void syncNow(TPipe& pipe, TEventID id) {
    SetFlag<kEvent>(id);
    WaitFlag<kEvent>(id);
}

__aicore__ inline void foldRows(LocalTensor<float>& tile) {
    uint32_t remain = static_cast<uint32_t>(kRows);
    while (remain > 1) {
        const uint32_t half = remain / 2;
        Add(tile, tile, tile[half * kVBlock], static_cast<int32_t>(half * kVBlock));
        remain = half;
    }
}

__aicore__ inline void foldVector(LocalTensor<float>& vec, int32_t base, int32_t length) {
    int32_t remain = length;
    while (remain > 8) {
        const int32_t half = remain / 2;
        Add(vec[base], vec[base], vec[base + half], half);
        remain = half;
    }
}

}  // namespace

extern "C" __global__ __aicore__ void chunk_kda_fwd(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR scale, GM_ADDR a_log, GM_ADDR dt_bias, GM_ADDR lower_bound,
    GM_ADDR initial_state, GM_ADDR output, GM_ADDR final_state,
    GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workspace;
    (void)tiling;

    const int32_t blk = static_cast<int32_t>(GetBlockIdx());
    const int32_t head = blk / kVSplit;
    const int32_t vpart = blk % kVSplit;
    if (head >= kHeads) return;

    const int32_t headOffset = head * kDim * kDim;

    GlobalTensor<bfloat16_t> qGm, kGm, vGm, gGm, outGm;
    GlobalTensor<float> betaGm, scaleGm, logGm, biasGm, lowerGm, stateGm, finalGm;
    qGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(q));
    kGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(k));
    vGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(v));
    gGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(g));
    betaGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(beta));
    scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale));
    logGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a_log));
    biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dt_bias));
    lowerGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lower_bound));
    stateGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(initial_state) + headOffset);
    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output));
    finalGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(final_state) + headOffset);

    TPipe pipe;
    TBuf<TPosition::VECCALC> stateBuf, decBuf, prodBuf, onesBuf, biasBuf, bcBuf, tokBuf, outBuf, scrBuf;
    pipe.InitBuffer(stateBuf, kTile * sizeof(float));
    pipe.InitBuffer(decBuf, kTile * sizeof(float));
    pipe.InitBuffer(prodBuf, kTile * sizeof(float));
    pipe.InitBuffer(onesBuf, kTile * sizeof(float));
    pipe.InitBuffer(biasBuf, kDim * sizeof(float));
    pipe.InitBuffer(bcBuf, 3 * kBcElems * sizeof(float));
    pipe.InitBuffer(tokBuf, 8 * kDim * sizeof(bfloat16_t));
    pipe.InitBuffer(outBuf, kDim * sizeof(bfloat16_t));
    pipe.InitBuffer(scrBuf, 1024 * sizeof(float));

    LocalTensor<float> state = stateBuf.Get<float>();
    LocalTensor<float> sdec = decBuf.Get<float>();
    LocalTensor<float> prod = prodBuf.Get<float>();
    LocalTensor<float> ones = onesBuf.Get<float>();
    LocalTensor<float> bias = biasBuf.Get<float>();
    LocalTensor<float> decBc = bcBuf.Get<float>();
    LocalTensor<float> ktBc = bcBuf.Get<float>()[kBcElems];
    LocalTensor<float> qtBc = bcBuf.Get<float>()[2 * kBcElems];
    LocalTensor<bfloat16_t> tok = tokBuf.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> outBf = outBuf.Get<bfloat16_t>();
    LocalTensor<float> scr = scrBuf.Get<float>();

    LocalTensor<float> q32 = scr[0];
    LocalTensor<float> k32 = scr[kDim];
    LocalTensor<float> g32 = scr[2 * kDim];
    LocalTensor<float> v32 = scr[3 * kDim];
    LocalTensor<float> d32 = scr[4 * kDim];
    LocalTensor<float> t1 = scr[5 * kDim];
    LocalTensor<float> sq = scr[6 * kDim];
    LocalTensor<float> sk = scr[7 * kDim];
    LocalTensor<float> sums = scr[960];

    const TEventID evMte2V = pipe.FetchEventID(HardEvent::MTE2_V);
    const TEventID evVS = pipe.FetchEventID(HardEvent::V_S);
    const TEventID evSV = pipe.FetchEventID(HardEvent::S_V);
    const TEventID evVMte3 = pipe.FetchEventID(HardEvent::V_MTE3);
    const TEventID evMte3V = pipe.FetchEventID(HardEvent::MTE3_V);
    const TEventID evMte2V0 = pipe.FetchEventID(HardEvent::MTE2_V);
    const TEventID evMte2V1 = pipe.FetchEventID(HardEvent::MTE2_V);
    const TEventID evVM2 = pipe.FetchEventID(HardEvent::V_MTE2);

    Duplicate(ones, 1.0f, kTile);
    DataCopy(bias, biasGm[head * kDim], kDim);
    DataCopy(state, stateGm[vpart * kVBlock],
             DataCopyParams{static_cast<uint16_t>(kRows), 8, 8, 0});
    syncNow<HardEvent::MTE2_V>(pipe, evMte2V);

    DataCopy(tok, qGm[head * kDim], kDim);
    DataCopy(tok[kDim], kGm[head * kDim], kDim);
    DataCopy(tok[2 * kDim], gGm[head * kDim], kDim);
    DataCopy(tok[3 * kDim], vGm[head * kDim + vpart * kVBlock], kVBlock);
    SetFlag<HardEvent::MTE2_V>(evMte2V0);

    const float queryScale = scaleGm.GetValue(0);
    const float lower = lowerGm.GetValue(0);
    sums.SetValue(0, logGm.GetValue(head));
    syncNow<HardEvent::S_V>(pipe, evSV);
    Exp(sums[0], sums[0], 1);
    syncNow<HardEvent::V_S>(pipe, evVS);
    const float aLogExp = sums.GetValue(0);

    float betaCur = betaGm.GetValue(head);
    for (int32_t step = 0; step < kTokens; ++step) {
        const int32_t base = (step * kHeads + head) * kDim;
        const int32_t vbase = base + vpart * kVBlock;
        const float betaRaw = betaCur;
        const int32_t cur = (step & 1) * 4 * kDim;
        LocalTensor<bfloat16_t> tcur = tok[cur];

        WaitFlag<HardEvent::MTE2_V>(step & 1 ? evMte2V1 : evMte2V0);

        Cast(q32, tcur, RoundMode::CAST_NONE, kDim);
        Cast(k32, tcur[kDim], RoundMode::CAST_NONE, kDim);
        Cast(g32, tcur[2 * kDim], RoundMode::CAST_NONE, kDim);
        Cast(v32, tcur[3 * kDim], RoundMode::CAST_NONE, kVBlock);

        SetFlag<HardEvent::V_MTE2>(evVM2);
        WaitFlag<HardEvent::V_MTE2>(evVM2);

        if (step + 1 < kTokens) {
            const int32_t baseN = ((step + 1) * kHeads + head) * kDim;
            const int32_t nxt = ((step + 1) & 1) * 4 * kDim;
            betaCur = betaGm.GetValue((step + 1) * kHeads + head);
            DataCopy(tok[nxt], qGm[baseN], kDim);
            DataCopy(tok[nxt + kDim], kGm[baseN], kDim);
            DataCopy(tok[nxt + 2 * kDim], gGm[baseN], kDim);
            DataCopy(tok[nxt + 3 * kDim], vGm[baseN + vpart * kVBlock], kVBlock);
            SetFlag<HardEvent::MTE2_V>(((step + 1) & 1) ? evMte2V1 : evMte2V0);
        }

        Mul(sq, q32, q32, kDim);
        Mul(sk, k32, k32, kDim);
        foldVector(sq, 0, kDim);
        foldVector(sk, 0, kDim);

        sums.SetValue(24, -betaRaw);
        syncNow<HardEvent::V_S>(pipe, evVS);
        float qsum = 0.0f;
        float ksum = 0.0f;
        for (int32_t i = 0; i < 8; ++i) {
            qsum += sq.GetValue(i);
            ksum += sk.GetValue(i);
        }
        sums.SetValue(8, qsum);
        sums.SetValue(16, ksum);
        syncNow<HardEvent::S_V>(pipe, evSV);
        Adds(sums[8], sums[8], kNormEps, 1);
        Adds(sums[16], sums[16], kNormEps, 1);
        Rsqrt(sums[8], sums[8], 1);
        Rsqrt(sums[16], sums[16], 1);
        Exp(sums[24], sums[24], 1);
        syncNow<HardEvent::V_S>(pipe, evVS);
        const float qFactor = queryScale * sums.GetValue(8);
        const float kFactor = sums.GetValue(16);
        const float gain = 1.0f / (1.0f + sums.GetValue(24));

        Muls(q32, q32, qFactor, kDim);
        Muls(k32, k32, kFactor, kDim);

        Add(t1, g32, bias, kDim);
        Muls(t1, t1, aLogExp, kDim);
        Muls(t1, t1, -1.0f, kDim);
        Exp(t1, t1, kDim);
        Adds(t1, t1, 1.0f, kDim);
        Div(t1, ones, t1, kDim);
        Muls(t1, t1, lower, kDim);
        Exp(d32, t1, kDim);

        Brcb(ktBc, k32, 16, {1, 8});
        Brcb(qtBc, q32, 16, {1, 8});
        Brcb(decBc, d32, 16, {1, 8});

        Mul(state, state, decBc, 64, 128, {1, 1, 0, 8, 8, 1});
        Mul(prod, state, ktBc, 64, 128, {1, 1, 0, 8, 8, 1});
        foldRows(prod);

        Sub(sq, v32, prod, kVBlock);
        Muls(sq, sq, gain, kVBlock);
        MulAddDst(state, sq, ktBc, 64, 128, {1, 1, 0, 8, 0, 1});

        Mul(prod, state, qtBc, 64, 128, {1, 1, 0, 8, 8, 1});
        foldRows(prod);

        syncNow<HardEvent::MTE3_V>(pipe, evMte3V);
        Cast(outBf, prod, RoundMode::CAST_RINT, kVBlock);
        syncNow<HardEvent::V_MTE3>(pipe, evVMte3);
        DataCopy(outGm[vbase], outBf, kVBlock);
    }

    syncNow<HardEvent::V_MTE3>(pipe, evVMte3);
    DataCopy(finalGm[vpart * kVBlock], state,
             DataCopyParams{static_cast<uint16_t>(kRows), 8, 0, 8});
}

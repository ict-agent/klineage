#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t kDim = 128;
constexpr int32_t kHeads = 96;
constexpr int32_t kTokens = 4096;
constexpr float kNormEps = 1e-6f;

__aicore__ inline void syncInput(TPipe& pipe) {
    const event_t event = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(event);
    WaitFlag<HardEvent::MTE2_S>(event);
}

__aicore__ inline void syncOutput(TPipe& pipe) {
    const event_t event = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(event);
    WaitFlag<HardEvent::S_MTE3>(event);
}

__aicore__ inline void scalarToVector(TPipe& pipe) {
    const event_t event = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(event);
    WaitFlag<HardEvent::S_V>(event);
}

__aicore__ inline void vectorToScalar(TPipe& pipe) {
    const event_t event = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(event);
    WaitFlag<HardEvent::V_S>(event);
}

extern "C" __global__ __aicore__ void chunk_kda_fwd(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR scale, GM_ADDR a_log, GM_ADDR dt_bias, GM_ADDR lower_bound,
    GM_ADDR initial_state, GM_ADDR output, GM_ADDR final_state,
    GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workspace;
    (void)tiling;

    const int32_t head = GetBlockIdx();
    if (head >= kHeads) return;

    GlobalTensor<bfloat16_t> qGm, kGm, vGm, gGm, outputGm;
    GlobalTensor<float> betaGm, scaleGm, logGm, biasGm, lowerGm, initialGm, finalGm;
    qGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(q));
    kGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(k));
    vGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(v));
    gGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(g));
    betaGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(beta));
    scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scale));
    logGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a_log));
    biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dt_bias));
    lowerGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lower_bound));
    initialGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(initial_state));
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output));
    finalGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(final_state));

    TPipe pipe;
    TBuf<TPosition::VECCALC> stateBuf, tokenBuf, scratchBuf, biasBuf;
    pipe.InitBuffer(stateBuf, kDim * kDim * sizeof(float));
    pipe.InitBuffer(tokenBuf, 5 * kDim * sizeof(bfloat16_t));
    pipe.InitBuffer(scratchBuf, 9 * kDim * sizeof(float));
    pipe.InitBuffer(biasBuf, kDim * sizeof(float));
    LocalTensor<float> state = stateBuf.Get<float>();
    LocalTensor<bfloat16_t> token = tokenBuf.Get<bfloat16_t>();
    LocalTensor<float> scratch = scratchBuf.Get<float>();
    LocalTensor<float> bias = biasBuf.Get<float>();

    DataCopy(state, initialGm[head * kDim * kDim], kDim * kDim);
    DataCopy(bias, biasGm[head * kDim], kDim);
    syncInput(pipe);

    const float queryScale = scaleGm.GetValue(0);
    const float lower = lowerGm.GetValue(0);
    scratch.SetValue(8 * kDim, logGm.GetValue(head));
    scalarToVector(pipe);
    Exp(scratch[8 * kDim], scratch[8 * kDim], 1);
    vectorToScalar(pipe);
    const float logGain = scratch.GetValue(8 * kDim);

    for (int32_t step = 0; step < kTokens; ++step) {
        const int32_t tokenBase = (step * kHeads + head) * kDim;
        DataCopy(token[0], qGm[tokenBase], kDim);
        DataCopy(token[kDim], kGm[tokenBase], kDim);
        DataCopy(token[2 * kDim], vGm[tokenBase], kDim);
        DataCopy(token[3 * kDim], gGm[tokenBase], kDim);
        const event_t inputReady = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(inputReady);
        WaitFlag<HardEvent::MTE2_V>(inputReady);
        for (int32_t input = 0; input < 4; ++input) {
            Cast(scratch[input * kDim], token[input * kDim], RoundMode::CAST_NONE, kDim);
        }
        vectorToScalar(pipe);

        float qNorm = kNormEps;
        float kNorm = kNormEps;
        for (int32_t key = 0; key < kDim; ++key) {
            const float query = scratch.GetValue(key);
            const float kValue = scratch.GetValue(kDim + key);
            qNorm += query * query;
            kNorm += kValue * kValue;
        }
        scratch.SetValue(8 * kDim, qNorm);
        scratch.SetValue(8 * kDim + 1, kNorm);
        scalarToVector(pipe);
        Rsqrt(scratch[8 * kDim], scratch[8 * kDim], 2);
        vectorToScalar(pipe);
        const float qFactor = queryScale * scratch.GetValue(8 * kDim);
        const float kFactor = scratch.GetValue(8 * kDim + 1);
        for (int32_t key = 0; key < kDim; ++key) {
            const float gate = scratch.GetValue(3 * kDim + key);
            const float bound = logGain * (gate + bias.GetValue(key));
            scratch.SetValue(6 * kDim + key, -bound);
            scratch.SetValue(4 * kDim + key, scratch.GetValue(key) * qFactor);
            scratch.SetValue(5 * kDim + key, scratch.GetValue(kDim + key) * kFactor);
        }
        scalarToVector(pipe);
        Exp(scratch[6 * kDim], scratch[6 * kDim], kDim);
        vectorToScalar(pipe);
        for (int32_t key = 0; key < kDim; ++key) {
            const float sigmoid = 1.0f / (1.0f + scratch.GetValue(6 * kDim + key));
            scratch.SetValue(6 * kDim + key, lower * sigmoid);
        }
        scalarToVector(pipe);
        Exp(scratch[6 * kDim], scratch[6 * kDim], kDim);
        vectorToScalar(pipe);

        const float betaValue = betaGm.GetValue(step * kHeads + head);
        scratch.SetValue(8 * kDim, -betaValue);
        scalarToVector(pipe);
        Exp(scratch[8 * kDim], scratch[8 * kDim], 1);
        vectorToScalar(pipe);
        const float gain = 1.0f / (1.0f + scratch.GetValue(8 * kDim));
        for (int32_t value = 0; value < kDim; ++value) {
            const int32_t rowBase = value * kDim;
            float predicted = 0.0f;
            for (int32_t key = 0; key < kDim; ++key) {
                const float decayed = state.GetValue(rowBase + key) *
                                      scratch.GetValue(6 * kDim + key);
                state.SetValue(rowBase + key, decayed);
                predicted += decayed * scratch.GetValue(5 * kDim + key);
            }
            const float delta = gain *
                (scratch.GetValue(2 * kDim + value) - predicted);
            float readout = 0.0f;
            for (int32_t key = 0; key < kDim; ++key) {
                const float updated = state.GetValue(rowBase + key) +
                                      delta * scratch.GetValue(5 * kDim + key);
                state.SetValue(rowBase + key, updated);
                readout += updated * scratch.GetValue(4 * kDim + key);
            }
            scratch.SetValue(7 * kDim + value, readout);
        }
        scalarToVector(pipe);
        Cast(token[4 * kDim], scratch[7 * kDim], RoundMode::CAST_RINT, kDim);
        const event_t outputReady = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(outputReady);
        WaitFlag<HardEvent::V_MTE3>(outputReady);
        DataCopy(outputGm[tokenBase], token[4 * kDim], kDim);
        const event_t consumed = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(consumed);
        WaitFlag<HardEvent::MTE3_S>(consumed);
    }

    syncOutput(pipe);
    DataCopy(finalGm[head * kDim * kDim], state, kDim * kDim);
}

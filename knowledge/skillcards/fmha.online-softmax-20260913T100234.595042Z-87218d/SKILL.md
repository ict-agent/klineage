---
skill_id: fmha.online-softmax
intent: Avoid a full-row maximum prepass with online softmax and accumulator rescaling.
preconditions:
- 'Data types: the numerical contract permits rounding differences from tile-relative
  probabilities and rescaled accumulation; arithmetic must represent the exponential
  factors, and each row must have a finite maximum in its first processed tile to
  avoid an undefined initial scale.'
- 'Layout: score partials, reduction participants, and PV components have known row
  and tile ownership so each maximum and scale reaches the matching accumulators;
  no additional contiguity or alignment is required.'
- 'Storage: materialized tile storage remains visible from softmax through PV and
  has room for row metadata, so rescaling factors can accompany the probabilities
  across separate launches.'
- 'Pipeline: score production completes before softmax reads, every reader completes
  before score storage is overwritten, and PV follows the same tile order before storage
  reuse; otherwise probabilities or scales can be stale or applied to the wrong accumulated
  prefix.'
- 'Hardware: no additional accelerator feature is needed beyond the existing arithmetic
  and synchronization; each tile allocation must fit probability_bytes + row_owner_count
  * sizeof(scale_element), because the added metadata shares that allocation.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Replace the full-row maximum prepass with online softmax in
`solution/attention.cuh`. For each key tile, update the running maximum `m`,
compute `alpha = exp2((old_m - m) * kLog2Scale)`, rescale the preceding denominator,
and exponentiate scores relative to `m`. Save `alpha` beside that tile's rounded
probabilities. Before PV accumulates the tile, multiply each preceding output
component by its row's `alpha`.

This removes one score traversal. In exact arithmetic, changing normalization
origins cancels between the numerator and denominator. Rounded probabilities and
rescaled accumulators can differ numerically; retain the supplied oracle and
precision requirements.

Apply these edits:

1. In `solution/ops.cuh`, append `float scale[kThreads][2]` to `SoftmaxState`.
   Keep `prob_state`, score-slab allocations, and the existing capacity assertion.
2. In `consumer`, retain initialization to `maximum=-INFINITY` and `sum=0`.
   Remove the entire first descending `n` loop, which only computes `maximum`,
   and its following `reduce<Reduce::Maximum>` loop. Add `scale[2]` to the local
   float arrays. In the remaining tile loop, call
   `softmax(score, maximum, sum, scale, s)` and store both scales after `store_prob`.
3. Replace `softmax` with the After function. Add the scale argument to `row_sum`
   and restore its fused initial rescale/add. Keep the rest of `row_sum` unchanged.
4. Add the After `rescale` helper. In `attention_values`, add local `scale[2]`,
   load it from the tile state, and call `rescale` immediately before `prob_value`.
   Retain output initialization, normalization, and final stores.

Keep the four launches and their ordering:
`attention_scores -> attention -> attention_values -> attention_epilogue`.
QK publishes all scores before softmax. Keep both warp barriers in `reduce` and
the CTA barrier before `store_prob`: probability writes change the score layout
and must wait for every score reader. The later PV launch sees completed
probabilities and scales. Allocations remain alive through the caller-stream
launches; do not reuse them before PV finishes reading.

Retain score masking for every tile, invalid-Q/K zero loads, and bounded output
stores. The descending traversal begins with a nonempty key tile. Valid rows have
a finite first maximum; padded query rows use zero scores. Keep the finite-max
fallback, exponential implementation, scalar conversion helpers, reciprocal
behavior, and grouped reduction tree.

Check the existing capacity assertion after adding metadata. Inspect that the
maximum prepass is absent, each tile publishes its scales, and PV rescales before
its first FMA for that tile. Use the existing Kernel evaluator with the unchanged
problem and policy to check numerical acceptance and latency; keep evidence
outside this card.

## Example configuration

Replay with the existing CUDA SM90 build and all existing compile flags:
`-O3 -std=c++17 --use_fast_math --resource-usage -lineinfo -DNDEBUG`.
The workload has eight sequences, 16384 packed tokens, 64 heads, dimension 128,
FP16 Q/K/V/output, int32 offsets, and FP32 scores, maxima, sums, and accumulators.
The oracle's tolerances remain unchanged.

Preserve `kM=128`, `kN=176`, 256 threads, 128-thread ownership groups, 32-thread
warps, 88 scores, 44 packed probability registers, and 64 PV components per thread.
Each lane owns two rows. For score index `i`, its local row slot is
`(i % kFragmentSize) / kPairElems`; the key column is
`(tid % kLanesPerRow) * kPairElems + (i / kFragmentSize) * kInputPanelCols
+ i % kPairElems`. Four row lanes cooperate in the existing grouped reduction.
PV uses the same local row slot.

Global Q/K/V strides are `(kHeads*kDim, kDim, 1)` in half elements; probabilities
are row-major `[kM,kN]` within each CTA's tile. Each score slab contains
`kMaxQTiles*kHeads*kThreads` `ScoreState` records. Preserve `kMaxQTiles=135`,
`kMaxKeyTiles=94`, grid `kMaxQTiles*kHeads=8640`, and zero dynamic shared memory.
The CTA score region is 90112 bytes. Its probability payload is 45056 bytes;
adding 2048 bytes of scales yields 47104 bytes and fits the existing allocation.
No host ABI, launch, pointer-table, or allocation changes are needed.

Retain descending key-tile traversal, increasing-dimension QK FMAs, and
increasing-key PV FMAs within each tile. Keep `kLog2Scale=0.127517432f` and
`kScale=0.0883883461f`, FP16 round-to-nearest probability/output conversion,
separate exponential and sum traversals, repeated `score_offset` calls, scalar
output stores, per-component reciprocals, global reduction scratch, score/probability
storage reuse, and separate output materialization.

# Precondition

- Data types: the contract must permit rounding differences caused by changing
  normalization origins for probabilities and rescaling prior accumulators.
  Otherwise the real-arithmetic recurrence does not guarantee compliant outputs.
  Arithmetic must represent the exponential factors. Each row's first processed
  tile must have a finite maximum: subtracting two negative infinities would make
  the initial scale undefined. No particular operand dtype is intrinsic.
- Layout: row and tile ownership must be known for score partials, reduction
  participants, and PV components. The row maximum must include every partial,
  and its scale must multiply the matching numerator and denominator. Incorrect
  correspondence changes the softmax. No extra contiguity or alignment is needed.
- Storage: tile storage is already shared by separate softmax and PV launches and
  remains visible through both. Space for accompanying row metadata is necessary
  because PV must recover the normalization change used for that probability tile.
- Pipeline: scores must be published before softmax reads them; all score readers
  must finish before their storage is overwritten. PV must visit tiles in the
  same order as softmax so each scale acts on the correct accumulated prefix.
  Probabilities and metadata must remain intact until PV finishes. These are
  ordering and lifetime requirements, independent of a particular stage count.
- Hardware: the existing arithmetic and synchronization suffice; no additional
  accelerator feature is required. Per-tile capacity must cover
  `probability_bytes + row_owner_count * sizeof(scale_element)` because the
  metadata accompanies probabilities in the same allocation. Count duplicated
  row owners when metadata is stored separately for each owner.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These fragments show only the edited regions. Keep surrounding code unchanged;
the numbered edits above specify deletion of the maximum prepass.

## Before

```cuda
// ops.cuh
struct SoftmaxState {
    __half prob[kProbTileElems];
};

// attention.cuh: row_sum signature and initial expression; keep the remaining loop.
__device__ __noinline__ float row_sum(const float (&score)[kQkRegs], float sum,
                                     int row) {
    float result = sum + score[row * kPairElems];
    // Existing increasing-col loop and return follow.
}

__device__ __forceinline__ void softmax(float (&score)[kQkRegs], const float (&maximum)[2],
                                      float (&sum)[2]) {
    #pragma unroll
    for (int row = 0; row < 2; ++row) {
        const float m = maximum[row];
        const float finite_max = m == -INFINITY ? 0.f : m;
        #pragma unroll
        for (int col = 0; col < kQkRegs / 2; ++col) {
            int i = (col / 2) * 4 + row * 2 + col % 2;
            const float scaled = score_offset(finite_max);
            score[i] = exp2f(score[i] * kLog2Scale - scaled);
        }
        sum[row] = row_sum(score, sum[row], row);
    }
}

// consumer: after its full-row maximum prepass, inside its second tile loop.
softmax(score, maximum, sum);
convert(score, prob);
__syncthreads();
auto& state = prob_state(p, n);
store_prob(prob, state.prob);

// attention_values: inside its descending tile loop.
const auto& state = prob_state(p, n);
prob_value(out, state.prob, p.v + base, valid, wg);
```

## After

```cuda
// ops.cuh: retain the existing sizeof assertion immediately after this structure.
struct SoftmaxState {
    __half prob[kProbTileElems];
    float scale[kThreads][2];
};

// attention.cuh: keep the remaining row_sum loop unchanged.
__device__ __noinline__ float row_sum(const float (&score)[kQkRegs], float sum,
                                     float scale, int row) {
    float result = fmaf(scale, sum, score[row * kPairElems]);
    // Existing increasing-col loop and return follow.
}

__device__ __forceinline__ void softmax(float (&score)[kQkRegs], float (&maximum)[2],
                                      float (&sum)[2], float (&scale)[2], Reduction& s) {
    #pragma unroll
    for (int row = 0; row < 2; ++row) {
        float m = maximum[row];
        #pragma unroll
        for (int col = 0; col < kQkRegs / 2; ++col)
            m = fmaxf(m, score[(col / 2) * 4 + row * 2 + col % 2]);
        m = reduce<Reduce::Maximum>(m, s);
        scale[row] = exp2f((maximum[row] - m) * kLog2Scale);
        maximum[row] = m;
        const float finite_max = m == -INFINITY ? 0.f : m;
        #pragma unroll
        for (int col = 0; col < kQkRegs / 2; ++col) {
            int i = (col / 2) * 4 + row * 2 + col % 2;
            const float scaled = score_offset(finite_max);
            score[i] = exp2f(score[i] * kLog2Scale - scaled);
        }
        sum[row] = row_sum(score, sum[row], scale[row], row);
    }
}

__device__ __forceinline__ void rescale(float (&out)[kPvRegs], const float (&scale)[2]) {
    #pragma unroll
    for (int i = 0; i < kPvRegs; ++i) out[i] *= scale[(i % 4) / 2];
}

// consumer: add local scale[2]; delete the maximum prepass and its row reduction.
// Keep maximum=-INFINITY and sum=0 initialization. In its sole tile loop:
softmax(score, maximum, sum, scale, s);
convert(score, prob);
__syncthreads();
auto& state = prob_state(p, n);
store_prob(prob, state.prob);
#pragma unroll
for (int r = 0; r < 2; ++r) state.scale[tid][r] = scale[r];

// attention_values: add local scale[2]. Inside its descending tile loop:
const auto& state = prob_state(p, n);
#pragma unroll
for (int r = 0; r < 2; ++r) scale[r] = state.scale[tid][r];
rescale(out, scale);
prob_value(out, state.prob, p.v + base, valid, wg);
```

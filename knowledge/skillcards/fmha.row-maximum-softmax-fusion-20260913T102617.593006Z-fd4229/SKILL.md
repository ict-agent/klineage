---
skill_id: fmha.row-maximum-softmax-fusion
intent: Fuse row-maximum reduction with softmax production to retain maxima across
  both passes.
preconditions:
- 'Data types: register-held maxima must preserve the representation and rounding
  observed after scratch reload; otherwise softmax offsets change.'
- 'Layout: each row maximum can be produced by its consuming CTA with matching lane
  ownership; register forwarding cannot communicate across CTAs.'
- 'Storage: scores remain readable through both passes, and maximum scratch has no
  consumers outside these stages; removing scratch must not discard needed data.'
- 'Pipeline: score production precedes maximum reduction, reduced maxima precede exponentiation,
  and all readers finish before score or reduction storage is reused; preserve participating-lane
  publication and reuse barriers.'
- 'Hardware: no additional feature or shared-memory capacity is required; fusion uses
  ordinary per-thread values and the existing reduction synchronization.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Fuse row-maximum reduction into softmax production. Keep each lane's reduced
maxima in local values across the two score passes, removing their global
store/reload and one caller-stream launch.

In `solution/attention.cuh`, replace `consumer`'s block that loads
`maxima.maximum` with the After initialization and full-row maximum scan below.
This scan already exists in `attention_maximum`; preserve its descending key-tile
order, increasing fragment-index order, bounds checks, and grouped reduction.
Delete the entire `attention_maximum` kernel after moving the scan.

In `solution/ops.cuh`, remove only `EpilogueState::maximum`. In
`solution/kernel.cu`, remove only the `attention_maximum` launch and its following
error check. The existing allocation derives its size from `sizeof(EpilogueState)`
and needs no further edit. Keep the destination-passing pybind11 ABI and caller
stream. The remaining launch configuration stays unchanged.

Each CTA owns one query tile and head. Its score fragments and row statistics
use index `size_t(blockIdx.x) * kThreads + threadIdx.x`. Each lane owns two query
rows; the existing row-lane group reduces their maxima. Keep the same ownership
so each softmax consumer receives exactly its own row maxima. Q/K/V and output
remain packed NHD; score fragments retain their current global layout.

The preceding score launch completes before the fused kernel. Preserve both
`__syncwarp(kFullWarpMask)` calls in `reduce`: partials must be published before
gathers and gathers must finish before slot reuse. Each lane consumes its own
completed reduction, so moving this work requires no new CTA barrier. Retain
`consumer`'s `__syncthreads()` before a score tile becomes probability storage;
all CTA readers must finish before those bytes are overwritten. PV and epilogue
launches remain ordered after softmax on the caller stream.

Retain the invalid-CTA return, per-score key bounds, padded-score handling,
full-row fixed maximum, FP32 arithmetic and reduction grouping, exponential
expression, scalar FP16 rounding, and normalization. Do not change compile flags
or fuse the separate exponential and summation traversals.

## Example configuration

The workload has 16384 tokens, 64 heads, dimension 128, and eight sequences.
Q/K/V and output are FP16; maxima, scores, sums, and accumulators are FP32.
Packed row stride is `kHeads * kDim = 8192` half elements. The default sequences
have 2048 tokens; retain offset-derived lengths and ragged bounds.

Preserve `kM=128`, `kN=176`, `kThreads=256`, `kQkRegs=88`, `kPvRegs=64`, and
`kProbRegs=44`. Four lanes cooperate per row; each lane owns two rows.
Query-row ownership is `warp*16 + lane/4 + row*8`. A fragment's key column is
`(tid%4)*2 + (i/4)*8 + i%2`. `kMaxQTiles=135` and `kMaxKeyTiles=94` retain the
existing overprovisioned ragged grid and key-slab bounds.

All remaining launches use grid `kMaxQTiles*kHeads`, block `kThreads`, zero
dynamic shared memory, and `__launch_bounds__(kThreads,1)`. The sequence becomes
scores, fused maximum/softmax, descending per-key-tile PV launches, then epilogue.
The per-thread epilogue record shrinks from 70 to 68 floats. Keep the separate
score slabs, in-place score/probability reuse, global reduction scratch, scalar
QK/PV loops, separate `row_sum`, `score_offset`, `scalar_half`, scalar output
stores, and repeated per-component reciprocal instructions.

Preserve `kLog2Scale=0.127517432f`, `kScale=0.0883883461f`, the SM90 backend,
and flags `-O3 -std=c++17 --use_fast_math --resource-usage -lineinfo -DNDEBUG`.
These are replay settings, not prerequisites of launch fusion.

# Precondition

- Data types: retaining a maximum locally must reproduce the representation and
  rounding seen by the original scratch consumer. Any rounding formerly caused
  by the store/reload must remain explicit; otherwise exponential offsets change.
- Layout: a row's producer and consumers must fit the same CTA with matching
  lane ownership. A register value cannot replace communication to another CTA.
- Storage: score storage must remain readable for the maximum and exponential
  passes. Maximum scratch must serve only communication between these stages;
  deleting it would otherwise strand another consumer.
- Pipeline: score producers finish before the scan, and row reductions finish
  before exponentiation. Keep participating-lane publication and reader-completion
  barriers for reduction scratch, and complete score reads before probability
  writes reuse their storage. Removing the launch boundary must preserve these
  ordering and visibility guarantees.
- Hardware: no additional feature or shared-memory capacity is required. Ordinary
  per-thread values carry maxima; the existing reduction synchronization supplies
  the required communication.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The snippets show separate edit sites. `consumer` already defines `tid`, `last`,
`length`, `score`, `maximum`, `sum`, and its reduction reference `s`. Keep all code
after the replaced initialization block. Delete `attention_maximum` in full when
applying After; its scan is reproduced below.

## Before

```cuda
// solution/ops.cuh
struct EpilogueState {
    float out[kPvRegs];
    float lse[2];
    float denominator[2];
    float maximum[2];
};

// solution/kernel.cu: between score and PV launches.
check(cudaLaunchKernelEx(&config, attention_maximum, p));
check(cudaGetLastError());
check(cudaLaunchKernelEx(&config, attention, p));
check(cudaGetLastError());

// solution/attention.cuh: attention_maximum publishes its reduced maxima.
auto& state = p.epilogue[size_t(blockIdx.x) * kThreads + tid];
#pragma unroll
for (int r = 0; r < 2; ++r) state.maximum[r] = maximum[r];

// solution/attention.cuh: consumer reloads them before its score traversal.
const auto& maxima = p.epilogue[size_t(blockIdx.x) * kThreads + tid];
#pragma unroll
for (int r = 0; r < 2; ++r) {
    maximum[r] = maxima.maximum[r];
    sum[r] = 0.f;
}
```

## After

```cuda
// solution/ops.cuh
struct EpilogueState {
    float out[kPvRegs];
    float lse[2];
    float denominator[2];
};

// solution/kernel.cu: attention now computes its own maxima.
check(cudaLaunchKernelEx(&config, attention, p));
check(cudaGetLastError());

// solution/attention.cuh: replace consumer's maxima-loading block.
#pragma unroll
for (int r = 0; r < 2; ++r) {
    maximum[r] = -INFINITY;
    sum[r] = 0.f;
}
#pragma unroll 1
for (int n = last; n >= 0; --n) {
    const auto& scores = p.scores[n][size_t(blockIdx.x) * kThreads + tid];
    #pragma unroll
    for (int i = 0; i < kQkRegs; ++i) {
        const int col = (tid % kLanesPerRow) * kPairElems
                        + (i / kFragmentSize) * kInputPanelCols + i % kPairElems;
        if (n * kN + col >= length) continue;
        const int row = (i % kFragmentSize) / kPairElems;
        maximum[row] = fmaxf(maximum[row], scores.value[i]);
    }
}
#pragma unroll
for (int r = 0; r < 2; ++r) maximum[r] = reduce<Reduce::Maximum>(maximum[r], s);
```

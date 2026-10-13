---
skill_id: fmha.warp-partitioned-row-maximum
intent: Partition row-maximum scans across warp lanes and exchange their partial maxima.
preconditions:
- 'Data types: no additional dtype restriction; scratch exchange must preserve each
  partial''s representation and maximum semantics, or redistribution changes row statistics.'
- 'Layout: each row has a known partition-to-lane mapping within one warp, with complete
  coverage and distinct scratch slots for concurrently live partials; otherwise the
  gather mixes or omits values.'
- 'Storage: assigned lanes can read their score partitions, and existing writable
  global scratch can hold all live lane partials without aliasing unread data; scans
  and exchange require these values and slots.'
- 'Pipeline: score production finishes before scans; all lanes named by the warp mask
  reach barriers after partial writes and after gathers, preventing incomplete reads
  and premature scratch reuse.'
- 'Hardware: CUDA warp synchronization must order global-memory accesses; scratch
  capacity per live CTA must cover participating_lanes * sizeof(partial), or the exchange
  lacks ordering or storage.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Partition each row-maximum scan across cooperating warp lanes. Each lane scans
one score partition, writes its partial to existing global reduction scratch,
then gathers the row's partials. This removes duplicate scans of every partition
by every lane.

In `solution/attention.cuh`, replace only the block in `attention_maximum`
between `last` initialization and the `Publish maxima` comment with After below.
Before shows the current independent scans. Restore `p.reduction[blockIdx.x]`
and `reduce<Reduce::Maximum>`; both the allocation and helper remain available
for the sum pass. Change no host launch, allocation, ABI, or other kernel.

`ScoreState` stores one fragment per thread at
`p.scores[n][blockIdx.x * kThreads + tid]`. The independent version reads all
owners in the row group; the optimized version reads only `tid`. The existing
`reduce` helper in `solution/ops.cuh` publishes one scalar in `s.reduction[tid]`,
uses `__syncwarp(kFullWarpMask)`, gathers the group's four slots, evaluates
`fmaxf(fmaxf(a, c), fmaxf(b, d))`, then synchronizes again before reuse.
Keep its volatile global accesses and both barriers. The helper's sum branch
and all sum callers remain unchanged.

Preserve descending key-tile traversal, increasing fragment-index traversal,
`-INFINITY` initialization, `fmaxf` semantics, and the final grouping. Keep the
per-score key bound and CTA-uniform invalid-work return. Padded query rows still
participate in the warp barriers; output bounds remain in the epilogue.
The caller-stream score launch finishes before these reads; the maximum launch
finishes before softmax reads `state.maximum`. Successive row reductions reuse
the same slots only after every gather completes. The later sum launch reuses
them after this kernel finishes. Each invocation retains its own reduction
allocation through its caller-stream launches.

Inspect the compiled `attention_maximum` for global partial stores and gathers
when applying this optimization; also preserve both source warp barriers.
Preserve compiler flags; no new compiler controls are needed. Use the existing
evaluator with the complete problem for correctness and latency.

## Example configuration

Retain packed contiguous FP16 Q/K/V/output `[16384,64,128]`, FP32 scores and
maxima, eight sequences, 128 query rows and 176 keys per tile. The row stride is
`kRow = kHeads * kDim = 8192` half elements. Keep the original scale constants,
FP32 scalar FMA orders, FP16 rounding, masks, and ragged-sequence handling.

The grid has `kMaxQTiles * kHeads = 135 * 64 = 8640` CTAs of 256 threads, with
zero dynamic shared memory. Four consecutive lanes own the columns of each row;
each lane owns two rows and 88 score values. A warp owns 16 query rows. Score
index `i` selects row `(i % 4) / 2` and column
`(tid % 4) * 2 + (i / 4) * 8 + i % 2`. Preserve this mapping and the four-partition
maximum tree exactly for this replay.

Keep all 94 score slabs, score-to-probability storage reuse, separate launches,
scalar conversions/stores, independent QK/PV dot products, and the existing
cooperative sum reduction. `Reduction` has 256 float slots (1024 bytes) per CTA;
the unchanged allocation totals 8,847,360 bytes for this grid. Keep
`-O3 -std=c++17 --use_fast_math --resource-usage -lineinfo -DNDEBUG` and the
problem's SM90 CUDA target. These instance settings are not general prerequisites
for partitioning a maximum scan.

# Precondition

- Data types: no additional dtype restriction. Partial writes and reads must
  preserve representation and maximum semantics; changing them changes the row
  statistic. Retain the existing operator, identity, scan order, and final grouping.
- Layout: a known partition-to-lane mapping covers each row within one warp.
  Concurrent partials need distinct scratch slots. An omitted partition loses
  candidate maxima; an incorrect slot or cross-warp group makes the gather read
  the wrong or unordered value.
- Storage: each assigned lane can read its partition, and existing writable
  global scratch has room for all live partials without aliasing unread data.
  Otherwise the partition scan lacks input or the exchange corrupts live values.
- Pipeline: score production completes before partition scans. Every lane named
  by the warp mask reaches the barrier after partial writes and the barrier after
  gathers. The first establishes reader visibility; the second prevents the next
  row or later use from overwriting values still being read.
- Hardware: CUDA warp synchronization orders global-memory accesses. Per live
  CTA, scratch covers `participating_lanes * sizeof(partial)` bytes. Missing
  ordering permits stale reads; insufficient capacity aliases or overruns slots.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The snippets replace only the local maximum-scan block in `attention_maximum`.
`p`, `tid`, `length`, and `last` already exist; publication of `maximum` follows.
The existing reduction helper and its allocation supply the exchange.

## Before

```cuda
    constexpr int kRowsPerLane = 2;
    const int leader = tid - tid % kLanesPerRow;
    float partial[kLanesPerRow][kRowsPerLane];
    float maximum[kRowsPerLane];

    // Each lane scans every row partition without exchanging partial maxima.
    #pragma unroll
    for (int owner = 0; owner < kLanesPerRow; ++owner) {
        #pragma unroll
        for (int r = 0; r < kRowsPerLane; ++r) partial[owner][r] = -INFINITY;

        #pragma unroll 1
        for (int n = last; n >= 0; --n) {
            const auto& scores = p.scores[n][size_t(blockIdx.x) * kThreads + leader + owner];
            #pragma unroll
            for (int i = 0; i < kQkRegs; ++i) {
                const int col = owner * kPairElems
                                + (i / kFragmentSize) * kInputPanelCols + i % kPairElems;
                if (n * kN + col >= length) continue;
                const int row = (i % kFragmentSize) / kPairElems;
                partial[owner][row] = fmaxf(partial[owner][row], scores.value[i]);
            }
        }
    }

    // Preserve the original four-partition maximum grouping locally.
    #pragma unroll
    for (int r = 0; r < kRowsPerLane; ++r) {
        maximum[r] = fmaxf(fmaxf(partial[0][r], partial[2][r]),
                          fmaxf(partial[1][r], partial[3][r]));
    }
```

## After

```cuda
    float maximum[2];
    auto& s = p.reduction[blockIdx.x];

    // Find the full-row maximum before producing any probabilities.
    #pragma unroll
    for (int r = 0; r < 2; ++r) maximum[r] = -INFINITY;
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

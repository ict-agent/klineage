---
skill_id: fmha.tail-tile-score-mask
intent: Skip score-bound checks on complete key tiles.
preconditions:
- 'Data types: tile-index arithmetic must represent tile bases and bounds without
  overflow, or the full-tile validity proof fails. No additional score dtype requirement:
  skipped masks would leave valid scores unchanged.'
- 'Layout: keys occupy a logical prefix of length L in tiles of positive width W,
  and each owned column satisfies 0 <= col < W. Only tile ceil(L/W)-1 can then contain
  out-of-range columns. No additional physical contiguity or alignment requirement.'
- 'Storage: no additional requirement; the guarded operation updates thread-local
  scores and introduces no transfer or shared allocation.'
- 'Pipeline: tail masking must finish before score reductions and exponentiation.
  Only the per-score mask may be conditional; producer visibility, collective participation,
  and completion before scratch reuse must remain intact.'
- 'Hardware: no additional feature or capacity requirement; the transformation uses
  integer comparison and conditional execution with no new buffers.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Skip score-bound checks on complete key tiles. In `solution/attention.cuh`,
`consumer`, wrap the score-masking loop immediately after loading `scores.value`
in `if (n == last)`. Keep the loop and its predicate unchanged. No helper,
allocation, layout, launch, or compiler-flag change is needed.

For sequence length `L` and tile width `W`, `last = ceil(L/W)-1`.
For `n < last` and `0 <= col < W`, `n*W+col < L`; every skipped mask
would leave its score unchanged. The final tile retains elementwise checks,
including when its length is exactly `W`. Other masks, if present, must remain
independent of this bound-check guard.

Each thread keeps its score fragment. Mask invalid keys with `-INFINITY` before
`softmax`; never place `softmax`, conversion, or synchronization inside the new
guard. Preserve scalar QK/PV FMA order, grouped row reductions, online rescaling,
FP16 probability/output rounding, exponentiation, and normalization.

Caller-stream launch ordering publishes QK scores before `consumer` reads them.
Keep the CTA barrier after conversion: every score reader must finish before
probabilities overwrite the same score slab. Keep both warp barriers in `reduce`
for partial visibility and safe slot reuse. Subsequent launches consume the
published probability and epilogue states on the same stream.

Build from the complete edited source bundle with the existing problem and flags.
Use the Kernel evaluator for correctness and timing. Inspect generated `attention`
code: full tiles must bypass the score-mask block, while tail comparisons remain.
Keep these checks and measurements outside this card.

## Example configuration

Preserve `kM=128`, `kN=176`, `kThreads=256`, `kQkRegs=88`, `kPvRegs=64`,
and `kProbRegs=44`. The workload has eight sequences, 16,384 packed tokens,
64 heads, and head dimension 128. Q/K/V/output are contiguous FP16 NHD tensors;
offsets are int32. Arithmetic and row state use FP32. Keep the supplied oracle,
input descriptors, tolerances, and noncausal attention semantics.

For `tid=threadIdx.x`, score slot `i` owns key column
`(tid%4)*2 + (i/4)*8 + i%2`, ranging from 0 through 175. Two row slots share
these columns. Four lanes cooperate per row; the retained row mapping is
`(tid/32)*16 + (tid%32)/4 + ((i%4)/2)*8` within the CTA's query tile.
The score slab uses `[CTA][thread].value[i]`; rounded probabilities retain the
existing K-major panel layout. Preserve both layouts and all stores.

The loop visits `n=last` down through zero, where
`last=(length+kN-1)/kN-1`. Length comes from the current batch's KV offsets;
retain this runtime value for ragged inputs. The maximum launch remains
`kMaxQTiles*kHeads` CTAs, 256 threads each, with zero dynamic shared memory.
Keep all four launches: `attention_scores`, `attention`, `attention_values`,
and `attention_epilogue`, on the caller stream. Keep the existing allocation
bounds, score/probability slab reuse, global reduction scratch, separate
normalization/output pass, scalar conversion helpers, and scalar output stores.

Retain the SM90 target and flags `-O3`, `-std=c++17`, `--use_fast_math`,
`--resource-usage`, `-lineinfo`, and `-DNDEBUG`. Retain both loop pragmas:
`#pragma unroll 1` on key traversal and `#pragma unroll` on score masking.

# Precondition

- Data types: index arithmetic must represent tile bases and bounds without
  overflow; wrapping a tile address invalidates the proof that earlier tiles
  are complete. No additional score dtype requirement applies because this
  optimization skips only masks whose predicates are false.
- Layout: keys form a logical prefix of length `L`, partitioned into positive
  width `W` tiles. Every thread-owned column is within `[0,W)`. Thus only
  `ceil(L/W)-1` can have invalid columns; holes or oversized fragment columns
  invalidate that inference. Physical contiguity and alignment add no requirement
  because the transformation leaves memory accesses unchanged.
- Storage: no additional requirement. The existing mask operates on thread-local
  scores; guarding it neither moves data nor allocates storage.
- Pipeline: invalid tail scores must be masked before reductions and
  exponentiation, or padding enters softmax. Guard only the per-score mask.
  Preserve producer visibility, collective participation, and completion before
  scratch reuse; moving those operations under a tail-only guard can expose
  incomplete data, break collectives, or overwrite unread scores.
- Hardware: no additional feature or capacity requirement. Integer comparison
  and conditional execution implement the guard; there is no new buffer demand.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The snippets replace only the mask block in `consumer`, after the score-fragment
load and before `softmax`. All variables and constants already exist there.
The surrounding descending key loop and synchronization stay unchanged.

## Before

```cuda
// Check every tile's score bounds, including complete tiles.
#pragma unroll
for (int i = 0; i < kQkRegs; ++i) {
    const int col = (tid % 4) * 2 + (i / 4) * 8 + i % 2;
    if (n * kN + col >= length) score[i] = -INFINITY;
}
```

## After

```cuda
// Only the last key tile can contain padding.
if (n == last) {
    #pragma unroll
    for (int i = 0; i < kQkRegs; ++i) {
        const int col = (tid % 4) * 2 + (i / 4) * 8 + i % 2;
        if (n * kN + col >= length) score[i] = -INFINITY;
    }
}
```

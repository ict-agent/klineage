---
skill_id: fmha.softmax-exponential-sum-fusion
intent: Fuse softmax exponential production with row-sum accumulation.
preconditions:
- 'Data types: producer values and reduction arithmetic must retain their representation,
  addition order, and required rounding; otherwise consuming values directly can change
  the sum.'
- 'Layout: each thread must produce and sum the same row subset in the same index
  order; otherwise fusion changes ownership or reduction order. No additional contiguity
  or alignment is required for direct scalar consumption.'
- 'Storage: exponential values and the running partial sum must be accessible to their
  owning thread, without another observer requiring the separate traversal; fusion
  removes those intermediate rereads.'
- 'Pipeline: exponential production must not depend on the evolving sum. Each value
  must be produced before accumulation, and sums and remaining buffer reads must finish
  before dependent consumers or buffer reuse; fusion must preserve these dependencies.'
- 'Hardware: no additional feature or storage capacity is required; the fused traversal
  uses the existing scalar operations and partial-sum accumulator.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Fuse softmax exponential production with its thread-local row-sum traversal in
`solution/attention.cuh`. Consume each new exponential immediately instead of
rereading the row fragment through `row_sum`.

In `softmax`, retain maximum reduction and scale calculation. Restore
`sum[row] *= scale[row]` after `maximum[row] = m`, append
`sum[row] += score[i]` immediately after each exponential assignment, and remove
the trailing `row_sum` call. Delete the now-unused `row_sum` helper, including its
`__noinline__` and `#pragma unroll 1`: these prevent compiler fusion in the
separate traversal. Preserve the producer loop's unrolling and every other helper.
The Before snippet shows the helper and the per-row tail of `softmax`; replace
that tail with After after deleting the helper. The existing preceding code
provides `m`, `row`, `score`, `maximum`, `sum`, and `scale`.

Preserve the first update's fused multiply-add rounding. The separate helper
explicitly computes `fmaf(scale, old_sum, first_exponential)`; the restored
multiply followed by the first addition contracts under the retained compiler
flags. Inspect generated code for this contraction and the subsequent ordered
additions. If contraction is absent, express the first update explicitly with
`fmaf` and add the remaining values in the same order. Keep exponential argument
contraction, flush-to-zero behavior, and FP16 rounding unchanged.

Thread ownership, array indexing, probability storage, launches, and barriers do
not change. Retain score bounds masking before softmax; masked scores contribute
zero exponentials. Each thread completes its local sum before the existing
four-lane sum reduction. Keep `convert`, the CTA barrier before score storage is
reused for probabilities, warp barriers protecting reduction scratch, and the
caller-stream ordering between all four kernels. Keep FP32 exponentials for
conversion and the rounded probability tile for PV. Check correctness and latency
with `klineage.harness.evaluate` using the supplied problem and timing policy;
keep its evidence outside this card.

## Example configuration

Replay the existing packed FP16 Q/K/V and output workload: 16,384 tokens, 64 heads,
head dimension 128, and eight supplied sequences of 2,048 tokens. Keep FP32 scores,
row statistics and scalar FMA accumulators. Preserve `kLog2Scale = 0.127517432f`,
`kScale = 0.0883883461f`, approximate reciprocal normalization, and separate
round-to-nearest FP16 conversions.

Keep `kM = 128`, `kN = 176`, `kQkRegs = 88`, `kPvRegs = 64`,
`kPairElems = 2`, `kFragmentSize = 4`, and four lanes per row. Each thread owns
two row subsets of 44 scores. Its subset index is
`i = (col / 2) * 4 + row * 2 + col % 2`, visited in increasing `col`.
Physical ownership remains
`query_row = (tid / 32) * 16 + (tid % 32) / 4 + row * 8` and
`key_col = (tid % 4) * 2 + (col / 2) * 8 + col % 2`.
The CTA's FP16 probability tile remains row-major `[kM, kN]`.

Keep 256 threads, `kMaxQTiles * kHeads = 8640` CTAs, zero dynamic shared memory,
and the four caller-stream launches: `attention_scores`, `attention`,
`attention_values`, `attention_epilogue`. Keep reverse key-tile traversal,
online maximum/sum updates, global score/probability slab reuse, global reduction
scratch, scalar QK/PV loops, ragged bounds, and the existing ABI.
Retain SM90a compilation and flags `-O3 -std=c++17 --use_fast_math
--resource-usage -lineinfo -DNDEBUG`.

# Precondition

- Data types: direct consumption must retain the produced values' representation
  and the reduction's addition order and rounding. Changing intermediate precision
  or contraction changes the numerical result even when the formula is identical.
- Layout: producer and sum traversal must visit the same thread-owned row subset
  in the same index order. Otherwise immediate accumulation changes ownership or
  reduction order. Direct scalar consumption adds no contiguity or alignment rule.
- Storage: the producing thread must have access to each exponential and its own
  running partial sum. No observer may require the removed separate traversal;
  its intermediate rereads are what fusion eliminates.
- Pipeline: exponential production must be independent of the evolving partial
  sum, so accumulation can move earlier. Each exponential must precede its addition;
  all sums must finish before dependent consumers. Remaining buffer readers must
  finish before reuse. These dependencies protect value visibility and prevent
  premature consumption or overwrite; fusion does not remove their synchronization.
- Hardware: no additional feature or storage capacity is required. Existing scalar
  operations and the existing partial-sum accumulator execute the fused traversal.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

```cuda
// Keep the sum traversal separate from exponential production.
__device__ __noinline__ float row_sum(const float (&score)[kQkRegs], float sum,
                                     float scale, int row) {
    // Preserve the input compiler's fused rescaling and first addition.
    float result = fmaf(scale, sum, score[row * kPairElems]);

    #pragma unroll 1
    for (int col = 1; col < kQkRegs / kPairElems; ++col) {
        const int i = (col / kPairElems) * kFragmentSize
                      + row * kPairElems + col % kPairElems;
        result += score[i];
    }
    return result;
}

// Per-row tail inside softmax, after the maximum reduction.
scale[row] = exp2f((maximum[row] - m) * kLog2Scale);
maximum[row] = m;
const float finite_max = m == -INFINITY ? 0.f : m;
#pragma unroll
for (int col = 0; col < kQkRegs / 2; ++col) {
    int i = (col / 2) * 4 + row * 2 + col % 2;

    // Recompute the row offset for every score.
    const float scaled = score_offset(finite_max);
    score[i] = exp2f(score[i] * kLog2Scale - scaled);
}

// Sum only after this thread has produced every exponential in the row.
sum[row] = row_sum(score, sum[row], scale[row], row);
```

## After

```cuda
// Delete row_sum. Replace the same per-row tail inside softmax.
scale[row] = exp2f((maximum[row] - m) * kLog2Scale);
maximum[row] = m;
sum[row] *= scale[row];
const float finite_max = m == -INFINITY ? 0.f : m;
#pragma unroll
for (int col = 0; col < kQkRegs / 2; ++col) {
    int i = (col / 2) * 4 + row * 2 + col % 2;

    // Recompute the row offset for every score.
    const float scaled = score_offset(finite_max);
    score[i] = exp2f(score[i] * kLog2Scale - scaled);
    sum[row] += score[i];
}
```

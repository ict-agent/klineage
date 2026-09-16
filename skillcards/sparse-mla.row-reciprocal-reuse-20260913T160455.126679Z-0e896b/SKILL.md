---
skill_id: sparse-mla.row-reciprocal-reuse
intent: Reuse each row normalization reciprocal across its output elements.
preconditions:
- 'Data types: repeated scales must use identical reciprocal arithmetic, precision,
  and zero handling; reuse must preserve the separate output multiplication and rounding.'
- 'Layout: each thread must identify outputs sharing a divisor; incorrect grouping
  applies the wrong scale. No contiguity or alignment requirement is added.'
- 'Storage: finalized divisors must be available in each owning thread before its
  output loop; private scale reuse requires no interthread exchange or additional
  shared storage.'
- 'Pipeline: divisor production must finish before scale computation, and divisors
  must remain unchanged through their output group. No shared buffer is introduced;
  existing accumulator-completion and buffer-reuse ordering remains intact.'
- 'Hardware: no additional feature or shared-memory capacity is required; ordinary
  scalar arithmetic and private scalar storage support reciprocal reuse.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Hoist output-normalization reciprocals out of the per-element loop in
`solution/attention.cuh::store_output<Group>`. Compute one scale per owned row,
then reuse it for every output element belonging to that row.

Delete `output_scale` from `solution/hopper.cuh`. Its volatile `mov.b32` preserves
all divisor bits but hides value identity from the compiler, forcing independent
reciprocal evaluations. Retain all compiler flags.

Insert `float scale[2]` and the row loop immediately before `wait<0>()`.
Replace the inner conversion loop with the After snippet. The index
`i % 4 >= 2` selects the second owned row for fragment positions 2 and 3,
and the first row for positions 0 and 1. Retain the tile loop, global destination
calculation, and `save_output` mapping.

`reduce_sum` has finalized each thread's `l` before `store_output` starts.
Neither row divisor changes during output conversion. Keep `wait<0>()` before
reading `o`: reciprocal computation uses only completed divisors and may precede
that wait. No launch, synchronization, shared layout, buffer lifetime, or bounds
handling changes are needed. Keep all existing waits, fences, and producer/consumer
arrivals. No new shared buffer needs publication or reuse protection.

Retain `l[row] == 0.0f ? 0.0f : 1.0f / l[row]`, followed by FP32 multiplication
and `__float2bfloat16_rn`. Do not substitute direct output division: its rounding
can differ. Preserve the problem, oracle, tolerances, workload, and build flags.

## Example configuration

The supplied sparse MLA prefill has 8192 tokens, 128 heads, QK width 576,
value width 512, and 2048 selected indices. Q/KV/output use BF16; reductions,
reciprocals, max logits, and LSE use FP32. The softmax scale is
`0.1352337788608801f`.

Each CTA owns one token and a 64-head slice. Retain 16384 CTAs, 384 threads,
one-CTA clusters, and 231376 dynamic shared-memory bytes. Two consumer warpgroups
have 128 threads each; a third warpgroup produces KV tiles. Retain SM90 WGMMA,
its unswizzled shared layouts, resident Q, the two KV buffers, asynchronous copies,
overlapping QK/PV scheduling, online softmax, probability-tile aliasing, shared
reductions, and direct global output stores.

Each consumer thread owns two rows and 128 output values in one 256-column half.
`row_index(row, lane)` selects its head within the CTA. `kHalfTiles == 4` and
`kScoreRegs == 32`; each four-register fragment group alternates two values from
each owned row. Reuse reduces reciprocal evaluations from 128 to 2 per consumer
thread. Global output is contiguous `[token, head, value]`; `save_output` retains
its paired-column mapping. These dimensions and layouts configure this replay,
not the general technique.

Retain `--use_fast_math`, all other compile flags, the caller stream, and the
configured `kernel.cu::kernel` destination-passing ABI. Query, output, and sparse
index bounds retain their current checks and masks.

# Precondition

- Data types: every reused scale must represent the same reciprocal expression
  under the same precision, arithmetic mode, and zero rule. Otherwise, hoisting
  can change the value being multiplied. Preserve separate output multiplication
  and final rounding. The technique does not require a particular output dtype.
- Layout: each thread must know which outputs share each divisor. That grouping
  selects the cached scale; grouping unrelated rows would normalize incorrectly.
  No physical contiguity or alignment is required by scalar reuse.
- Storage: each thread must already have its finalized divisor values available
  before the output loop. The hoisted expression needs those values at its new
  location. Scales remain private; no cross-thread exchange or additional shared
  storage is needed.
- Pipeline: finish divisor production before computing scales, and keep each
  divisor unchanged while its outputs reuse the result. Otherwise the cached
  reciprocal is premature or stale. Preserve existing accumulator-completion and
  buffer-reuse ordering. No shared buffer is introduced, so no new publication,
  reader-visibility, or reuse barrier is required.
- Hardware: no additional hardware feature or shared-memory capacity is needed.
  Ordinary scalar arithmetic and private scalar storage implement the reuse;
  the retained matrix instructions are independent of it.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

`solution/hopper.cuh` contains this compiler-control helper:

```cuda
__device__ __forceinline__ float output_scale(float sum) {
    // Keep each reciprocal independent without changing the divisor's bits.
    asm volatile("mov.b32 %0, %1;" : "=f"(sum) : "f"(sum));
    return sum == 0.0f ? 0.0f : 1.0f / sum;
}
```

The body of `solution/attention.cuh::store_output<Group>` is:

```cuda
wait<0>();
#pragma unroll
for (int tile = 0; tile < kHalfTiles; ++tile) {
    Bf16 converted[kScoreRegs];
#pragma unroll
    for (int i = 0; i < kScoreRegs; ++i) {
        // Recompute normalization for each output element.
        const float scale = output_scale(l[i % 4 >= 2]);
        converted[i] = __float2bfloat16_rn(o[tile * kScoreRegs + i] * scale);
    }
    const int global_tile = Group * kHalfTiles + tile;
    Bf16* dest = params.output + (token * kHeads + head) * kValue + global_tile * kTile;
    save_output(converted, dest, lane);
}
```

## After

Delete `output_scale`; replace only the `store_output<Group>` body:

```cuda
float scale[2];
#pragma unroll
for (int row = 0; row < kRowsPerThread; ++row)
    scale[row] = l[row] == 0.0f ? 0.0f : 1.0f / l[row];

wait<0>();
#pragma unroll
for (int tile = 0; tile < kHalfTiles; ++tile) {
    Bf16 converted[kScoreRegs];
#pragma unroll
    for (int i = 0; i < kScoreRegs; ++i)
        converted[i] = __float2bfloat16_rn(o[tile * kScoreRegs + i] * scale[i % 4 >= 2]);
    const int global_tile = Group * kHalfTiles + tile;
    Bf16* dest = params.output + (token * kHeads + head) * kValue + global_tile * kTile;
    save_output(converted, dest, lane);
}
```

---
skill_id: kda.thread-local-decay-factor-reuse
intent: Reuse rounded decay factors across query and key consumers.
preconditions:
- 'Data types: consumers must require the same exponent evaluation and factor rounding;
  otherwise one result cannot replace both evaluations. No particular dtype is inherent
  to reuse.'
- 'Layout: each pair of consumers must belong to the same thread and use the same
  exponent operand; thread-local reuse cannot deliver a factor to another thread.
  No additional contiguity or alignment is required.'
- 'Storage: no additional placement requirement; existing operands may come from any
  memory level because reuse changes only the lifetime of a computed scalar.'
- 'Pipeline: operand producers must finish before factor evaluation, and operands
  must remain unchanged through their consumers; an intervening update invalidates
  reuse. Thread-local factors require no additional barrier, participation rule, or
  buffer-reuse ordering.'
- 'Hardware: no additional feature or shared-memory capacity is required; reuse uses
  ordinary scalar registers and the existing exponent operation.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Reuse each rounded positive decay factor between the decayed query and key,
and each rounded inverse factor between the inverse and residual keys.
This removes duplicate exponent evaluations for the same thread-owned element.

In `solution/prepare.cuh`, replace the four `decay_each` calls in `prepare`'s
innermost decay loop with the two local factors shown below. Delete the now-unused
`decay_each` definition. Use the existing `exp2_fast` from `solution/native.cuh`;
retain its `ex2.approx.ftz.f32` instruction and the explicit BF16 conversion before
multiplication. Preserve the written multiplication order and intermediate BF16
rounding. Do not combine the inverse factor with the chunk-total exponent.

`decay_each` is deliberately `__noinline__` and contains volatile assembly:
inlining duplicated expressions lets the compiler merge evaluations even with
volatile assembly. Restoring reuse removes that compiler-control helper and its
call overhead. Leave all other helpers and compiler flags unchanged.

Keep the gather loop, `rg`, `rgt`, `rq`, `rk`, scalar shared loads/stores, offsets,
launches, and barriers unchanged. The gather completes before the existing CTA
barrier; each thread then consumes its immutable register snapshots. The factors
remain local to that thread and expire after their two uses. No new shared buffer,
visibility operation, or storage-reuse barrier is needed. Existing shared stores
still complete before the following CTA barrier and matrix-product consumers.
The fixed tiling covers valid elements; preserve its bounds handling.

Check correctness and latency with the supplied problem and the existing Kernel
evaluator. Inspect generated code to confirm that the per-consumer calls disappear
and each sign's factor is evaluated once per owned element. Retain the complete
problem, oracle, workload, tolerances, seed, and timing policy.

## Example configuration

The bundle targets `nvidia-sm90a-cuda13` with CUDA and preserves its pybind11
`kernel` ABI, destination outputs, and caller stream. Inputs use contiguous BTHD
with batch 1, 4096 tokens, 96 heads, and dimension 128. Each preparation CTA has
256 threads, handles 16 tokens for one head, and uses 41856 dynamic shared bytes;
the grid is `(kTiles,kHeads)`, where `kTiles=256`.

For `lane=tid%32`, `warp=tid/32`, `group=lane/4`, and `pair=lane%4`, retain
`m,n in [0,2)`, `j in [0,2)`, `t=m*2+n`, `r=m*8+warp`, and
`c=n*64+group*8+pair*2`. Each thread owns eight elements `(r,c+j)`.
`rg[t][j]` is that element's FP32 cumulative log2 gate; `rgt[t][j]` is its
FP32 chunk total. `rq` and `rk` hold normalized BF16 query/key values.
The output pairs use `offset<kChunk>(r,c)` and retain BF16 storage.

Keep `kScale=0.08838834764831843f`, BF16 multiplication boundaries, normalization,
triangular inversion, BF16 tensor-core products with FP32 accumulators, and
chunked recurrence unchanged. Preserve the recurrence's 256 threads, 124672
dynamic shared bytes, global BF16 carry, and stream-ordered launch per chunk.
These settings describe this replay, not requirements for factor reuse.

# Precondition

- Data types: paired consumers need identical exponent semantics and factor
  rounding. A result rounded or evaluated differently cannot replace both
  computations. Reuse itself imposes no particular dtype.
- Layout: the paired consumers must use the same exponent operand within one
  thread. Another thread cannot read that thread's local factor directly.
  Contiguity and alignment add no requirements to this register-only change.
- Storage: no additional placement requirement applies. Operands may come from
  any existing memory level; reuse changes only a computed scalar's lifetime,
  without requiring a memory-level transfer or shared buffer.
- Pipeline: producers finish before factor evaluation, and no operand update may
  intervene between consumers; otherwise the shared result can become stale.
  Local factors add no barrier, participation requirement, or buffer-reuse
  ordering. Existing producer visibility and shared-storage ordering remain.
- Hardware: no additional hardware feature or shared-memory capacity is needed.
  The transformation uses ordinary scalar registers and the exponent operation
  already used by the separate evaluations.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Both inner-loop snippets use the existing `t`, `j`, `qd`, `kd`, `ki`, `kr`,
`rq`, `rk`, `rg`, and `rgt`. The helper in Before is at namespace scope;
delete it when applying After. All surrounding loops and stores stay unchanged.

## Before

```cuda
// Namespace scope: preserve distinct per-consumer evaluations.
__device__ __noinline__ BF16 decay_each(float x) {
    float value;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(value) : "f"(x));
    return BF16(value);
}

// Inside prepare's innermost decay loop.
qd.h[j] = rq[t][j] * decay_each(rg[t][j]) * BF16(kScale);
kd.h[j] = rk[t][j] * decay_each(rg[t][j]);
ki.h[j] = rk[t][j] * decay_each(-rg[t][j]);
kr.h[j] = rk[t][j] * decay_each(-rg[t][j]) * BF16(exp2_fast(rgt[t][j]));
```

## After

```cuda
BF16 e = BF16(exp2_fast(rg[t][j]));
qd.h[j] = rq[t][j] * e * BF16(kScale);
kd.h[j] = rk[t][j] * e;
BF16 ie = BF16(exp2_fast(-rg[t][j]));
ki.h[j] = rk[t][j] * ie;
kr.h[j] = rk[t][j] * ie * BF16(exp2_fast(rgt[t][j]));
```

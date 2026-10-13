---
skill_id: normalization.reciprocal-norm-hoisting
intent: Hoist loop-invariant reciprocal norms for reuse across normalized elements.
preconditions:
- 'Data types: repeated reciprocal-norm evaluations must use identical operands, precision,
  and rounding; otherwise reusing one result can change normalized values.'
- 'Layout: one thread must normalize multiple elements sharing a completed norm, enabling
  reuse without communication; no additional contiguity or alignment is required.'
- 'Storage: norm operands must already be available to their consuming thread; no
  additional memory placement is required because only scalar results are reused locally.'
- 'Pipeline: reductions and producer visibility must precede norm evaluation, and
  norm operands must remain invariant through all uses; retain ordering before shared-input
  reuse and normalized-output consumption.'
- 'Hardware: no additional feature is required; existing scalar reciprocal-square-root
  arithmetic and per-thread storage suffice.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Hoist reciprocal-norm evaluation out of the per-element normalization loop in
`solution/prepare.cuh::prepare`. Each thread currently calls `norm_each(qs)` and
`norm_each(ks)` for every owned element, although its completed squared sums do
not change. Compute `qi = rsqrtf(qs+kNormEps)` and `ki = rsqrtf(ks+kNormEps)` once
after the XOR reduction, then multiply every owned query/key element by its
corresponding inverse. Delete `norm_each` from the same file.

The helper's `__noinline__` and volatile PTX enforce repeated evaluation in the
deoptimized bundle. Removing the helper and both calls is part of this replay;
retain all other compiler controls and compile flags. With the retained fast-math
build, `rsqrtf` supplies the same approximate reciprocal-square-root operation as
the helper. Preserve epsilon addition, FP32 multiplication, and the subsequent
BF16 conversion. Do not change squared-sum accumulation or XOR reduction order.

Keep volatile scalar query/key reloads, shared-memory addresses, ownership,
barriers, and launches unchanged. Each thread reads and overwrites only its own
normalized elements. Reduction exchanges finish before inverse evaluation; the
following CTA barrier publishes all normalized rows before later readers. No new
buffer or synchronization is needed, and existing barriers continue to protect
shared storage before reuse. Keep all existing bounds handling; this fixed
workload has complete rows and chunks.

Validate replay with `klineage.harness.evaluate` on the unchanged problem. Inspect
generated preparation code to confirm one query and one key reciprocal per
thread, reused across its elements. Keep validation and measurement evidence
outside this card.

## Example configuration

Preserve `BATCH=1`, `TOKENS=4096`, `HEADS=96`, `HEAD_DIM=128`, `kChunk=16`,
`kPrepareThreads=256`, `kNormElems=8`, and `kNormEps=1e-6f`. Preparation launches
`grid=(kTiles,kHeads)`, `block=kPrepareThreads`, and
`shared_bytes=sizeof(PrepareShared)`; retain its launch bounds and buffer layout.
Thread `tid` owns row `tid/16`, starting column `(tid%16)*8`, and eight consecutive
elements of each row-major shared query/key matrix. Each row's sixteen threads
reduce local FP32 squared sums using XOR partners `8,4,2,1`; every participant
receives the completed sum. Keep both per-thread sum values unchanged.

Inputs and normalized shared query/key values are BF16; sums and inverse norms
are FP32. Retain all later BF16 rounding, tensor-core products, triangular
substitution, shared staging, chunk scheduling, and caller-stream ordering.
Keep the configured tensor ABI, full problem/oracle, and compile flags, including
`--use_fast_math`. These are replay settings, not prerequisites for scalar reuse.

# Precondition

- Data types: every repeated reciprocal-norm evaluation must have identical
  operands and use the same precision and rounding. A cached value can replace
  those evaluations only if it preserves each result and the subsequent
  normalization arithmetic; changing precision during hoisting can change output.
- Layout: a thread must own multiple normalization consumers of the same completed
  norm. This supplies reuse within that thread without communication. Contiguity
  and alignment add no requirement: they affect addressing, which stays unchanged.
- Storage: norm operands must be available to the consuming thread before it
  computes the reusable scalar. No particular shared/global placement or new
  buffer is required; unavailable operands would prevent the hoisted evaluation.
- Pipeline: finish norm reductions and establish producer visibility before
  evaluating inverses. Keep their operands invariant until the last use, or a
  single cached value would be stale. Preserve ordering that finishes shared-input
  reads before storage reuse and publishes normalized writes before consumers;
  hoisting does not replace those synchronization duties.
- Hardware: no additional feature is required. The existing scalar reciprocal
  square root and per-thread storage can compute and retain the inverse values;
  no collective instruction or extra shared-memory capacity is introduced.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The first Before block is the namespace-scope helper to delete. Replace the
normalization loop shown next with After, immediately following the completed
XOR reduction in `prepare`. Retain its surrounding barriers and all other code.

## Before

```cuda
// Keep reciprocal norms per element; calls prevent compiler reuse.
__device__ __noinline__ float norm_each(float sum) {
    float inverse;
    asm volatile("rsqrt.approx.ftz.f32 %0, %1;"
      : "=f"(inverse) : "f"(sum+kNormEps));
    return inverse;
}
```

```cuda
#pragma unroll
for (int j = 0; j < kNormElems; ++j) {
    // Reload owned inputs; volatile loads prevent compiler register reuse.
    float q = bf_float(ld_scalar(s.q+row*kDim+col+j));
    float k = bf_float(ld_scalar(s.k+row*kDim+col+j));
    // Recompute each norm without changing its reduction or rounding.
    s.q[row*kDim+col+j] = BF16(q*norm_each(qs));
    s.k[row*kDim+col+j] = BF16(k*norm_each(ks));
}
```

## After

```cuda
float qi = rsqrtf(qs+kNormEps), ki = rsqrtf(ks+kNormEps);
#pragma unroll
for (int j = 0; j < kNormElems; ++j) {
    // Reload owned inputs; volatile loads prevent compiler register reuse.
    float q = bf_float(ld_scalar(s.q+row*kDim+col+j));
    float k = bf_float(ld_scalar(s.k+row*kDim+col+j));
    s.q[row*kDim+col+j] = BF16(q*qi);
    s.k[row*kDim+col+j] = BF16(k*ki);
}
```

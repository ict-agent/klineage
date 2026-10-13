---
skill_id: kda.thread-local-squared-norm-reuse
intent: Reuse loop-invariant squared-norm reductions across a thread's normalized
  elements.
preconditions:
- 'Data types: repeated reductions must yield identical values under the same conversions,
  reduction order, and rounding; otherwise reusing one result changes normalization.
  No particular element dtype is required by hoisting.'
- 'Layout: multiple consumers in one thread must use the same reduction addresses
  and ordering; differing rows or lane-dependent trees need separate results. No additional
  contiguity or alignment is required.'
- 'Storage: reduction operands must be readable without required per-read side effects;
  eliminating observable reads would change behavior. Existing thread-local scalar
  results can serve later consumers.'
- 'Pipeline: operand producers must finish and make values visible before the first
  reduction, and inputs must remain unchanged through the last consumer; otherwise
  a reused result is stale. No communication buffer or new barrier is required.'
- 'Hardware: no additional feature or shared-memory capacity is required; ordinary
  scalar computation and thread-local storage implement this reuse.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Hoist the Q/K squared-norm reductions out of the normalization element loop in
`solution/prepare.cuh::prepare`. Each thread computes the same pair of reductions
for every element it owns. Compute that pair once and retain it for all consumers.

Apply these edits only in `solution/prepare.cuh`:

1. In the leaf of `norm_tree`, replace its two `ld_global_scalar` calls with
   ordinary `args.q[src]` and `args.k[src]` reads. These volatile loads only prevent
   the compiler from restoring reduction reuse in the deoptimized version.
2. Delete `norm_sum_each`, including its comment. Its non-inlined call enforces
   repeated evaluation and is unnecessary after hoisting.
3. Move the `NormSum sums` and `qs, ks` declarations before the normalization
   loop, calling `norm_tree<1>` directly as shown below.

Retain the complete `norm_tree` recursion: each leaf accumulates increasing `j`,
and each parent adds the `lane` subtree to the `lane ^ Delta` subtree. Preserve
BF16-to-FP32 conversion, FP32 arithmetic, compiler arithmetic flags, per-element
`norm_each` calls, epsilon, and final BF16 conversion. Do not move reciprocal
square roots out of the loop or share reductions between threads.

Input addressing, output ownership, launch dimensions, and barriers are unchanged.
Inputs are already visible on the caller stream. Each thread writes its normalized
Q/K elements to shared memory; the existing CTA barrier publishes them to decay
preparation. The reduction results are private scalars, with no communication
scratch to synchronize or reuse. Preserve all later shared-buffer barriers.

## Example configuration

Retain B=1, T=4096, H=96, D=128, chunk size 16, and 256 chunks. Q/K are contiguous
BF16 BTHD tensors, with row base `((tile*kChunk+row)*kHeads+head)*kDim` and byte
stride `kHeads*kDim*sizeof(BF16)` between token rows. The working sums are FP32.

Keep `kNormElems=8`, `kNormLanes=kDim/kNormElems=16`, and `NormSum {q,k}`.
Thread `tid` owns row `tid/kNormLanes`, columns
`(tid%kNormLanes)*kNormElems + j`, and reduction-tree lane `tid%kNormLanes`.
Each reduction covers the full row, using sixteen leaves of eight elements.
The optimized thread reuses its two sums across its eight element iterations.

Keep preparation grid `(kTiles,kHeads)=(256,96)`, 256 threads, launch bounds
`(kPrepareThreads,8)`, and 41856 dynamic shared-memory bytes. The supplied dimensions
give complete rows and tiles; retain the existing fixed-shape ABI checks. Do not
introduce padding or change bounds handling. Other shapes need valid row ownership
and bounds around both reduction reads and normalized stores.

Retain `kNormEps=1e-6f`, the existing approximate reciprocal square root and BF16
rounding, and all `compile_flags`, including fast math. Keep per-element volatile
raw-input reads and every other `ld_global_scalar` use. Retain scalar fragment
transfers, shared normalized operands, gate-prefix recomputation, triangular
inversion, BF16 MMA with FP32 accumulation, shared recurrence state, and the 256
ordered recurrence launches with 256 threads and 124672 dynamic shared-memory
bytes each. The configured callable, tensor ABI, caller stream, and complete
problem remain unchanged.

Check correctness and timing with `klineage.harness.evaluate` using the unchanged
problem. Inspect generated preparation code: the full squared-norm reduction
should execute once per thread before all element consumers, while reciprocal
square roots remain per element. Keep measurements outside this card.

# Precondition

- Data types: each repeated reduction must produce the same value with unchanged
  conversions, reduction grouping, and rounding. Hoisting selects one evaluation
  for every consumer; nondeterministic or differently rounded evaluations cannot
  be substituted. The technique itself does not require a particular dtype.
- Layout: consumers in a thread must use identical reduction addresses and
  ordering. A different row or lane-dependent addition tree may produce a different
  result and needs its own reduction. Hoisting adds no alignment or contiguity rule.
- Storage: operand reads must have no required per-read side effects. Removing
  observable reads would change behavior; the volatile qualifiers here only enforce
  deoptimization. The existing scalar results are thread-local and can be retained
  for later consumers without moving input storage.
- Pipeline: producers must complete and publish inputs before the first reduction.
  Inputs must stay unchanged until the final consumer; mutation would invalidate
  reuse. There is no communication buffer to overwrite and no additional barrier
  or buffer-reuse dependency introduced by retaining private scalar results.
- Hardware: no additional hardware feature or shared-memory capacity is needed.
  Ordinary scalar instructions and thread-local storage suffice.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The first fragment is inside the existing `norm_tree` leaf. The second is the
helper definition. The last replaces the normalization loop after `norm_base` is
defined. Retain the surrounding row/column calculation and following CTA barrier.

## Before

```cuda
// norm_tree leaf; retain the surrounding loop and additions.
float q = bf_float(ld_global_scalar(args.q+src));
float k = bf_float(ld_global_scalar(args.k+src));

// Recompute each element's reduction; calls and volatile loads prevent reuse.
__device__ __noinline__ NormSum norm_sum_each(
    const PrepareArgs& args, int base, int lane) {
    return norm_tree<1>(args,base,lane);
}

// prepare, immediately after norm_base.
#pragma unroll
for (int j = 0; j < kNormElems; ++j) {
    const NormSum sums = norm_sum_each(args,norm_base,tid % kNormLanes);
    float qs = sums.q, ks = sums.k;
    const int src = ((tile*kChunk+row)*kHeads+head)*kDim+col+j;
    float q = bf_float(ld_global_scalar(args.q+src));
    float k = bf_float(ld_global_scalar(args.k+src));
    s.q[row*kDim+col+j] = BF16(q*norm_each(qs));
    s.k[row*kDim+col+j] = BF16(k*norm_each(ks));
}
```

## After

```cuda
// norm_tree leaf; retain the surrounding loop and additions.
float q = bf_float(args.q[src]);
float k = bf_float(args.k[src]);

// Delete norm_sum_each; retain norm_tree and norm_each.

// prepare, immediately after norm_base: reuse reductions within this thread.
const NormSum sums = norm_tree<1>(args,norm_base,tid % kNormLanes);
float qs = sums.q, ks = sums.k;
#pragma unroll
for (int j = 0; j < kNormElems; ++j) {
    const int src = ((tile*kChunk+row)*kHeads+head)*kDim+col+j;
    float q = bf_float(ld_global_scalar(args.q+src));
    float k = bf_float(ld_global_scalar(args.k+src));
    s.q[row*kDim+col+j] = BF16(q*norm_each(qs));
    s.k[row*kDim+col+j] = BF16(k*norm_each(ks));
}
```

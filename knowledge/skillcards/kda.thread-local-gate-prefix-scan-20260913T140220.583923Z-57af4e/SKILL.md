---
skill_id: kda.thread-local-gate-prefix-scan
intent: Reuse cumulative gate prefixes with a thread-local running sum.
preconditions:
- 'Data types: retain the accumulator representation, term evaluation, and ordered
  additions; changing rounding or reassociating a reused prefix can change results.'
- 'Layout: one thread owns an ordered sequence and its overlapping prefixes, with
  known indexing; prefix reuse requires consecutive outputs to extend the same sequence.'
- 'Storage: term inputs remain stable and readable throughout prefix production; otherwise
  reusing a prior sum can differ from recomputation. No additional memory-level placement
  is required.'
- 'Pipeline: input producers complete and become visible before prefix production;
  prefix endpoints are visited in increasing order, and consumers and buffer reuse
  wait for their dependencies. This prevents stale terms, incomplete outputs, and
  overwrites.'
- 'Hardware: no additional feature or capacity requirement; the running scan uses
  the scalar accumulator already required by independent prefix computation.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Reuse the preceding cumulative gate when producing the next prefix in
`solution/prepare.cuh::prepare`. Replace the `if (tid < kDim)` block shown below.
Move the accumulator reset outside the row loop, remove the inner prefix loop,
and evaluate only the new row's term. For a sequence of length N, this reduces
term evaluations from N(N+1)/2 to N without changing the left-to-right sum.

Each participating thread owns one key channel for one head and chunk. It reads
`s.gate[row * kDim + tid]`, writes the matching FP32 log2 prefix to
`s.g[row * kDim + tid]`, and writes the last prefix to `s.gt[tid]`. Preserve
`expf`, the volatile gain and bias loads, `sigmoid`, `kGateScale`, and the
accumulation expression. The saved prefix remains FP32: do not quantize it,
change the addition grouping, replace the scan with a tree, or move factor
computations out of the row loop.

The deoptimized loops use `#pragma unroll 1` so compilation retains independent
prefix evaluation; their volatile input loads also prevent merging repeated
term evaluations. Apply the After block's original unrolling hint to the single
running scan. No compiler-flag change is needed.

Input staging completes before the existing block barriers. Preserve those
barriers and the barriers after gate production: later decay consumers read the
shared prefixes, and the shared buffers must remain intact until they finish.
The scan is thread-private and adds no communication, allocation, synchronization,
or launch change. Keep the `tid < kDim` guard. Row bounds remain
`0 <= r < kChunk`; for this full-chunk workload no partial-row handling is needed.

After replay, use the existing Kernel evaluator with the unchanged problem for
correctness and latency. Inspect generated gate code for one carried sum and
one term evaluation per row. Keep measurements outside this card.

## Example configuration

The problem is `kda_prefill`, CUDA on `nvidia-sm90a-cuda13`, with batch 1,
4096 tokens, 96 heads, and head dimension 128. Preserve chunks of 16 rows,
256 preparation threads, and grid `(kTiles, kHeads) = (256, 96)`.
Threads 0 through 127 each scan one channel. Gate terms drop from 136 to 16
per participating thread. Input gate storage is BF16; bias, gain input,
accumulator, prefixes, and chunk totals are FP32. `s.gate` and `s.g` are
row-major `[kChunk, kDim]` shared arrays; `s.gt` has `kDim` elements.
Bias and gain inputs remain global. Keep `kGateScale = -5 * log2(e)` and
all existing BF16 conversions at downstream decay and matrix-product boundaries.

Retain `PrepareShared` (41856 bytes), the inverse working-row and transpose
scratch, and `RecurShared` (124672 bytes). Preserve query/key normalization,
shared operand staging, BF16 tensor-core products with FP32 accumulators,
triangular inversion, correction reuse, and the sequential recurrence launches
on the caller stream. Keep the ABI, complete problem, build flags, bounds checks,
seed, tolerances, and timing policy unchanged. These are replay settings and
retained mechanisms, not prerequisites of prefix reuse.

# Precondition

- Data types: keep the same accumulator representation and each term's arithmetic.
  Reusing a saved prefix is equivalent to recomputation only with the same
  rounding and ordered additions. No particular input dtype is required by
  reuse itself; narrowing the saved prefix or reassociating additions breaks
  this equivalence.
- Layout: a thread owns one sequence and its overlapping prefix outputs.
  Known indexing must identify the next term and corresponding output.
  Consecutive prefixes must extend the same ordered sequence; independent
  sequences cannot share a running sum. No extra contiguity or alignment is
  required by the scalar scan.
- Storage: term inputs stay readable and stable throughout prefix production.
  Mutating an earlier term would make its recomputed prefix differ from the
  reused value. No additional memory-level placement is required; reuse
  retains a private accumulator already present in the preceding implementation.
- Pipeline: producers must finish and make inputs visible before scanning.
  Visit prefix endpoints in increasing order so each saved sum is the next
  output's predecessor. Consumers wait for their prefixes, and source/output
  buffers cannot be overwritten while dependent reads remain. These are data
  dependencies; the technique requires no fixed stage count or new barrier.
- Hardware: no additional feature or capacity requirement. Ordinary scalar
  arithmetic and the existing accumulator suffice; tensor-core instructions
  used elsewhere are independent of this scan transformation.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Replace only this block inside `prepare`; surrounding declarations, barriers,
and consumers remain unchanged. All referenced helpers and constants already
exist in the deoptimized bundle.

## Before

```cuda
if (tid < kDim) {
    float sum = 0.0f;
    // Recompute each prefix; rolled loops keep prefix reuse absent.
    #pragma unroll 1
    for (int r = 0; r < kChunk; ++r) {
        sum = 0.0f;
        #pragma unroll 1
        for (int p = 0; p <= r; ++p) {
            // Retain per-term gain and bias loads and arithmetic order.
            float gain = expf(ld_global_scalar(a_log+head));
            float bias = ld_global_scalar(args.bias+head*kDim+tid);
            float g = gain * (bf_float(s.gate[p*kDim+tid])+bias);
            sum += kGateScale * sigmoid(g);
        }
        s.g[r*kDim+tid] = sum;
    }
    s.gt[tid] = sum;
}
```

## After

```cuda
if (tid < kDim) {
    float sum = 0.0f;
    #pragma unroll
    for (int r = 0; r < kChunk; ++r) {
        // Reload the gain input per row to prevent compiler hoisting.
        float gain = expf(ld_global_scalar(a_log+head));
        // Reload each row's bias; volatile loads prevent register reuse.
        float bias = ld_global_scalar(args.bias+head*kDim+tid);
        float g = gain * (bf_float(s.gate[r*kDim+tid])+bias);
        sum += kGateScale * sigmoid(g);
        s.g[r*kDim+tid] = sum;
    }
    s.gt[tid] = sum;
}
```

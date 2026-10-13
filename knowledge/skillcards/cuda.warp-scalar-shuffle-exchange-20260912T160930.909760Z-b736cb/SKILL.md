---
skill_id: cuda.warp-scalar-shuffle-exchange
intent: Exchange scalar values directly between warp lanes with shuffle instructions.
preconditions:
- 'Data types: exchanged values need a lossless shuffle-supported register representation;
  transferring them must not introduce arithmetic conversions.'
- 'Layout: each consumer has a known source lane in the same 32-lane CUDA warp; shuffles
  cannot exchange across warps. Register exchange adds no memory-contiguity or alignment
  requirement.'
- 'Storage: producers already hold outgoing values in registers, and the shared scratch
  serves only these lane exchanges; other scratch readers would prevent its removal.'
- 'Pipeline: every lane named in the mask, including each source, must reach the same
  exchange after producing its value. Existing scratch barriers publish writes and
  finish reads before reuse; remove them only with all corresponding scratch accesses,
  retaining ordering for other shared data.'
- 'Hardware: synchronized CUDA warp shuffles must be supported; they replace scratch
  traffic without requiring additional shared-memory capacity.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Replace shared-memory scalar exchanges with CUDA warp shuffles in
`solution/prepare.cuh`: the XOR reductions in `prepare` and pivot broadcasts in
`invert`. Each source already holds the exchanged value in a register. Shuffles
remove scratch writes, reads, and their two warp barriers without changing
arithmetic or lane ownership.

In `prepare`, replace the complete exchange block inside the `delta` loop with
two `__shfl_xor_sync` additions. Keep the loop order and full mask: its XOR partners
stay within each row's lane group. In `invert`, replace the pivot write, both
warp barriers, and scratch read with `__shfl_sync`; retain the subsequent
conditional `fmaf`. The pivot source is `(tid & ~7) | step`. Only the first warp
enters `invert`, so its thread IDs are also lane IDs.

Remove `norm_q`, `norm_k`, and `pivot`, and their associated comment, from
`PrepareShared` in `solution/native.cuh`. Restore its size assertion as shown
below. No host edit is needed: `solution/native.cu::launch` already uses
`sizeof(PrepareShared)` for the dynamic shared-memory attribute and launch.
Keep all other fields, offsets, launches, maps, and compile flags unchanged.

Remove only the barriers surrounding these eliminated scalar exchanges.
Retain the CTA barriers that publish normalization and gate results, the
`invert` warp barrier before matrix loads, and every async fence and pipeline
barrier. No new buffers need initialization or reuse tracking. Neither loop
adds bounds handling: retain the fixed launch and existing `tid >= kWarp`
return, and do not predicate shuffle execution on `i > step`.

Preserve every FP32 addition and `fmaf`, the XOR reduction tree, substitution
order, BF16 conversion point, and matrix operation. The optimization transfers
bits; it does not change the numerical approximation.

## Example configuration

The supplied workload is `kda_prefill`, with batch 1, 4096 tokens, 96 heads,
head dimension 128, and 16-token chunks. `prepare` launches grid `(256,96)` with
256 threads and `__launch_bounds__(kPrepareThreads,8)`. Shared storage shrinks
from 42368 to 40192 bytes. The removed arrays contain two sets of 256 FP32
partials and 32 FP32 pivots: 2176 bytes, including the same final struct padding
as before.

Each normalization row uses 16 threads, each holding eight FP32 query and key
values. Two rows share a warp; XOR deltas are 8, 4, 2, 1. Each inversion pivot
comes from one of four eight-lane groups; preserve `step < 7` and `p < step`.
`kAllLanes` remains `0xffffffffu`. These are replay settings, not general shuffle
prerequisites.

Retain BF16 input and intermediate rounding, FP32 scalar work and MMA
accumulators, eight-column shared slabs, `ldmatrix`/`stmatrix`/`movmatrix`,
`mma.sync`, TMA transfers, and separate phase storage. Recurrence retains 192
threads, 128 compute threads, three input stages, two output stages, and 160768
shared bytes. Keep the caller CUDA stream, tensor ABI, scale/gate constants,
fast-math build settings, and complete problem contract.

# Precondition

- Data types: exchanged values must have a lossless representation accepted by
  CUDA shuffles. The operation only moves register words; floating-point format
  and arithmetic remain unchanged. Adding conversions would alter the bits
  supplied to the existing arithmetic.
- Layout: every consumer's source lane must be known and lie in the same
  32-lane CUDA warp. Shuffles cannot access another warp. Global or shared-memory
  contiguity and alignment add no requirement once the source is in a register.
- Storage: each outgoing value already resides in its producer's register.
  The eliminated arrays must have no readers beyond these exchanges; otherwise
  removing their writes or allocations would strand those readers.
- Pipeline: all lanes named by the common mask, including the source lane,
  must execute the same collective after producing their values. In the preceding
  implementation, a warp barrier publishes scratch writes and another finishes
  reads before reuse. Both become removable only when the whole corresponding
  scratch exchange disappears. Ordering for other shared producers, consumers,
  and buffer reuse must remain; a shuffle does not replace those memory barriers.
- Hardware: the target must support synchronized CUDA warp-shuffle instructions.
  They use registers already holding the exchanged values and require no new
  shared-memory capacity.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The blocks below are three edits at the named locations; intervening source
stays unchanged. The inversion conditional and normalization loop are included
to make arithmetic order and collective participation explicit.

## Before

```cuda
// solution/native.cuh: tail of PrepareShared, after ready.
// Separate lane exchanges preserve the original reduction grouping.
float norm_q[kPrepareThreads], norm_k[kPrepareThreads];
float pivot[kWarp];
// Existing size assertion after the shared structs:
static_assert(sizeof(PrepareShared) == 42368);

// solution/prepare.cuh: prepare normalization loop.
#pragma unroll
for (int delta = 8; delta >= 1; delta >>= 1) {
    // Exchange the same XOR partners without changing additions.
    s.norm_q[tid] = qs;
    s.norm_k[tid] = ks;
    __syncwarp(kAllLanes);
    qs += s.norm_q[tid ^ delta];
    ks += s.norm_k[tid ^ delta];
    __syncwarp(kAllLanes);
}

// solution/prepare.cuh: invert inner loop.
#pragma unroll
for (int p = 0; p < step; ++p) {
    // Publish each pivot, then finish reads before scratch reuse.
    s.pivot[tid] = inv[p];
    __syncwarp(kAllLanes);
    float pivot = s.pivot[(tid & ~7) | step];
    __syncwarp(kAllLanes);
    if (i > step) inv[p] = fmaf(scale, pivot, inv[p]);
}
```

## After

```cuda
// solution/native.cuh: remove the three exchange arrays and their comment.
// Keep the remaining PrepareShared fields and other structs unchanged.
static_assert(sizeof(PrepareShared) == 40192);

// solution/prepare.cuh: prepare normalization loop.
#pragma unroll
for (int delta = 8; delta >= 1; delta >>= 1) {
    qs += __shfl_xor_sync(kAllLanes,qs,delta);
    ks += __shfl_xor_sync(kAllLanes,ks,delta);
}

// solution/prepare.cuh: invert inner loop.
#pragma unroll
for (int p = 0; p < step; ++p) {
    float pivot = __shfl_sync(kAllLanes, inv[p], (tid & ~7) | step);
    if (i > step) inv[p] = fmaf(scale, pivot, inv[p]);
}
```

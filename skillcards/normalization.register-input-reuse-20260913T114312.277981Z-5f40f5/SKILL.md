---
skill_id: normalization.register-input-reuse
intent: Reuse normalization inputs in registers across reduction and scaling.
preconditions:
- 'Data types: no additional dtype restriction; both reads and their conversions must
  yield identical compute values, so caching preserves representation and arithmetic.'
- 'Layout: both passes must use the same elements in the same owning thread; otherwise
  a private cache supplies the wrong values. No additional contiguity or alignment
  is needed.'
- 'Storage: ordinary shared-memory operands are reread by their owners; their values
  must remain unchanged between reads, since a private cache cannot observe intervening
  writes.'
- 'Pipeline: producers must finish and publish operands before capture; cached values
  must remain live until use, and shared-buffer reuse must wait for uncached readers.
  These dependencies prevent stale values and premature overwrites.'
- 'Hardware: enough per-thread and CTA register allocation for cached values plus
  other live state; spilling the cache defeats register-resident reuse. No additional
  instruction-set feature is required.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Cache normalization inputs in per-thread registers across the reduction, eliminating
the second shared-memory read and BF16-to-FP32 conversion of each q/k element.

In `solution/prepare.cuh`, replace the normalization block in `prepare`, from its
`float qs` declaration through the barrier after normalized stores, with After.
Each thread captures its owned q/k elements during the sum-of-squares pass and
reuses them when scaling. Keep the intervening shared-memory XOR reduction intact.
The complete replacement block appears below; surrounding code stays unchanged.

The Before block deliberately calls `ld_scalar` on its second pass. That helper's
volatile shared loads prevent the compiler from recovering the removed register
reuse. Replace only these normalization calls with cached array accesses; retain
`ld_scalar` and its other callers. Keep compile flags unchanged. Constant array
indices and the existing unrolling expose the cache as scalar registers.

Ownership, row-major shared q/k layout, bounds, launch, and synchronization stay
unchanged. The producer barrier precedes the first pass; each thread alone writes
its owned elements after the reduction. Retain the reduction's publication/read
barriers and the final CTA barrier before subsequent consumers read normalized q/k.
Do not remove any barrier or reuse shared storage earlier.

Preserve increasing-j accumulation, XOR addition partners and order, `kNormEps`,
`rsqrtf`, FP32 multiplication, and the existing BF16 output rounding. The cached
values must equal the repeated conversion of the unchanged shared operands.
Inspect generated code to confirm the scaling pass uses retained registers without
q/k reloads or cache spills. Check correctness and latency with the Kernel evaluator
on the unchanged problem; retain its oracle, workload, tolerances, seed, and timer.

## Example configuration

The workload is B=1, T=4096, H=96, D=128. `kChunk=16`, `kTiles=256`,
`kPrepareThreads=256`, `kNormElems=8`, and `kWarp=32` stay unchanged.
Thread `tid` owns row `tid/16` and columns `(tid%16)*8 + j`, for `0 <= j < 8`.
Thus every thread owns valid elements and needs no new bounds guard.
Shared q/k are BF16 row-major `[kChunk,kDim]` with row stride
`kDim*sizeof(BF16)`; the cache holds sixteen FP32 values per thread.
The norm reduction uses XOR distances 8,4,2,1 and `kNormEps=1e-6f`.

Retain preparation launch `grid=(kTiles,kHeads)`, 256 threads, 42368 dynamic shared
bytes, and `__launch_bounds__(kPrepareThreads,8)`. Retain recurrence launch
`grid=(1,kHeads)`, 256 threads, and 124672 dynamic shared bytes. These are replay
settings, not prerequisites of register caching. Preserve the SM90 CUDA build and
all compiler flags, including fast math and the register-usage option.

Retain BF16 MMA with FP32 accumulators, scalar fragment transfers, shared transpose
scratch, eight-column intermediate slabs, gate preparation, chunked recurrence,
workspace publication, and all state/output conversions. Keep the destination
passing pybind11 ABI, caller device/stream, and stream-owned scratch allocation.

# Precondition

- Data types: no additional dtype restriction. Capturing a value replaces its later
  load and conversion, so both must yield the same compute representation. Preserve
  the arithmetic sequence and rounding; caching does not authorize precision changes.
- Layout: each later use must refer to the same element captured by that thread.
  Private registers cannot supply another thread's value without communication.
  No new alignment or contiguity is needed because address mapping is unchanged.
- Storage: the preceding implementation rereads ordinary shared-memory operands.
  Those operands must remain unchanged between reads; otherwise caching would hide
  an update. The Before block's volatile loads enforce deoptimization and are not
  a requirement to observe external changes.
- Pipeline: producers finish and make operands visible before capture. Cached values
  stay live through their last use. Preserve completion of uncached readers before
  shared-buffer overwrite or reuse; otherwise readers can see incomplete or replaced
  data. No particular stage count follows from these dependencies.
- Hardware: available per-thread and CTA register allocation must accommodate the
  cached values and other simultaneously live state. Cache demand in register words
  is the sum of each retained value's register footprint. Spilling defeats the
  intended register reuse; no additional instruction-set feature is required.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

```cuda
    float qs = 0.0f, ks = 0.0f;
    #pragma unroll
    for (int j = 0; j < kNormElems; ++j) {
        float q = bf_float(s.q[row*kDim+col+j]);
        float k = bf_float(s.k[row*kDim+col+j]);
        qs += q*q;
        ks += k*k;
    }
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
    float qi = rsqrtf(qs+kNormEps), ki = rsqrtf(ks+kNormEps);
    #pragma unroll
    for (int j = 0; j < kNormElems; ++j) {
        // Reload owned inputs; volatile loads prevent compiler register reuse.
        float q = bf_float(ld_scalar(s.q+row*kDim+col+j));
        float k = bf_float(ld_scalar(s.k+row*kDim+col+j));
        s.q[row*kDim+col+j] = BF16(q*qi);
        s.k[row*kDim+col+j] = BF16(k*ki);
    }
    __syncthreads();
```

## After

```cuda
    float q[kNormElems], k[kNormElems], qs = 0.0f, ks = 0.0f;
    #pragma unroll
    for (int j = 0; j < kNormElems; ++j) {
        q[j] = bf_float(s.q[row*kDim+col+j]);
        k[j] = bf_float(s.k[row*kDim+col+j]);
        qs += q[j]*q[j];
        ks += k[j]*k[j];
    }
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
    float qi = rsqrtf(qs+kNormEps), ki = rsqrtf(ks+kNormEps);
    #pragma unroll
    for (int j = 0; j < kNormElems; ++j) {
        s.q[row*kDim+col+j] = BF16(q[j]*qi);
        s.k[row*kDim+col+j] = BF16(k[j]*ki);
    }
    __syncthreads();
```

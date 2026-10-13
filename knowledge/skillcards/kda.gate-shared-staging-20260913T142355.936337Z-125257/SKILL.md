---
skill_id: kda.gate-shared-staging
intent: Reuse raw gate inputs through cooperative shared-memory staging.
preconditions:
- 'Data types: staging must copy gate representations exactly; no arithmetic dtype
  is required by the transfer itself, because conversion and arithmetic remain at
  the consumer.'
- 'Layout: repeated gate reads belong to one CTA, and known input strides and ownership
  permit complete cooperative coverage and matching shared indices; scalar transfers
  need only element alignment.'
- 'Storage: gate values reside in global memory and remain stable through their CTA''s
  reads; otherwise a staged snapshot can differ from repeated loads.'
- 'Pipeline: input producers must finish before copying, all CTA threads must reach
  the copy-completion barrier before consumption, and all readers must finish before
  buffer reuse; these rules prevent incomplete or overwritten reads.'
- 'Hardware: CUDA CTA shared memory and barriers are required; live shared storage
  including the gate tile must fit the kernel''s allowed per-CTA capacity.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Stage each CTA's raw gate tile once in shared memory, then reuse it in every
prefix evaluation. Apply this only to gate-input transfers in
`solution/native.cuh` and `solution/prepare.cuh`.

In `PrepareShared`, rename `gate_reserved` to `gate`. In `load_prepare`, add
`s.gate[i] = args.g[src]` after the existing query/key stores. The cooperative
loop already maps linear tile indices to the input's token/head/dimension
coordinates. In `prepare`'s inner prefix loop, replace the global gate address
and load with `s.gate[p*kDim+tid]`. Remove only the BF16 overload of
`ld_global_scalar`; retain its FP32 overload and all existing volatile accesses.
The BF16 overload prevents the deoptimized compiler from reusing gate loads;
the staged implementation no longer needs that control.

Each CTA owns one `(tile, head)`. Copy threads own indices
`i = threadIdx.x + n*kPrepareThreads`. Prefix thread `tid < kDim` reads column
`tid` across rows `p`. Global rows have stride `kHeads*kDim` elements; staged
rows have stride `kDim`. Gate staging changes no element bits, BF16-to-FP32
conversion, prefix summation order, activation, or rounding.

Retain both barriers after `load_prepare` and every later barrier. The first
orders cooperative writes before any consumer. This gate buffer is not
overwritten during the CTA; its lifetime ends after the prefix reads. Input
production and preparation remain ordered on the caller stream. Keep launch
geometry, shared allocation, ABI, compile flags, and workspace unchanged.
The reserved array already supplies capacity, so restoring staging adds no
allocation or offset changes.

## Example configuration

Preserve BTHD shape `(1, 4096, 96, 128)`, BF16 raw gates, and FP32 gate
arithmetic. Chunks contain 16 tokens. The preparation grid is `(256, 96)`
with 256 threads per CTA. Each thread copies eight elements; prefix threads
0 through 127 consume 16 rows. The gate tile contains 2048 elements, occupies
4096 bytes, and starts at a 128-byte-aligned member. `PrepareShared` remains
41856 bytes; existing static shared scratch remains unchanged.

All chunks are complete, so the existing copy loop and `tid < kDim` guard
cover valid coordinates without padding or new bounds checks. Preserve the
rolled nested prefix loops, repeated FP32 gain/bias loads, scalar fragment
transfers, normalization grouping, MMA, BF16 rounding, recurrence launches,
and state handoffs. This change does not restore prefix-sum reuse.

Use the existing evaluator with the supplied problem for correctness and timing.
Inspect generated preparation code to confirm one cooperative global-to-shared
gate copy and shared gate reads inside the prefix loop.

# Precondition

- Data types: copying must preserve the gate representation exactly. The transfer
  has no additional arithmetic dtype requirement: conversions and arithmetic
  remain at the consumer, so staging must not add rounding.
- Layout: repeated gate consumers must lie within one CTA. Known strides and
  ownership must let cooperative writers cover every consumed element and readers
  select the corresponding shared index. Scalar copies need only element alignment;
  neither vector alignment nor contiguous global rows is required.
- Storage: gates must already be available in global memory and stable throughout
  the CTA's reads. Concurrent changes could make a shared snapshot differ from
  the preceding implementation's repeated loads.
- Pipeline: input producers must complete before copies begin. Every CTA thread
  must reach the barrier that makes completed copies visible to consumers.
  Readers must complete before any buffer overwrite or reuse. These dependencies
  prevent partial publication and stale or overwritten reads; no stage count is
  required.
- Hardware: CUDA must provide CTA shared memory and CTA barriers. Total live
  shared storage, including `gate_tile_elements*sizeof(gate_element)`, must fit
  the permitted per-CTA capacity; otherwise the staging allocation cannot launch.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These fragments identify three replacement sites and the removable helper.
All surrounding loops, stores, arithmetic, and barriers remain unchanged.

## Before

```cuda
// solution/native.cuh: PrepareShared member.
alignas(128) BF16 gate_reserved[kTileElems], kr[kTileElems];

// solution/native.cuh: BF16 overload only.
__device__ __forceinline__ BF16 ld_global_scalar(const BF16* p) {
    uint16_t bits;
    asm volatile("ld.volatile.global.b16 %0, [%1];"
      : "=h"(bits) : "l"(p) : "memory");
    return __ushort_as_bfloat16(bits);
}

// solution/prepare.cuh: load_prepare copy loop.
for (int i = tid; i < kTileElems; i += kPrepareThreads) {
    const int row = i / kDim, col = i % kDim;
    const int src = ((token + row) * kHeads + head) * kDim + col;
    s.q[i] = args.q[src];
    s.k[i] = args.k[src];
}

// solution/prepare.cuh: prepare inner prefix loop, after gain/bias loads.
const int src = ((tile*kChunk+p)*kHeads+head)*kDim+tid;
float g = gain * (bf_float(ld_global_scalar(args.g+src))+bias);
sum += kGateScale * sigmoid(g);
```

## After

```cuda
// solution/native.cuh: restore the member; delete the BF16 load overload.
alignas(128) BF16 gate[kTileElems], kr[kTileElems];

// solution/prepare.cuh: stage raw gates alongside normalization inputs.
for (int i = tid; i < kTileElems; i += kPrepareThreads) {
    const int row = i / kDim, col = i % kDim;
    const int src = ((token + row) * kHeads + head) * kDim + col;
    s.q[i] = args.q[src];
    s.k[i] = args.k[src];
    s.gate[i] = args.g[src];
}

// solution/prepare.cuh: prepare inner prefix loop, after gain/bias loads.
float g = gain * (bf_float(s.gate[p*kDim+tid])+bias);
sum += kGateScale * sigmoid(g);
```

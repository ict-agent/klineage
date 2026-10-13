---
skill_id: kda.raw-input-shared-staging
intent: Reuse raw normalization inputs through CTA shared-memory staging.
preconditions:
- 'Data types: shared copies must preserve input representation and arithmetic/rounding
  order; otherwise normalization changes. No particular numeric dtype is intrinsic
  to copying bits.'
- 'Layout: repeated reads within a CTA need known global/shared index mappings and
  consumer ownership to receive the correct reused elements. No additional contiguity
  or alignment is needed beyond valid typed accesses.'
- 'Storage: raw global inputs must remain stable throughout their consumers, and both
  tiles must be accessible within one CTA; otherwise a shared snapshot cannot serve
  the same values.'
- 'Pipeline: producers finish before copying; participating threads reach a publication
  barrier before shared reads; all raw readers finish before overwrite; normalized
  writes become visible before later reads or reuse. These prevent incomplete or overwritten
  reads; no fixed stage count is required.'
- 'Hardware: CTA shared memory and block barriers, with capacity for elements(q_tile)*sizeof(q_element)
  + elements(k_tile)*sizeof(k_element) plus other live shared storage; otherwise the
  simultaneous tiles do not fit. Staging requires no matrix instruction.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Stage raw query/key normalization inputs once per CTA in shared memory, then
reuse them for the sum of squares and normalization passes. This replaces two
global reads per element with one global read and two shared reads. Reuse the
existing `PrepareShared::q` and `k` arrays before they hold normalized values.

In `solution/prepare.cuh`, insert the `load_prepare` helper below `PrepareArgs`.
Call it in `prepare` immediately before its first two existing `__syncthreads()`
calls. Replace the raw query/key loads in both normalization loops with the
shared loads shown below. Remove each loop's now-unused global `src` variable.
Leave sums, XOR exchanges, `norm_each` calls, normalized stores, and all subsequent
computation unchanged. No host, launch, allocation, or compile-flag change is needed.

The cooperative loader assigns linear tile positions `i = tid + n*kPrepareThreads`.
It gathers BTHD global inputs into row-major `[token_in_chunk, feature]` shared
arrays with element stride `kDim`. The normalization owner remains
`row = tid/(kDim/kNormElems)`, `col = tid%(kDim/kNormElems)*kNormElems` and owns
`col ... col+kNormElems-1`. The two mappings differ; the first existing CTA barrier
makes all cooperative writes visible to normalization owners.

Global producers finish through caller-stream ordering before `prepare` runs.
Each normalizer finishes reading its own raw elements before overwriting those
same elements with normalized values. No other normalizer reads those positions;
cross-thread reduction exchanges only partial sums in `norm_q` and `norm_k`.
Retain the CTA barrier after normalized stores before later consumers read them.
The second initial barrier and all remaining synchronization are preserved.

Copy the operand bits unchanged. Retain FP32 partial-sum order, XOR reduction
partners, epsilon, reciprocal-square-root evaluation, and each BF16 rounding point.
Use ordinary shared loads in the sum loop and existing volatile `ld_scalar` loads
in the normalization loop; this restores staging without adding register reuse.
Keep `ld_global_scalar` for its other callers. No cache-policy or descriptor change
is part of this technique.

## Example configuration

Replay uses BATCH=1, TOKENS=4096, HEADS=96, HEAD_DIM=128, `kChunk=16`,
`kTiles=256`, `kPrepareThreads=256`, and `kNormElems=8`. Each CTA handles one
16-by-128 tile for one head. Sixteen threads normalize each row, eight elements
per thread, with XOR deltas 8, 4, 2, 1. Every tile and ownership range is complete,
so retain the existing absence of tail predicates. A generalized partial tile must
guard global loads and output writes while keeping barrier participation uniform.

Inputs q/k are contiguous BF16 BTHD arrays. Their token stride is
`kHeads*kDim*sizeof(BF16)` (24576 bytes); each shared row is
`kDim*sizeof(BF16)` (256 bytes). Each q/k tile occupies 4096 bytes, already reserved
in `PrepareShared`. Its 41856-byte dynamic allocation and separate static scratch
remain unchanged. Preserve `prepare<<<dim3(kTiles,kHeads),kPrepareThreads,
sizeof(PrepareShared),stream>>>` and `__launch_bounds__(kPrepareThreads,8)`.

Retain normalization epsilon 1e-6, scale 0.08838834764831843f, the log2 gate scale,
all build flags, and BF16 intermediates with FP32 accumulation. Keep scalar
fragment loads/stores, shared lane exchanges, tensor-core matrix products,
triangular inversion, gate-prefix recomputation, global preparation workspace,
and the 256 caller-stream recurrence launches. Recurrence uses 256 threads per
head and 124672 dynamic shared bytes; it is outside this change.

Validate with the existing Kernel evaluator and unchanged problem. Inspect the
compiled preparation kernel to confirm raw global-to-shared stores precede
normalization and both consumers load shared memory. Keep measurements outside
this card.

# Precondition

- Data types: shared copies must preserve input representation, and replacing
  loads must preserve arithmetic and rounding order. Otherwise staging changes
  normalization values. No particular numeric dtype is intrinsic to copying bits.
- Layout: repeated reads within a CTA must have known global/shared index mappings
  and consumer ownership. Wrong mappings deliver another element to a consumer;
  without repeated reads there is no global-load reuse. No additional contiguity
  or alignment is required beyond valid typed accesses.
- Storage: raw inputs must be available in global memory and stable throughout
  their consumers. Reusing a staged snapshot would otherwise replace a newer
  value. Both input tiles must be accessible to their consumers within one CTA.
- Pipeline: input producers must finish before copying, and all participating
  threads must reach the publication barrier before consuming shared values.
  Every raw reader must finish before its location is overwritten, and normalized
  writes must be visible before later reads or buffer reuse. These dependencies
  prevent stale, uninitialized, or overwritten reads; no fixed stage count is needed.
- Hardware: CTA shared memory and block barriers must be available. Capacity must
  cover `elements(q_tile)*sizeof(q_element) + elements(k_tile)*sizeof(k_element)`
  plus other simultaneously live shared storage; otherwise both raw tiles cannot
  be staged together. No matrix instruction is required by the staging itself.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

```cuda
// prepare: initialization, before normalization.
int tid = threadIdx.x;
int tile = blockIdx.x;
int head = blockIdx.y;

__syncthreads();
__syncthreads();

// Sum-of-squares loop: retain qs += q*q and ks += k*k afterward.
const int src = ((tile*kChunk+row)*kHeads+head)*kDim+col+j;
float q = bf_float(args.q[src]);
float k = bf_float(args.k[src]);

// Normalization loop: retain norm_each calls and BF16 shared stores afterward.
const int src = ((tile*kChunk+row)*kHeads+head)*kDim+col+j;
float q = bf_float(ld_global_scalar(args.q+src));
float k = bf_float(ld_global_scalar(args.k+src));
```

## After

```cuda
// Add below PrepareArgs; each input element is copied once per CTA.
__device__ __forceinline__ void load_prepare(
    PrepareShared& s, const PrepareArgs& args, int tile, int head) {
    const int tid = threadIdx.x;
    const int token = tile * kChunk;

    for (int i = tid; i < kTileElems; i += kPrepareThreads) {
        const int row = i / kDim, col = i % kDim;
        const int src = ((token + row) * kHeads + head) * kDim + col;
        s.q[i] = args.q[src];
        s.k[i] = args.k[src];
    }
}

// prepare: publish raw tiles before normalization owners read them.
int tid = threadIdx.x;
int tile = blockIdx.x;
int head = blockIdx.y;

load_prepare(s,args,tile,head);
__syncthreads();
__syncthreads();

// Sum-of-squares loop: arithmetic remains unchanged.
float q = bf_float(s.q[row*kDim+col+j]);
float k = bf_float(s.k[row*kDim+col+j]);

// Normalization loop: preserve the separate, volatile consumer loads.
float q = bf_float(ld_scalar(s.q+row*kDim+col+j));
float k = bf_float(ld_scalar(s.k+row*kDim+col+j));
```

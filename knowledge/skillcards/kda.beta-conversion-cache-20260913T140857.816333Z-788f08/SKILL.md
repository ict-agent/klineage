---
skill_id: kda.beta-conversion-cache
intent: Preconvert reused logits once into a shared global cache.
preconditions:
- 'Data types: consumers must need the same deterministic element conversion and rounding;
  a cached result cannot represent different conversions.'
- 'Layout: consumer indices must map to known source elements, so cache lookups select
  the same values; no additional alignment or collective thread mapping is required.'
- 'Storage: consumers in separate kernels read the same device-global source elements;
  their converted values must be shareable through device-global storage.'
- 'Pipeline: the source must remain unchanged for all consumers; conversion must finish
  before readers, and every reader must finish before cache overwrite or release.'
- 'Hardware: ordinary CUDA kernels and global memory with room for N*sizeof(ConvertedType)
  extra bytes alongside existing live allocations; no specialized transfer instruction
  is needed.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Move repeated beta-logit conversion out of `load_prepare` in
`solution/prepare.cuh` and `load_input` in `solution/recurrence.cuh`.
A new `prepare_beta` kernel converts each source element once into a token-major
BF16 global array of `N=kTokens*kHeads` elements. Both consumers load that
array into their existing shared beta buffers. This shares conversion work and reduces consumer global-load width.

In `solution/native.cu`, add `BF16* beta = nullptr` as the first `Buffers` field.
Allocate it in `buffers`, before workspace allocation, using the existing
per-device, per-caller-stream cache. Add `prepare_beta` inside the anonymous
namespace. Launch it immediately after successful `buffers(b,stream)`, before
preparation and recurrence, and point both argument structures at `b->beta`.
Populate the cache on every call; only its allocation persists.

In `PrepareArgs`, restore `const BF16 *q, *k, *beta, *g` and
`const float* bias`. In `RecurArgs`, restore
`const BF16 *v, *beta, *kd, *qd, *kr, *inv, *mqk` and `const float* gt`.
Other fields stay unchanged. Remove only `BF16(...)` around valid beta loads
in both consumers. Retain the existing source-index guard and zero fill.

The producer gives thread `index` sole ownership of cache element `index`.
Each consumer retains its current `i = threadIdx.x` strided loop and computes
`src = head*kTokens + tile*kChunk + i`, then
`gathered = (src%kTokens)*kHeads + src/kTokens`.
The guard tests `src < kTokens*kHeads` before loading. Preserve this logical
head rollover even when the staging extent exceeds one chunk. Physical source
and cache element order match; their byte strides differ by element size.

Use the existing caller stream for the new launch. Its completion orders global
cache visibility before both consumers. Keep their CTA barriers after shared
loads, all subsequent readers, and all barriers before shared storage reuse.
The next call's conversion follows the preceding call's readers on that stream;
separate stream entries own separate allocations. No additional device barrier
or CPU synchronization is needed.

Preserve the exact FP32-to-BF16 round-to-nearest-even conversion before sigmoid;
do not cache sigmoid results. Keep all later BF16 rounding, FP32 arithmetic,
reduction grouping, compile flags, ABI checks, and output writes unchanged.
No compiler-control change is required: this restores an explicit allocation
and launch boundary. Rebuild the complete bundle and check correctness and
latency with `klineage.harness.evaluate` using the unchanged problem.

## Example configuration

The supplied workload has batch 1, 4096 tokens, 96 heads, and head dimension 128.
Beta is contiguous FP32 `[1,4096,96]`; the cache is contiguous BF16 in the same
order. Its row strides are respectively `96*sizeof(float)` and
`96*sizeof(BF16)`, and its allocation is 786432 bytes.

Use `kPrepareThreads=256`: the conversion launch has 1536 blocks and a guarded
last block. Retain 16-token chunks, `kBetaElems=32`, and both consumer loops.
Preparation retains grid `(256,96)` with 256 threads and 41856 dynamic shared
bytes; recurrence retains 256 sequential launches, grid `(1,96)`, 256 threads,
and 124672 dynamic shared bytes. These sizes are replay settings, not cache
requirements.

Keep the existing row-major shared layouts, BF16 tensor-core products and
fragment adapters, shared correction/readout staging, scalar volatile accesses,
normalization grouping, gate-prefix computation, and global BF16 state handoff.
Keep the SM90 target and all existing build flags. The tensor-core requirements
belong to retained computation, not this cache.

# Precondition

- Data types: every consumer must use the same deterministic conversion with
  the same rounding. Hoisting that conversion preserves its bits; differing
  conversions or context-dependent rounding could not share one cached value.
- Layout: the source element selected by each consumer must be known. The cache
  lookup must select that element's conversion, or it changes values. There is
  no additional alignment or collective thread-mapping requirement: ordinary
  scalar indexing suffices, and other strides only need adapted indexing.
- Storage: separate consumer kernels already read common device-global source
  elements. Global cache storage must be visible to all those consumers;
  kernel-local storage cannot share values across their launch boundaries.
- Pipeline: the source must remain unchanged across the consumers served by one
  conversion. The producer must complete before any cache read, and all readers
  must complete before overwrite or release. Otherwise readers can see stale,
  incomplete, or replaced values. Existing shared-load visibility and safe
  shared-buffer reuse remain ordered; this technique needs no extra stage count.
- Hardware: ordinary CUDA execution and enough global memory for
  `N*sizeof(ConvertedType)` extra bytes while the existing allocations remain
  live. Allocation would fail without that capacity. No specialized copy,
  collective, or matrix instruction is required by conversion caching.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The snippets show the changed declarations and load expressions plus insertion
points in existing functions. Keep each file's other fields and statements.
`BF16`, dimensions, `status`, and `stream` already exist in the bundle.

## Before

```cuda
// solution/native.cu: Buffers; launch assignments.
struct Buffers { BF16* carry = nullptr; void* workspace = nullptr; };
// buffers() allocates workspace and carry only.
// launch() proceeds directly from buffers() to argument setup.
recur.beta = static_cast<const float*>(inputs[4]);
prep.beta = static_cast<const float*>(inputs[4]);

// solution/prepare.cuh: leading PrepareArgs fields; load_prepare loop.
const BF16 *q, *k, *g;
const float *beta, *bias;
// Inside the existing beta loop:
const int src = head * kTokens + token + i;
const int gathered = (src % kTokens) * kHeads + src / kTokens;
s.beta[i] = src < kTokens * kHeads ? BF16(args.beta[gathered]) : BF16(0);

// solution/recurrence.cuh: leading RecurArgs fields; load_input loop.
const BF16 *v, *kd, *qd, *kr, *inv, *mqk;
const float *beta, *gt;
// Inside the existing beta loop:
const int src = head * kTokens + token + i;
const int gathered = (src % kTokens) * kHeads + src / kTokens;
in.beta[i] = src < kTokens * kHeads ? BF16(args.beta[gathered]) : BF16(0);
```

## After

```cuda
// solution/native.cu: replace Buffers.
struct Buffers { BF16* beta = nullptr; BF16* carry = nullptr; void* workspace = nullptr; };

// buffers(): insert after selecting b, before allocating b.workspace.
if (!b.beta) {
    status = cudaMalloc(reinterpret_cast<void**>(&b.beta),
        size_t(kTokens)*kHeads*sizeof(BF16));
    if (status != cudaSuccess) return status;
}

// Add inside the anonymous namespace, after buffers().
__global__ void prepare_beta(const float* source, BF16* destination) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if (index >= kTokens*kHeads) return;
    // Reuse one rounded value across all consumers.
    destination[index] = BF16(source[index]);
}

// launch(): insert after buffers(b,stream) succeeds.
prepare_beta<<<(kTokens*kHeads+kPrepareThreads-1)/kPrepareThreads,
    kPrepareThreads,0,stream>>>(static_cast<const float*>(inputs[4]),b->beta);

// Replace the existing argument assignments in launch().
recur.beta = b->beta;
prep.beta = b->beta;

// solution/prepare.cuh: leading PrepareArgs fields; load_prepare loop.
const BF16 *q, *k, *beta, *g;
const float* bias;
// Inside the existing beta loop:
const int src = head * kTokens + token + i;
const int gathered = (src % kTokens) * kHeads + src / kTokens;
s.beta[i] = src < kTokens * kHeads ? args.beta[gathered] : BF16(0);

// solution/recurrence.cuh: leading RecurArgs fields; load_input loop.
const BF16 *v, *beta, *kd, *qd, *kr, *inv, *mqk;
const float* gt;
// Inside the existing beta loop:
const int src = head * kTokens + token + i;
const int gathered = (src % kTokens) * kHeads + src / kTokens;
in.beta[i] = src < kTokens * kHeads ? args.beta[gathered] : BF16(0);
```

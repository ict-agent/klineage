---
skill_id: fmha.query-tile-count-precomputation
intent: Precompute query-tile counts once per call for reuse across attention CTAs.
preconditions:
- 'Data types: count arithmetic must exactly represent offset differences, ceiling-division
  numerators, head-expanded counts, and prefix sums; overflow or rounding would select
  the wrong tile. No attention-value dtype requirement is added.'
- 'Layout: known sequence endpoints and a stable sequence-to-tile mapping must yield
  the same tile counts for multiple attention CTAs; otherwise one precomputed count
  cannot replace their local calculations. No tensor alignment requirement is added.'
- 'Storage: sequence offsets already reside in device-global memory and remain unchanged
  throughout the call; the producer must read the same endpoints as the original consumers.'
- 'Pipeline: offset production must finish before the metadata producer, the producer
  must finish before attention reads its output, and all readers must finish before
  workspace reuse; otherwise readers can observe stale, incomplete, or overwritten
  counts.'
- 'Hardware: CUDA device-kernel launches and device-global memory with capacity for
  at least one exact count per sequence are required to publish and retain the reusable
  counts; no tensor-core or shared-memory feature is required by this transformation.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Precompute each sequence's query-tile count once per call, then reuse it in
all attention CTAs. In `solution/scheduler.cuh`, move
`ceil((q_offsets[b + 1] - q_offsets[b]) / kM)` from `tile` into a
`prepare` kernel. Preserve the sequence scan, head expansion, prefix sums,
query-tile ownership, and out-of-range sentinel.

Add device-global metadata storage to `Workspace` in `solution/kernel.cu`
and its pointer to `Params` in `solution/ops.cuh`. Populate it on every call:
sequence lengths can change even when tensor shapes do not. Launch `prepare`
on the caller stream immediately before `attention`. Kernel completion on
that stream publishes all producer writes; no producer CTA barrier is needed
because each producer thread owns distinct entries.

In `solution/attention.cuh`, pass `p.metadata` to `tile`. In both `consumer`
and `epilogue`, obtain the sequence from the identity batch plane instead of
using `work.w` directly. These lookups retain the original tile mapping.
Metadata contains no Q/K/V values and changes no floating-point operation.

Keep input offsets ready before `prepare`, and serialize reuse of a workspace
until its previous attention readers finish. The existing caller-stream order
provides these edges for serialized calls. Concurrent streams require separate
workspaces or explicit dependencies, not a block barrier. No shared-memory
layout, math-group participation, or attention buffer-lifetime change is needed.

After reconstruction, use `klineage.harness.evaluate` on the supplied problem
for correctness and timing. Inspect the compiled producer and metadata reads
to confirm restoration. Keep measurements outside this card.

## Example configuration

Preserve CUDA platform `nvidia-sm90a-cuda13`, the destination-passing pybind11
callable `kernel.cu::kernel`, caller device/stream, config, and compile flags
`-O3 -std=c++17 --use_fast_math --resource-usage -lineinfo -DNDEBUG`.

This replay uses eight sequences, 16384 packed tokens, 64 heads, head dimension
128, and signed int32 offsets. Q/K/V/output remain contiguous FP16 NHD with
row stride `kHeads * kDim` elements (8192 here); none are relocated.
Preserve FP32 accumulations, reverse KV traversal, online softmax and its
reduction order, probability/output round-to-nearest FP16 conversion, tail
zero-filling and masking, and scalar output stores.

The query tile is `kM=128`, KV tile `kN=176`, and attention has 256 threads
in two 128-thread math groups. Keep WGMMA QK `m64n176k16` and PV
`m64n128k16`, their descriptor panels, fences, commits, and waits. Per-thread
arrays retain 88 score registers, 64 output registers, and 44 packed probability
words. Preserve the one-buffer synchronous Q/K/V loads, Q reload per KV tile,
shared-memory peer reductions, and CTA barriers. Dynamic shared storage stays
`sizeof(Shared)` (123904 bytes); its outer alignment stays 1024 bytes.

Keep attention grid `kMaxQTiles * kHeads` (8640 CTAs), block `kThreads`, and
its existing invalid-tile return. Add `prepare<<<1, kWarpSize, 0, stream>>>`;
only threads with `threadIdx.x < kBatch` write. This launch covers this batch;
other batch sizes require adapting producer coverage.

For exact replay, allocate three `kBatch`-entry int32 planes (96 bytes total):
`kSplitOffset=0`, `kBlocksOffset=kBatch`, `kBatchOffset=2*kBatch`, and
`kMetadata=3*kBatch`. The split plane contains ones and is unused by this
consumer; the batch plane is identity, with no sequence sorting. Only the count
plane eliminates repeated arithmetic. Preserve the original `prepare` signature
with its unused K-offset argument. These incidental fields do not add another
optimization or a prerequisite.

# Precondition

- Data types: count arithmetic must exactly represent offset differences,
  ceiling-division numerators, head-expanded counts, and prefix sums. Overflow
  or rounding would change the block-to-sequence decision. The transformation
  does not operate on attention values, so their dtype adds no requirement.
- Layout: sequence endpoints and the sequence-to-tile mapping must be known
  and stable, with multiple attention CTAs reusing each derived count. A
  consumer-dependent count cannot be replaced by one common value. Tensor
  contiguity and alignment add no requirement to this metadata transformation.
- Storage: the preceding kernel reads sequence offsets from device-global
  memory. Those endpoints must remain unchanged throughout the call, so the
  separate producer derives the same values as the original consumers.
- Pipeline: offsets must be ready before the producer; producer completion
  must precede every attention read; all readers must finish before the same
  workspace is overwritten. These edges prevent stale, incomplete, and
  overwritten metadata. They impose no attention stage count or CTA barrier.
- Hardware: CUDA must support device-kernel launches and device-global storage
  for at least `sequence_count * sizeof(count_type)` bytes. A separate producer
  needs these facilities to publish and retain counts. Tensor cores and shared
  memory belong to the retained attention computation, not this technique.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The blocks show only the changed regions. Keep `tile`'s existing range scan,
`end`, `within`, head/query mapping, return values, and all unshown code.
The global constants below go inside `namespace native` before `kMaxQTiles`.
The host fragments go in the indicated existing scopes.

## Before

```cuda
// solution/scheduler.cuh: tile signature and loop calculation.
__device__ __forceinline__ int4 tile(int index, const int* q_offsets) {
    int start = 0;
    for (int batch = 0; batch < kBatch; ++batch) {
        const int length = q_offsets[batch + 1] - q_offsets[batch];
        const int blocks = (length + kM - 1) / kM;
        // Existing range scan and returns follow unchanged.
    }
}

// solution/ops.cuh, Params: no metadata pointer.
// solution/kernel.cu, Workspace: only lse and device fields.
// solution/kernel.cu, kernel(): attention is the sole device launch.
check(cudaLaunchKernelEx(&config, attention, p));
check(cudaGetLastError());

// solution/attention.cuh, consumer().
const int4 work = tile(blockIdx.x, p.q_offsets);
if (work.w >= kBatch) return;
const int batch = work.w;

// solution/attention.cuh, epilogue().
const int batch = work.w;
```

## After

```cuda
// solution/scheduler.cuh: add metadata planes and their producer.
constexpr int kSplitOffset = 0;
constexpr int kBlocksOffset = kBatch;
constexpr int kBatchOffset = 2 * kBatch;
constexpr int kMetadata = 3 * kBatch;

__global__ void prepare(const int* qo, const int* ko, int* metadata) {
    const int lane = threadIdx.x;
    if (lane >= kBatch) return;

    const int q = qo[lane];
    const int qnext = qo[lane + 1];

    // Publish independent entries; sequence order stays unchanged.
    metadata[kSplitOffset + lane] = 1;
    metadata[kBlocksOffset + lane] = (qnext - q + kM - 1) / kM;
    metadata[kBatchOffset + lane] = lane;
}

// Replace tile's signature and count calculation; retain its range scan.
__device__ __forceinline__ int4 tile(int index, const int* metadata) {
    int start = 0;
    for (int batch = 0; batch < kBatch; ++batch) {
        const int blocks = metadata[kBlocksOffset + batch];
        // Existing range scan and returns follow unchanged.
    }
}

// solution/ops.cuh: add after Params::lse.
int* metadata;

// solution/kernel.cu: add after Workspace::lse.
int* metadata = nullptr;

// Workspace::ensure(): add after the existing lse allocation.
check(cudaMalloc(&metadata, sizeof(int) * kMetadata));

// kernel(): add after p.lse assignment.
p.metadata = scratch.metadata;

// After assigning both offset pointers, before attention launch configuration.
prepare<<<1, kWarpSize, 0, stream>>>(p.q_offsets, p.k_offsets, p.metadata);
check(cudaGetLastError());

// Existing config/launch remains; stream order publishes producer writes.
check(cudaLaunchKernelEx(&config, attention, p));
check(cudaGetLastError());

// solution/attention.cuh, consumer().
const int4 work = tile(blockIdx.x, p.metadata);
if (work.w >= kBatch) return;
const int batch = p.metadata[kBatchOffset + work.w];

// solution/attention.cuh, epilogue().
const int batch = p.metadata[kBatchOffset + work.w];
```

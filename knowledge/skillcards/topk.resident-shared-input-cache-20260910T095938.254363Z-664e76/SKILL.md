---
skill_id: topk.resident-shared-input-cache
intent: Reuse Top-K input keys across radix passes through a resident shared-memory
  cache.
preconditions:
- 'Data types: Copies must preserve key representation; staging must not change the
  ordering transform or emitted value bits.'
- 'Layout: Per-CTA source ranges, valid counts, and original indices must be known,
  with element-aligned typed accesses, so cooperative copies and consumers address
  the same keys.'
- 'Storage: Repeatedly read keys reside in global memory and remain unchanged through
  their last use; all consumers of each cached range belong to its owning CTA. This
  lets CTA-local shared memory serve stable repeated reads.'
- 'Pipeline: Input production completes before consumption, every CTA thread can reach
  publication barriers uniformly, and no required scratch reuse interrupts the input
  reuse interval; otherwise readers can observe incomplete or overwritten data.'
- 'Hardware: CUDA shared memory and CTA barriers, with capacity for live shared state
  plus cached keys and alignment padding; the allocation must permit the existing
  concurrent CTA group to reside.'
scope:
  cases:
  - topk
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Cache each CTA's input ranges in shared memory once, then reuse them for radix
histograms and final Top-K emission. This removes repeated global reads without
changing key ordering, integer counts, output indices, or selection semantics.

## Replay

Edit `solution/topk.cuh`, `solution/native.cuh`, and `solution/kernel.cu`.
Keep `config.toml`, the tensor ABI, caller stream, and existing computation.
The Before/After fragments below identify the replacement regions; omitted code
stays in place.

1. In `topk.cuh`, add `kDynamicBytes = kChunks * kChunkBytes + kAlign` after
   `kChunks`. Append `float edges[2 * kAlignItems];` after `Shared::scan` and
   change `kStaticBytes` to `8576`. Keep its size assertion.
2. Add `native::copy` before `scan_step` in `native.cuh`, as shown below. Each
   thread copies indices `threadIdx.x + n * blockDim.x`; its CTA barrier publishes
   the complete chunk before histogram readers run.
3. Immediately after `__shared__ Shared sm;` in `select`, allocate and align the
   dynamic key array as shown. CTA rank `r` owns chunk `j` from
   `input + offsets[j]`; store it at `keys + j * kChunkItems`. Retain the existing
   offset/count calculation and the head/tail rank ownership.
4. Replace first-pass global reads with synchronous copies followed by shared
   reads. Cache the head and tail in `sm.edges`. Later passes and `emit` read only
   these cached ranges. Keep `offsets[j]` as the emission index base; a shared
   offset is not an original token index.
5. In `prepare` in `kernel.cu`, after the nonportable-cluster attribute succeeds,
   add the dynamic-memory attribute and its error check shown below. Set
   `config.dynamicSmemBytes = topk::kDynamicBytes`. Keep the grid, block, cluster,
   carveout, launch bounds, device guard, and stream unchanged.

The later-pass fragment combines resident chunks into one contiguous scan.
For this configuration, `counts[0] == kChunkItems` and only the last chunk can be
short, so `counts[0] + counts[1]` excludes every unused cache slot. Global chunks
are separated by other CTAs' chunks, which is why Before uses a chunk loop.
If adapting the layout to permit holes, iterate cached chunks with their individual
counts instead of scanning across uninitialized slots.

## Ordering and bounds

Keep input-producer ordering on the caller stream. Every CTA thread calls each
cooperative copy, including threads with no valid load. Preserve the initial state-publication barrier, histogram-completion
barriers, cluster arrival/wait pairs, scan barriers, and output-counter barriers.
The cache is populated once and remains unchanged until the final emission;
histogram resets must not overwrite it. It is not rotated or reused during the
kernel. A later extension that reuses this storage must wait for all readers.

The edge counts are smaller than a warp in this replay. Thread `i` writes edge
`i` and reads that same element in the first histogram, so that first read needs
no cross-thread handoff. The existing barrier after first-pass edge histograms
publishes the edges before later uses. Preserve this ownership or add publication
synchronization if it changes.

Copy only valid chunk elements. Retain the histogram and emission bounds checks,
head/tail counts, and `out < kSelected` guard. Copy values without arithmetic or
conversion; preserve `ordered`, radix masks, tie counters, and widening to
`int64_t`. Output order and the choice among tied indices remain unspecified.

## Example configuration

The supplied workload is contiguous FP32 input `[1, 131072]`, FP32 values
`[1, 2048]`, and int64 indices `[1, 2048]`. Its row stride is the row extent in
elements. Preserve its complete problem and numerical requirements.

Keep 512 threads per CTA, 16 warps, and one 16-CTA cluster:
`grid=(1,16,1)`, `block=(512,1,1)`, `cluster=(1,16,1)`.
Retain `__launch_bounds__(kThreads, 1)`, the nonportable-cluster opt-in, and the
25% preferred shared carveout. SM90 cluster addressing, remote reductions,
remote state loads, warp scans, and overlapping local scans with cluster waits
belong to the retained selection implementation.

Keep 11 radix bits, 2048 histogram buckets, three passes with starting bits
21, 10, and 0, and eight striped register items per thread. Keep both histogram
and emission register batching. No asynchronous copies or rotating stages are
introduced.

Each CTA retains two 16 KiB chunks, each containing 4096 FP32 keys. Preserve the
128-byte alignment peel, 32-element alignment unit, and
`offsets[j] = head + (rank + j * kBlocks) * kChunkItems`.
The cache reserves 32768 payload bytes plus 128 alignment bytes, giving
`kDynamicBytes = 32896`. Its edge array holds 64 FP32 slots. Static shared memory
grows from 8320 to 8576 bytes; total requested shared memory becomes 41472 bytes
per CTA. These dimensions and launch choices specify this replay, rather than
requirements of synchronous caching.

# Precondition

- Data types: copies must preserve the key representation used by ordering and
  emission. A conversion could change a radix bucket or selected value. Caching
  itself imposes no particular floating-point type or arithmetic order; retain
  the operator's existing transform and output bits.
- Layout: source ranges, valid extents, and original index mappings must be known
  for each CTA. Typed accesses need element alignment. Without these mappings,
  cooperative stores or later lookups can select the wrong key or index. No
  additional vector alignment is required by scalar copying.
- Storage: the reused keys must be available in global memory and unchanged
  through their last use, including freedom from overlapping writes. Otherwise
  a cached snapshot differs from later global reads. Each cached range's
  consumers must belong to one CTA because ordinary shared memory is CTA-local.
- Pipeline: the input producer must complete before consumption. All CTA
  threads must be able to reach
  each publication barrier uniformly, or consumers can race incomplete copies
  or deadlock. The input reuse interval must finish before any required scratch
  reuse; otherwise a later reader can observe overwritten keys. No fixed stage
  count is required.
- Hardware: CUDA shared memory and CTA barriers must be available. The capacity
  requirement is `S_live + N_cache * sizeof(T) + P_align`, where `S_live` is other
  live shared state, `N_cache` includes every cached range, and `P_align` covers
  chosen alignment padding. The resulting allocation must fit per CTA and permit
  the existing concurrent CTA group to reside; otherwise the launch cannot
  support the cache. Cluster-specific instructions are retained computation,
  not a requirement of the caching technique.

# Scope

- Cases: topk
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

```cuda
// topk.cuh: Shared ends with Scan scan; kStaticBytes is 8320.
// select begins with __shared__ Shared sm; no input cache follows it.

// First-pass region after initial state publication.
#pragma unroll 1
for (int chunk = 0; chunk < kChunks; ++chunk) {
  histogram<Pass::first>(sm, input + offsets[chunk], counts[chunk],
      0, 0, start_bit(0), kBuckets - 1);
}
histogram<Pass::first>(sm, input, head_count, 0, 0, start_bit(0), kBuckets - 1);
histogram<Pass::first>(sm, input + kLength - tail, tail_count,
    0, 0, start_bit(0), kBuckets - 1);
__syncthreads();

// Later-pass region inside the existing pass loop.
if (pass != 0) {
  #pragma unroll 1
  for (int chunk = 0; chunk < kChunks; ++chunk)
    histogram<Pass::later>(sm, input + offsets[chunk], counts[chunk],
        splitter, prior_bit, bit, mask);
  histogram<Pass::later>(sm, input, head_count, splitter, prior_bit, bit, mask);
  histogram<Pass::later>(sm, input + kLength - tail, tail_count,
      splitter, prior_bit, bit, mask);
}

// Final emission region after output-counter synchronization.
#pragma unroll 1
for (int chunk = 0; chunk < kChunks; ++chunk)
  emit(sm, input + offsets[chunk], counts[chunk], offsets[chunk], splitter,
      final_bit, values, indices);
emit(sm, input, head_count, 0, splitter, final_bit, values, indices);
emit(sm, input + kLength - tail, tail_count, kLength - tail, splitter,
    final_bit, values, indices);

// kernel.cu: prepare has no dynamic-memory attribute; kernel uses:
config.dynamicSmemBytes = 0;
```

## After

```cuda
// native.cuh: insert inside namespace native, before scan_step.
__device__ __forceinline__ void copy(float* dst, const float* src,
    unsigned bytes) {
  // Publish every striped store before any thread reads this chunk.
  const unsigned count = bytes / sizeof(float);
  for (unsigned i = threadIdx.x; i < count; i += blockDim.x)
    dst[i] = src[i];
  __syncthreads();
}

// topk.cuh: add after kChunks.
constexpr int kDynamicBytes = kChunks * kChunkBytes + kAlign;
// Append inside Shared, after Scan scan;:
float edges[2 * kAlignItems];
// Replace the existing constant after Shared; retain the size assertion.
constexpr int kStaticBytes = 8576;

// select: insert immediately after __shared__ Shared sm;.
extern __shared__ char dynamic[];
float* keys = reinterpret_cast<float*>((reinterpret_cast<uintptr_t>(dynamic)
    + kAlign - 1) & ~(uintptr_t(kAlign) - 1));

// First-pass region: load once, then histogram shared keys.
#pragma unroll 1
for (int chunk = 0; chunk < kChunks; ++chunk) {
  native::copy(keys + chunk * kChunkItems, input + offsets[chunk],
      counts[chunk] * sizeof(float));
  histogram<Pass::first>(sm, keys + chunk * kChunkItems, counts[chunk],
      0, 0, start_bit(0), kBuckets - 1);
}
if (tid < head_count) sm.edges[tid] = input[tid];
if (tid < tail_count) sm.edges[kAlignItems + tid] = input[kLength - tail + tid];
histogram<Pass::first>(sm, sm.edges, head_count, 0, 0, start_bit(0), kBuckets - 1);
histogram<Pass::first>(sm, sm.edges + kAlignItems, tail_count,
    0, 0, start_bit(0), kBuckets - 1);
__syncthreads();

// Later-pass region: the cached chunks have no valid-range holes here.
if (pass != 0) {
  histogram<Pass::later>(sm, keys, counts[0] + counts[1],
      splitter, prior_bit, bit, mask);
  histogram<Pass::later>(sm, sm.edges, head_count, splitter, prior_bit, bit, mask);
  histogram<Pass::later>(sm, sm.edges + kAlignItems, tail_count,
      splitter, prior_bit, bit, mask);
}

// Final emission: preserve the original global index bases.
#pragma unroll 1
for (int chunk = 0; chunk < kChunks; ++chunk)
  emit(sm, keys + chunk * kChunkItems, counts[chunk], offsets[chunk], splitter,
      final_bit, values, indices);
emit(sm, sm.edges, head_count, 0, splitter, final_bit, values, indices);
emit(sm, sm.edges + kAlignItems, tail_count, kLength - tail, splitter,
    final_bit, values, indices);

// kernel.cu: insert in prepare after the nonportable-cluster check.
status = cudaFuncSetAttribute(topk::select,
    cudaFuncAttributeMaxDynamicSharedMemorySize, topk::kDynamicBytes);
TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
// kernel: replace the dynamic shared-memory launch size.
config.dynamicSmemBytes = topk::kDynamicBytes;
```

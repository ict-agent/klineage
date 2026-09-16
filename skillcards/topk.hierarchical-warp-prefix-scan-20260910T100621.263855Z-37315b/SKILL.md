---
skill_id: topk.hierarchical-warp-prefix-scan
intent: Compute histogram prefixes with warp shuffles and shared warp totals.
preconditions:
- 'Data types: count addition and prefix subtraction must be exact, with associative
  sums that fit the counter type; regrouping must preserve bucket decisions. Score
  dtype adds no requirement because only counts are scanned.'
- 'Layout: threads own consecutive, nonoverlapping runs of ordered buckets, in lane
  then warp order; this lets local and warp prefixes concatenate into the correct
  bucket prefix. Histogram accesses require only their element alignment.'
- 'Storage: each scanning CTA can read its complete histogram through CTA-visible
  storage; all bucket counts are needed to form the prefix without cross-CTA fetches
  during the scan.'
- 'Pipeline: histogram producers must finish and make counts visible before scanning,
  and counts must remain stable through all reads. Every CTA thread must reach the
  scan barrier, every lane named by the shuffle mask must participate, and scratch
  readers must finish before the next scan reuses storage; otherwise exchange or reuse
  races invalidate prefixes.'
- 'Hardware: CUDA synchronized upward shuffles over 32-lane warps, CTA shared memory,
  and CTA barriers are required by this implementation. Full-warp participation is
  required by its full mask. Shared capacity must cover retained storage plus one
  counter per warp and any structure padding, so warp totals can coexist with the
  histogram.'
scope:
  cases:
  - topk
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Replace repeated sums of earlier histogram buckets with a hierarchical prefix
scan in `solution/topk.cuh::scan`. Each thread sums its own consecutive bucket
run, scans thread totals within its warp, then adds totals of preceding warps.
This removes redundant histogram reads while preserving exact bucket counts and
exclusive prefixes.

## Replay

1. In `solution/topk.cuh`, add `kWarps` beside the launch constants, insert `Scan`
   before `Shared`, append `Scan scan;` to `Shared`, and change `kStaticBytes` as
   shown below. Existing field offsets remain unchanged.
2. Add `scan_step` inside `namespace native` in `solution/native.cuh`, before its
   closing brace. Keep the existing `kWarp` and `kWarpMask` constants.
3. Replace `scan` with the After implementation. Thread `t` owns buckets
   `[t*kBucketsPerThread, (t+1)*kBucketsPerThread)`. The last lane writes its warp
   total; a CTA barrier publishes all totals before other warps read them.
4. Retain both call sites and all caller synchronization in `select`. Nonleader
   CTAs scan their stable local histograms between cluster arrival and wait;
   rank zero calls `choose` only after remote histogram additions finish at the
   cluster wait. The pre-scan CTA barrier completes local atomic updates. The
   existing barrier before `reset` prevents overwriting histogram data while
   readers remain; subsequent scan calls occur after all previous scratch readers
   finish. The final pass never reuses scan scratch.

No launch or host ABI changes are needed. Preserve `solution/kernel.cu` and
`config.toml`, caller device/stream handling, zero dynamic shared memory, and
cluster resource attributes. Keep histogram construction, radix decisions,
chunk ownership, alignment peeling, emission, and their bounds checks intact.
Every bucket is assigned exactly once, so this scan needs no padding or bucket
bounds guard. For another bucket count or launch, adapt ownership and guard or
zero-pad incomplete runs while keeping every shuffle lane and barrier participant
active.

The scan operates only on counts. Preserve `ordered`, the bitwise FP32 key
ordering, exact selected FP32 bits, int64 token indices, and unspecified output
and tie order. Unsigned count regrouping is exact here; no floating-point
arithmetic, rounding, tolerance, or oracle changes are involved. Validate with
`klineage.harness.evaluate` on the unchanged problem; keep evidence outside this
card.

## Example configuration

Replay uses one row of 131072 FP32 scores and selects 2048 values with int64
indices. The CTA has 512 threads (16 warps), 2048 histogram buckets, and four
consecutive buckets per thread. Warp scans use offsets 1, 2, 4, 8, and 16, then
serially accumulate the 16 shared warp totals. Keep the shown unrolling.

Restore the original `Scan` layout: alignment 32, 16 unsigned warp totals,
16 reserved `warp_storage` bytes, and one reserved unsigned `block_prefix`.
Only `warps` participates in this scan; the reserved fields preserve the replay
layout. `Scan` occupies 96 bytes. `Shared` grows from 8224 to 8320 static bytes;
its existing histogram, state, and output-counter fields stay in place.

Retain the SM90 cluster grid `(1,16,1)`, cluster dimension `(1,16,1)`, block
`(512,1,1)`, `__launch_bounds__(kThreads,1)`, nonportable cluster-size allowance,
25% preferred shared-memory carveout, and build target `9.0a`. Cluster features
serve the retained algorithm, not the prefix-scan technique.

Retain 11-bit radix selection over 32-bit ordered keys (up to three passes),
shared histograms and atomic output allocation, cluster histogram reduction,
nonleader scan/reduction overlap, and descending CTA output-prefix pushes.
Retain eight striped input items per thread, 16 KiB chunks, two chunks per CTA,
128-byte input alignment peeling, and rereads from global memory on each pass
and emission. These settings do not constrain the scan technique.

# Precondition

- Data types: count sums must be associative and exact, and sums and exclusive
  prefix subtraction must fit the counter type. Otherwise regrouping changes
  bucket decisions. The score dtype adds no requirement: the scan never reads
  scores or changes their representation.
- Layout: each thread owns a consecutive, nonoverlapping run of ordered buckets;
  thread runs follow lane order within a warp and warp order within the CTA.
  The two prefix levels depend on this ordering. Histogram accesses need only
  element alignment; stronger alignment does not affect the scan.
- Storage: a scanning CTA can read all histogram counts through CTA-visible
  storage. Missing or inaccessible counts would make its prefix incomplete;
  no cross-CTA fetch supplies missing counts during the scan itself.
- Pipeline: local or remote histogram producers must complete and publish their
  writes before scan reads, and the histogram must remain unchanged until all
  reads finish. All CTA threads must reach the scratch-publication barrier;
  all lanes named by the shuffle mask must execute each shuffle. Previous
  scratch readers must finish before another scan overwrites warp totals.
  These rules prevent stale counts, invalid exchanges, and buffer reuse races.
- Hardware: this implementation needs CUDA synchronized upward shuffles across
  32-lane warps, CTA shared memory, and CTA barriers. Its full mask requires
  complete warp participation. Available shared capacity must cover retained
  storage plus `warps_per_CTA * sizeof(counter)` and structure padding; without
  it, the histogram and published warp totals cannot coexist.

# Scope

- Cases: topk
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The snippets replace `scan` and show its supporting declarations. All unshown
members of `Shared`, launch settings, and callers remain as described above.
`kWarp = 32` and `kWarpMask = 0xffffffffu` already exist in `native.cuh`.

## Before

```cuda
// solution/topk.cuh: Shared has no scan scratch.
constexpr int kStaticBytes = 8224;
static_assert(sizeof(Shared) == kStaticBytes);

__device__ __forceinline__ void scan(Shared& sm, unsigned (&counts)[kBucketsPerThread],
    unsigned (&prefixes)[kBucketsPerThread]) {
  const int tid = threadIdx.x;
  const int first = tid * kBucketsPerThread;
  #pragma unroll
  for (int i = 0; i < kBucketsPerThread; ++i)
    counts[i] = sm.hist[first + i];

  // Sum prior buckets directly; no cooperative prefix exchange.
  unsigned prefix = 0;
  for (int bucket = 0; bucket < first; ++bucket)
    prefix += sm.hist[bucket];

  #pragma unroll
  for (int i = 0; i < kBucketsPerThread; ++i) {
    prefixes[i] = prefix;
    prefix += counts[i];
  }
}
```

## After

```cuda
// solution/native.cuh: add inside namespace native.
__device__ __forceinline__ unsigned scan_step(unsigned value, int offset) {
  unsigned result;
  asm volatile("{ .reg .u32 v; .reg .pred p; "
      "shfl.sync.up.b32 v|p, %1, %2, 0, %3; "
      "@p add.u32 v, v, %1; mov.u32 %0, v; }"
      : "=r"(result) : "r"(value), "r"(offset), "r"(kWarpMask));
  return result;
}

// solution/topk.cuh: add inside namespace topk, before Shared.
constexpr int kWarps = kThreads / native::kWarp;
struct alignas(32) Scan {
  unsigned warps[kWarps];
  char warp_storage[kWarps];
  unsigned block_prefix;
};

// Append this member inside Shared, after local_tied:
//   Scan scan;
constexpr int kStaticBytes = 8320;
static_assert(sizeof(Shared) == kStaticBytes);

__device__ __forceinline__ void scan(Shared& sm, unsigned (&counts)[kBucketsPerThread],
    unsigned (&prefixes)[kBucketsPerThread]) {
  const int tid = threadIdx.x;
  const int lane = tid % native::kWarp;
  const int warp = tid / native::kWarp;
  unsigned total = 0;
  #pragma unroll
  for (int i = 0; i < kBucketsPerThread; ++i) {
    counts[i] = sm.hist[tid * kBucketsPerThread + i];
    total += counts[i];
  }

  // Scan thread totals, then publish one aggregate per warp.
  unsigned inclusive = total;
  #pragma unroll
  for (int step = 1; step < native::kWarp; step *= 2)
    inclusive = native::scan_step(inclusive, step);

  if (lane == native::kWarp - 1) sm.scan.warps[warp] = inclusive;
  __syncthreads();

  unsigned running = sm.scan.warps[0];
  unsigned warp_prefix = 0;
  #pragma unroll
  for (int w = 1; w < kWarps; ++w) {
    if (warp == w) warp_prefix = running;
    running += sm.scan.warps[w];
  }

  unsigned prefix = inclusive - total + warp_prefix;
  #pragma unroll
  for (int i = 0; i < kBucketsPerThread; ++i) {
    prefixes[i] = prefix;
    prefix += counts[i];
  }
}
```

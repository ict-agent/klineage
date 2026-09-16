---
skill_id: topk.shared-atomic-histogram
intent: Build shared histograms by distributing keys across threads and atomically
  accumulating bucket counts.
preconditions:
- 'Data types: integer counters must represent every accumulated bucket count and
  support the chosen atomic-add overload; key classification must remain exact because
  changed bucket membership changes selection.'
- 'Layout: known key indexing and bucket mapping must allow each valid key to contribute
  once to an in-range, naturally aligned counter; omissions, duplicates, or misaddressed
  counters corrupt the histogram.'
- 'Storage: keys must be readable by the contributing CTA, and its histogram must
  already reside in shared memory visible to all contributors and consumers; CTA-local
  atomics cannot combine separate CTA histograms.'
- 'Pipeline: key production and counter initialization must be visible before reads
  and updates; all updates must finish before histogram consumption, and all readers
  before reset or reuse, to prevent stale counts and lost contributions.'
- 'Hardware: shared-memory integer atomic addition for the counter representation
  and CTA barriers are required; histogram storage plus co-resident shared state,
  including padding, must fit the CTA shared-memory limit.'
scope:
  cases:
  - topk
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

In `solution/topk.cuh`, replace the complete `histogram<Pass>` helper below.
Assign keys to threads instead of assigning histogram buckets to threads. Each
thread loads indices `threadIdx.x + n*kThreads`, classifies each key, and adds
one to its shared bucket with `atomicAdd`. This removes a full key scan per
bucket. Remove the decomposition comment describing exclusive bucket ownership.

Keep `Shared::hist`, `reset`, `scan`, `choose`, `ordered`, and every call site.
The bucket address remains `sm.hist + ((bits >> bit) & mask)`. Later passes retain
the exact prefix comparison with `splitter`; the first pass counts every key.
The `index < count` guard covers partial chunks, the head, and the tail. Do not
add padding keys or change their original indices.

Keep the launch and caller stream unchanged. Existing barriers publish reset
before updates, finish histogram production before `choose`, and finish readers
before reset. Chunk calls accumulate into the same histogram without intervening
reset. Atomic updates need no barrier between chunks: no histogram consumer runs
until the existing CTA barrier. Input production remains ordered on the caller
stream; input stays unchanged throughout selection.

Preserve exact integer counts and the existing FP32 bit-to-key transformation.
No floating-point arithmetic or rounding changes. Output values retain their
input bits; int64 indices, unspecified output order, and unspecified tied-index
choice remain unchanged. Keep the independent atomic output reservations.

## Example configuration

Preserve this replay's contiguous FP32 input `[1, 131072]`, FP32 selected values
`[1, 2048]`, and int64 indices `[1, 2048]`. One CTA has 512 threads; launch bounds
are `(512, 1)`, grid is `(1, 1, 1)`, and dynamic shared memory is zero. Keep the
host's 25% preferred shared carveout (the existing Hopper 64 KiB partition).

There are 2048 unsigned 32-bit counters, four per thread in the Before mapping.
`Shared` occupies 8216 bytes, including 8192 histogram bytes. Retain 11-bit radix
refinement over three passes at bit offsets 21, 10, and 0; the last mask covers
10 bits. Keep all passes, prefix filtering, direct shared prefix sums, and the
threshold/tie output counters.

Keep 32 chunks of 4096 floats (16 KiB), the 128-byte alignment head/tail split,
and scalar global loads. The Before helper scans each chunk once per owned
bucket; the After helper visits each chunk key once across the CTA. Output
emission keeps its existing striped ownership. No cluster communication, input
cache, register key batches, async copies, or new buffers are introduced.

# Precondition

- Data types: counters use integer addition and must hold the largest count
  accumulated before reset. Overflow would change threshold selection. The
  chosen atomic overload must support their representation. Key classification
  must remain exact; no particular key dtype is required merely to count its
  bucket, but changing classification changes which values are selected.
- Layout: the valid key range and bucket mapping must be known so the new
  traversal covers each key exactly once. Each bucket must address an in-range,
  naturally aligned counter, as required by the atomic operation. Missing or
  repeated contributions and invalid counter addresses corrupt counts.
- Storage: contributing threads must read the keys, and the existing histogram
  must be CTA-shared and visible to its contributors and consumers. Updating
  another CTA's independent histogram requires a different communication design.
- Pipeline: key producers and counter initialization must complete and become
  visible before use. Histogram updates must finish before consumers read counts;
  consumers must finish before reset or reuse. Preserve these dependencies to
  avoid stale data, lost increments, and overwritten counts. They require no
  particular stage count.
- Hardware: the counter type needs shared-memory atomic integer addition and
  CTA barriers. Available CTA shared capacity must cover
  `bucket_count * sizeof(counter) + co_resident_state_and_padding`; otherwise
  the histogram and its existing state cannot reside together.

# Scope

- Cases: topk
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Both snippets use the existing constants, `Shared`, `Pass`, and `ordered` helper.
Replace only this helper; no launch or synchronization edits are needed.

## Before

```cuda
template <Pass mode>
__device__ __forceinline__ void histogram(Shared& sm, const float* keys, int count,
    unsigned splitter, int prior_bit, int bit, unsigned mask) {
  // Each bucket owner scans all keys, avoiding concurrent histogram writes.
  #pragma unroll
  for (int i = 0; i < kBucketsPerThread; ++i) {
    const unsigned bucket = threadIdx.x + i * kThreads;
    unsigned total = 0;

    #pragma unroll 1
    for (int index = 0; index < count; ++index) {
      const unsigned bits = ordered(keys[index]);
      if constexpr (mode == Pass::later) {
        if (((bits >> prior_bit) << prior_bit) != splitter) continue;
      }
      if (((bits >> bit) & mask) == bucket) ++total;
    }

    sm.hist[bucket] += total;
  }
}
```

## After

```cuda
template <Pass mode>
__device__ __forceinline__ void histogram(Shared& sm, const float* keys, int count,
    unsigned splitter, int prior_bit, int bit, unsigned mask) {
  // Consume each load immediately; retain striped thread ownership.
  #pragma unroll 1
  for (int index = threadIdx.x; index < count; index += kThreads) {
    const unsigned bits = ordered(keys[index]);
    if constexpr (mode == Pass::later) {
      if (((bits >> prior_bit) << prior_bit) != splitter) continue;
    }
    atomicAdd(sm.hist + ((bits >> bit) & mask), 1u);
  }
}
```

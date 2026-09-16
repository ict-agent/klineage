---
skill_id: topk.register-load-batching
intent: Batch independent input loads in registers before dependent selection work.
preconditions:
- 'Data types: no particular key dtype is required; preserve key bits and transforms.
  Histogram updates must commute, and output order and tied identities may vary, because
  grouping changes atomic interleaving.'
- 'Layout: each thread has a known enumeration of independent keys, allowing batches
  to visit each valid key exactly once. No additional contiguity or alignment beyond
  valid scalar accesses is required.'
- 'Storage: keys are directly readable by their consuming thread and remain unchanged
  throughout processing; output writes must not alias unread keys, or earlier loads
  may observe different values.'
- 'Pipeline: input producers and shared histogram/counter initialization must be visible
  before consumption. Retain barriers before shared reads/reset, and keep per-thread
  temporaries valid through their last use before reuse, to prevent stale or overwritten
  state.'
- 'Hardware: CUDA thread-private registers must accommodate the chosen batch payload
  plus other live state; spills would defeat register-resident staging.'
scope:
  cases:
  - topk
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Batch independent global loads in thread-private registers before histogram
atomics or output allocation. This exposes independent loads and separates their
latency from dependent work. Apply the change to both `histogram` and `emit` in
`solution/topk.cuh`; they use the same staging mechanism.

Add `constexpr int kItems = 8;` beside the existing tile constants. Replace the two
functions with the After snippets. Retain their signatures, call sites, and
`Pass` specializations. Keep `#pragma unroll 1` on batch traversal and unroll the
fixed-size load/consume loops so array slots can become scalar registers.

Thread `tid` continues to own local indices congruent to `tid` modulo `kThreads`.
For each batch, slot `i` owns `base + i*kThreads + tid`; advance `base` by
`kItems*kThreads`. `histogram` loads all valid slots before transforming keys and
incrementing buckets. `emit` retains both values and truncated ordered keys before
claiming output positions. The load phase never reads an index outside `count`;
the consume phase checks the same bound before accessing its slot. This also
handles short alignment-head and tail segments. Preserve `out < kSelected`.

No launch, shared layout, or synchronization change is needed. Input production
must precede the launch on the caller stream. Keep all `__syncthreads()` and
`native::sync()` calls in `select`: histogram initialization precedes atomics,
completed histograms precede merge/scan, readers finish before reset, and output
counter prefixes are visible before emission. Ordinary same-thread dependencies
make each register load available to its consumer; finish consumption before the
next batch refills the slots. There is no asynchronous copy or shared input cache.

Preserve `ordered`, radix masks, splitter filtering, integer counter updates,
tie-region selection, and widening of original indices. Values are copied without
floating-point arithmetic or rounding. Atomic interleavings can change output
order and selected tied indices; the supplied Top-K contract permits both. Preserve
the complete problem and numerical checker. Verify through
`klineage.harness.evaluate(kernel, Path.cwd())`; keep evidence outside this card.

## Example configuration

Replay uses one contiguous FP32 row of 131072 scores, selects 2048 values, and
writes FP32 values plus int64 indices in unspecified order. Keep 512 threads per
CTA and 16 CTAs in one SM90 cluster: grid and cluster `(1,16,1)`, block `(512,1,1)`,
zero dynamic shared bytes, and the caller CUDA stream. Retain launch bounds
`(kThreads,1)`, the nonportable-cluster attribute, and 25% shared carveout preference.

Keep 11 radix bits, up to three passes over 32-bit ordered keys, 2048 histogram
buckets, four contiguous buckets per scan thread, and the direct prefix summation.
`Shared` remains 8224 bytes per CTA. Keep cluster reductions, leader selection,
early termination, and descending CTA output-prefix pushes. These retained SM90
mechanisms are independent of register batching.

Keep the 128-byte alignment peel and 16-KiB chunks. Each CTA owns two strided chunks
at `head + (rank + chunk*kBlocks)*kChunkItems`; rank zero owns the head and the last
rank owns the tail. Within a chunk, each stripe consists of adjacent scalar loads
by adjacent threads. The replay batch has eight keys per thread: 32 payload bytes
in `histogram`, and 64 in `emit` including ordered-key words. These are source-level
payload sizes, not measured register allocations. Keep `kernel.cu`, `native.cuh`,
`config.toml`, and all other `topk.cuh` functions unchanged.

# Precondition

- Data types: no particular key dtype is required; register copies must retain
  key representation and existing transforms. Histogram updates must commute,
  and output order and tied identities may vary. Grouping changes cross-thread
  atomic interleaving, so an order-sensitive update or deterministic tie/output
  contract would require a different design.
- Layout: threads must have a known enumeration of independent keys that can be
  grouped without omissions or duplicates. The bounds predicate must cover every
  populated and consumed slot. No extra contiguity or alignment is required beyond
  valid scalar accesses; grouping loads does not introduce a vector-load instruction.
- Storage: each consuming thread must be able to read its keys directly, and the
  keys must remain unchanged throughout processing. Output writes must not alias
  unread keys: hoisting such a read could otherwise change which value it observes.
- Pipeline: producers must finish and make input visible before loads; shared
  histogram/counter initialization must be visible before updates. Retain barriers
  that finish shared updates before readers and finish readers before reset.
  Per-thread temporaries must be initialized before use and remain valid through
  their last reader before reuse. These ordering constraints prevent stale reads
  and premature reuse; they impose no fixed stage count.
- Hardware: CUDA thread-private register storage must hold the chosen batch's key
  payload and any retained derived words, alongside other live state. Account for
  per-thread limits and CTA allocation from the SM register file. Insufficient
  capacity causes spills and loses register-resident staging; no special copy or
  cluster instruction is introduced by this technique.

# Scope

- Cases: topk
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The surrounding constants, `Shared`, `Pass`, `ordered`, and callers already exist.
Before has no `kItems` constant. After adds it and replaces these two functions.

## Before

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

__device__ __forceinline__ void emit(Shared& sm, const float* keys, int count,
    int global_base, unsigned splitter, int bit, float* values, int64_t* indices) {
  // Finish each value/index pair before loading the next key.
  #pragma unroll 1
  for (int index = threadIdx.x; index < count; index += kThreads) {
    const float value = keys[index];
    const unsigned bits = (ordered(value) >> bit) << bit;
    if (bits > splitter) continue;
    unsigned* counter = bits == splitter ? &sm.tied : &sm.selected;
    const unsigned out = atomicAdd(counter, 1u);
    if (out >= kSelected) continue;
    values[out] = value;
    indices[out] = static_cast<int64_t>(global_base + index);
  }
}
```

## After

```cuda
constexpr int kItems = 8;

template <Pass mode>
__device__ __forceinline__ void histogram(Shared& sm, const float* keys, int count,
    unsigned splitter, int prior_bit, int bit, unsigned mask) {
  // Load eight striped keys before issuing any shared atomic.
  #pragma unroll 1
  for (int base = 0; base < count; base += kItems * kThreads) {
    float regs[kItems];
    #pragma unroll
    for (int i = 0; i < kItems; ++i) {
      const int index = base + i * kThreads + threadIdx.x;
      if (index < count) regs[i] = keys[index];
    }
    #pragma unroll
    for (int i = 0; i < kItems; ++i) {
      const int index = base + i * kThreads + threadIdx.x;
      if (index >= count) continue;
      const unsigned bits = ordered(regs[i]);
      if constexpr (mode == Pass::later) {
        if (((bits >> prior_bit) << prior_bit) != splitter) continue;
      }
      atomicAdd(sm.hist + ((bits >> bit) & mask), 1u);
    }
  }
}

__device__ __forceinline__ void emit(Shared& sm, const float* keys, int count,
    int global_base, unsigned splitter, int bit, float* values, int64_t* indices) {
  #pragma unroll 1
  for (int base = 0; base < count; base += kItems * kThreads) {
    float regs[kItems];
    unsigned bits[kItems];
    #pragma unroll
    for (int i = 0; i < kItems; ++i) {
      const int index = base + i * kThreads + threadIdx.x;
      if (index >= count) continue;
      regs[i] = keys[index];
      bits[i] = (ordered(regs[i]) >> bit) << bit;
    }
    #pragma unroll
    for (int i = 0; i < kItems; ++i) {
      const int index = base + i * kThreads + threadIdx.x;
      if (index >= count || bits[i] > splitter) continue;
      unsigned* counter = bits[i] == splitter ? &sm.tied : &sm.selected;
      const unsigned out = atomicAdd(counter, 1u);
      if (out >= kSelected) continue;
      values[out] = regs[i];
      indices[out] = static_cast<int64_t>(global_base + index);
    }
  }
}
```

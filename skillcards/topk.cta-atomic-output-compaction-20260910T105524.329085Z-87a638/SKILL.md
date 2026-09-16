---
skill_id: topk.cta-atomic-output-compaction
intent: Compact selected keys through CTA-shared atomic slot reservation.
preconditions:
- 'Data types: output order and threshold-tie identities may vary; preserve the selection
  predicate and value bits. Integer counters must represent every reservation and
  region offset without overflow, or slots can repeat.'
- 'Layout: each candidate is visited once and each output region has one CTA owner
  with a known slot-to-address mapping; duplicate visits or independent owners would
  collide. No vector alignment or contiguous input requirement is added.'
- 'Storage: candidate values and finalized threshold state are accessible to the owning
  CTA, and output writes cannot overwrite live inputs; reservations replace position
  computation, not the predicate or its data sources.'
- 'Pipeline: input and threshold producers complete and become visible before emission.
  CTA participants can synchronize before reservations, defer scratch reuse until
  emission completes, and order output consumers after stores; otherwise initialization,
  reuse, or consumption can race.'
- 'Hardware: CUDA shared-memory atomic fetch-add for the counter type and CTA barriers
  are available. Shared capacity must cover the existing state plus two naturally
  aligned counters and structure padding; atomic return values must uniquely reserve
  slots.'
scope:
  cases:
  - topk
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Replace repeated prefix counting in `solution/topk.cuh::emit` with CTA-shared
atomic slot reservation. Each qualifying key reserves one slot, avoiding a scan
of all preceding input keys. Keep the existing selection predicate and separate
strict-winner and threshold-tie regions.

Apply these changes only in `solution/topk.cuh`:

1. Append `unsigned selected` and `unsigned tied` to `Shared`, after `state`.
   Update `kStaticBytes` from 8208 to 8216; retain its size assertion.
2. In `select`'s initial `leader_thread` block, zero both counters after assigning
   `sm.state`. Preserve `reset(sm)` and the following CTA barrier.
3. After the radix loop's existing final CTA barrier, compute
   `selected = kSelected - sm.state.remaining`. The leader adds this base to
   `sm.tied`; all threads then synchronize before any `emit` call. This reserves
   `[0, selected)` for strict winners and `[selected, kSelected)` for ties.
4. In `emit`, replace the prefix-count block with a counter choice and
   `atomicAdd(counter, 1u)`. Use `sm.selected` for `bits < splitter` and `sm.tied`
   for `bits == splitter`. Retain rejection of `bits > splitter`, the
   `out >= kSelected` store guard, and both value/index stores.
5. Remove the decomposition comment about counting prior qualifying keys.

Each candidate retains its existing thread owner and chunk traversal. Counters
persist across all chunk, head, and tail calls; never reset them between calls.
Atomic return values provide unique slots within each region. There are exactly
`selected` strict winners; excess ties reserve slots beyond K and skip stores.
No emission barrier is needed between chunks because both counters remain live.
Do not reuse their storage until all reservations and stores complete. Output
consumers remain ordered after the kernel on the caller stream; atomics reserve
positions and do not publish completion of subsequent value/index stores.

Preserve `ordered`, radix refinement, `sm.state.remaining`, and every threshold
barrier. The final bit is zero, so the tie group uses complete ordered keys.
Copy original FP32 values unchanged and widen original indices to int64. The
contract permits arbitrary output ordering and arbitrary tied indices; atomic
arrival order may change both. Do not change the problem or numerical policy.

Keep `solution/kernel.cu`, `solution/native.cuh`, `config.toml`, the host ABI,
caller stream, and launch unchanged. The snippets below are four independent
excerpts: shared declaration, initial state setup, post-radix setup, and the
qualifying-key portion of `emit`.

After replay, build the complete bundle and require acceptance from
`klineage.harness.evaluate` with the unchanged problem. Keep measurements outside
this card.

## Example configuration

This replay selects K=2048 from one contiguous FP32 row of S=131072. Inputs have
stride `(131072, 1)`; FP32 values and int64 indices have stride `(2048, 1)`.
One CTA uses 512 threads, grid `(1, 1, 1)`, block `(512, 1, 1)`, zero dynamic
shared memory, and `__launch_bounds__(kThreads, 1)`. The wrapper retains its 25%
shared-memory carveout preference and destination-passing `kernel` export.

Preserve three radix passes over 32-bit ordered keys, 11 nominal radix bits,
2048 histogram buckets, and four buckets per thread. Histogram owner `t` counts
buckets `t + i*kThreads`; `scan` assigns four consecutive buckets per thread and
directly sums preceding buckets. All keys are reread from global memory. Keep
these mechanisms and all existing unroll directives.

Retain 32 chunks of 16 KiB (4096 floats), 128-byte alignment handling, and
32-element alignment units. In `emit`, thread `t` visits local indices
`t + n*kThreads`; `global_base + index` is the original token index. Chunk,
head, and tail ranges jointly cover the row once. Prefix counting currently
recovers the row pointer with `keys - global_base`; the atomic replacement
removes that extra input scan without changing traversal or bounds.

`State` remains aligned to eight bytes. `Shared` grows by eight bytes for two
32-bit unsigned counters, from 8208 to 8216 bytes. The supplied target is
`nvidia-sm90a-cuda13`; no cluster, warp collective, async copy, shared input cache,
or cooperative prefix scan is introduced.

# Precondition

- Data types: output order and threshold-tie identities may vary; arrival order
  determines reservations, so a stable-order contract would fail. Preserve the
  selection predicate and stored value bits; the technique adds no floating-point
  arithmetic or input-dtype requirement. Integer counters must represent every
  reservation plus its region offset without overflow, which would repeat slots.
- Layout: each candidate is visited once, with one CTA owning each output region
  and a known slot-to-address mapping. Duplicate visits would duplicate outputs;
  independent CTA counters targeting the same region would collide. No additional
  vector alignment or contiguous input requirement applies to scalar reservation.
- Storage: candidate values and finalized threshold state must be accessible to
  their CTA so the unchanged predicate can classify each key. Output writes must
  not overwrite live inputs; otherwise later predicates could observe changed
  values. Reservation changes position computation, not these data sources.
- Pipeline: input and threshold producers must complete and become visible before
  emission. CTA participants must be able to synchronize before reservations and
  delay scratch reuse until emission completes; this permits visible initialization
  and prevents overwritten counters. Output consumers must wait for stores, since
  reservation alone does not publish a completed value/index pair.
- Hardware: CUDA must support shared-memory atomic fetch-add for the counter type
  and CTA barriers; unique returned values prevent concurrent writers sharing a
  slot. Available shared memory must fit existing state plus two naturally aligned
  counters and structure padding. Insufficient capacity prevents the launch;
  unsupported or misaligned atomic access cannot provide valid reservations.

# Scope

- Cases: topk
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

```cuda
// Shared declaration.
struct Shared {
  unsigned hist[kBuckets];
  State state;
};
constexpr int kStaticBytes = 8208;
static_assert(sizeof(Shared) == kStaticBytes);

// Initial state setup in select.
if (leader_thread) {
  sm.state = {kLength, kSelected, 0};
}
reset(sm);
__syncthreads();

// Existing barrier immediately after the radix loop; emit calls follow.
__syncthreads();

// In emit, after computing value and bits.
if (bits > splitter) continue;

// Count earlier keys in the same output region to assign a unique slot.
const float* input = keys - global_base;
unsigned out = bits == splitter ? kSelected - sm.state.remaining : 0;
#pragma unroll 1
for (int prior = 0; prior < global_base + index; ++prior) {
  const unsigned prior_bits = (ordered(input[prior]) >> bit) << bit;
  if (bits == splitter ? prior_bits == splitter : prior_bits < splitter) ++out;
}

if (out >= kSelected) continue;
values[out] = value;
indices[out] = static_cast<int64_t>(global_base + index);
```

## After

```cuda
// Shared declaration.
struct Shared {
  unsigned hist[kBuckets];
  State state;
  unsigned selected;
  unsigned tied;
};
constexpr int kStaticBytes = 8216;
static_assert(sizeof(Shared) == kStaticBytes);

// Initial state setup in select.
if (leader_thread) {
  sm.state = {kLength, kSelected, 0};
  sm.selected = 0;
  sm.tied = 0;
}
reset(sm);
__syncthreads();

// Existing barrier immediately after the radix loop.
__syncthreads();

// Reserve disjoint strict-winner and threshold-tie output regions.
const unsigned selected = kSelected - sm.state.remaining;
if (leader_thread) atomicAdd(&sm.tied, selected);
__syncthreads();

// In emit, after computing value and bits.
if (bits > splitter) continue;
unsigned* counter = bits == splitter ? &sm.tied : &sm.selected;
const unsigned out = atomicAdd(counter, 1u);
if (out >= kSelected) continue;
values[out] = value;
indices[out] = static_cast<int64_t>(global_base + index);
```

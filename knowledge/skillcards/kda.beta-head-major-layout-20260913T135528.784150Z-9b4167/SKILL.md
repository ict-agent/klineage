---
skill_id: kda.beta-head-major-layout
intent: Transpose the global beta buffer to coalesce per-head chunk reads.
preconditions:
- 'Data types: no additional dtype requirement; the layout permutation must preserve
  each stored element and the existing conversion and arithmetic rounding.'
- 'Layout: known producer and consumer coordinates and access sites, with neighboring
  reader lanes requesting successive elements along a currently strided axis; this
  permits consistent reindexing and contiguous reads without leaving an incompatible
  consumer.'
- 'Storage: a separately writable device-global intermediate, visible to all consumer
  blocks, with capacity for every converted element; disjoint source and destination
  prevent parallel transpose writes from corrupting unread source values.'
- 'Pipeline: the producer must finish before global readers, shared copies must become
  visible before arithmetic, and all readers must finish before either buffer is reused;
  these ordering dependencies prevent incomplete or overwritten reads.'
- 'Hardware: coalescing of adjacent-lane global accesses; no additional instruction,
  alignment beyond valid scalar accesses, or shared-memory capacity is required by
  the permutation.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Store the converted beta workspace as `[head, token]` instead of `[token, head]`.
Consecutive reader lanes then load adjacent beta elements for one head. Retain the
existing conversion pass: this change only permutes global addresses.

Apply all three edits below:

- `solution/native.cu`, `prepare_beta`: retain each thread's input index and BF16
  conversion; store at `head*kTokens+token`.
- `solution/prepare.cuh`, `load_prepare`: replace the token-major gather address
  with the logical `src` address when filling `s.beta[i]`.
- `solution/recurrence.cuh`, `load_input`: make the same replacement for `in.beta[i]`.

For extents `T` and `H`, producer index `t*H+h` maps to `h*T+t`. Each producer
thread owns one unique destination. Reader lane `i` still owns shared slot `i`;
shared beta layouts and arithmetic consumers remain unchanged. The global read
stride between consecutive tokens falls from `H*sizeof(BF16)` to `sizeof(BF16)`.
The producer's stores become strided; its input loads remain contiguous.

Preserve the producer bounds check. Each loader forms the logical head-major
index `src=head*T+token+i`, checks `src<T*H`, and zero-fills invalid slots.
At the end of a head, valid extra slots intentionally wrap into the next head.
Keep this behavior, including unused slots, rather than clamping at the head edge.

Retain the caller stream and launch order: `prepare_beta`, `prepare`, then the
ordered recurrence launches. Kernel completion publishes global beta. Keep the
existing CTA barriers after shared loading, all arithmetic barriers, and completion
before shared reuse. The stream-local global buffer may be overwritten only after
all prior consumers finish. No launch, allocation, synchronization, cache-policy,
or compiler-control change is needed.

Keep `BF16(source[index])` at the same conversion point. Preserve every subsequent
sigmoid, BF16 rounding, scalar operation, and MMA accumulation. Preserve the complete
problem, oracle, tolerances, workload inputs, and timing policy. Check the result
with `klineage.harness.evaluate`; inspect generated beta addresses to confirm the
layout. Keep verification evidence outside this card.

## Example configuration

This replay has batch 1, `kTokens=4096`, `kHeads=96`, `kDim=128`, `kChunk=16`, and
`kTiles=256`. Source beta is contiguous FP32 `[1,4096,96]`; its global intermediate
is BF16 with 393216 elements (786432 bytes). Token-major and head-major BF16 byte
strides are `(192,2)` and `(8192,2)`, respectively. Shared beta has `kBetaElems=32`
slots; each chunk uses the first 16. Padding retains head rollover and the final
allocation boundary's zero fill.

Keep 256 threads in each kernel: the conversion grid has 1536 blocks, preparation
uses `(256,96)`, and 256 recurrence launches each use `(1,96)`. The first 32 loader
threads own the 32 beta slots. Keep existing input, output, and BVK state layouts.

Retain all other constants and build flags, including `-O3`, fast math, SM90a,
launch bounds, and ptxas register-usage settings. Keep `PrepareShared=41856`,
`InputShared=18048`, and `RecurShared=124672` bytes, static scratch, BF16 carry,
workspace allocation, scalar volatile transfers, lane exchanges, chunk inversion,
and BF16 MMA with FP32 accumulators. These are replay settings and retained
mechanisms, not requirements of the global permutation.

# Precondition

- Data types: no additional dtype requirement. Permuting addresses must preserve
  each stored element; conversion and arithmetic rounding must stay unchanged.
  The transformation moves already converted values and requires no new arithmetic.
- Layout: producer and consumer coordinates must be known, and neighboring reader
  lanes must request successive elements along a strided axis. Otherwise this
  permutation does not provide contiguous reads. Every writer and reader must have
  a known access site; an unaccounted consumer would retrieve the wrong element.
- Storage: the preceding implementation must have a separately writable global
  intermediate visible to all consumer blocks, sized for every converted element.
  Source and destination must be disjoint because concurrent transpose writes could
  otherwise destroy values before their owners read them.
- Pipeline: complete production before global reads, and publish shared copies
  before arithmetic reads. Finish all readers before global or shared buffer reuse.
  Existing stream ordering and barriers can satisfy these dependencies; incomplete
  production or early reuse would expose stale or overwritten values.
- Hardware: adjacent-lane global accesses must coalesce for the intended benefit.
  No extra instruction, alignment beyond valid scalar accesses, or shared-memory
  capacity is required; only addresses within the existing global allocation change.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These are three separate replacement sites. Keep surrounding loops, signatures,
launches, and synchronization. `tid` and `token=tile*kChunk` already exist in both
loaders; all named constants and types are supplied by `native.cuh`.

## Before

```cuda
// solution/native.cu: prepare_beta body
int index = blockIdx.x*blockDim.x+threadIdx.x;
if (index >= kTokens*kHeads) return;
// Keep converted logits in the input's token-major layout.
destination[index] = BF16(source[index]);

// solution/prepare.cuh: load_prepare beta loop
for (int i = tid; i < kBetaElems; i += kPrepareThreads) {
    const int src = head * kTokens + token + i;
    // Gather token-major logits, preserving the logical head rollover.
    const int gathered = (src % kTokens) * kHeads + src / kTokens;
    s.beta[i] = src < kTokens * kHeads ? args.beta[gathered] : BF16(0);
}

// solution/recurrence.cuh: load_input beta loop
for (int i = tid; i < kBetaElems; i += kRecurThreads) {
    const int src = head * kTokens + token + i;
    // Gather token-major logits, preserving the logical head rollover.
    const int gathered = (src % kTokens) * kHeads + src / kTokens;
    in.beta[i] = src < kTokens * kHeads ? args.beta[gathered] : BF16(0);
}
```

## After

```cuda
// solution/native.cu: prepare_beta body
int index = blockIdx.x*blockDim.x+threadIdx.x;
if (index >= kTokens*kHeads) return;
int token = index/kHeads, head = index%kHeads;
destination[head*kTokens+token] = BF16(source[index]);

// solution/prepare.cuh: load_prepare beta loop
for (int i = tid; i < kBetaElems; i += kPrepareThreads) {
    const int src = head * kTokens + token + i;
    s.beta[i] = src < kTokens * kHeads ? args.beta[src] : BF16(0);
}

// solution/recurrence.cuh: load_input beta loop
for (int i = tid; i < kBetaElems; i += kRecurThreads) {
    const int src = head * kTokens + token + i;
    in.beta[i] = src < kTokens * kHeads ? args.beta[src] : BF16(0);
}
```

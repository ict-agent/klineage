---
skill_id: topk.radix-exact-bucket-termination
intent: Terminate radix refinement when the boundary bucket exactly fills the remaining
  selection.
preconditions:
- 'Data types: radix-prefix order must agree with selection order, and bucket populations,
  prefix sums, and residual ranks must be exact without overflow; their equality certifies
  that the entire boundary bucket belongs in the result.'
- 'Layout: each selection group must have one boundary-bucket owner and a common threshold
  for all participating threads; inconsistent decisions would select different buckets.
  No extra contiguity or alignment is needed because accesses are unchanged.'
- 'Storage: the completed histogram and residual rank must be available to the boundary
  owner, with selection state visible across its CTA; these provide and distribute
  the termination decision.'
- 'Pipeline: histogram producers must finish before selection, the owner must publish
  its decision before any thread branches, and all CTA threads must exit together.
  Histogram readers must finish before reset, and reset must finish before reuse;
  otherwise stale state, divergent barriers, or overwritten counts can result.'
- 'Hardware: existing CTA shared memory and barriers suffice; the selection state
  including a stop flag must fit the CTA allocation so every participant can read
  the decision.'
scope:
  cases:
  - topk
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Stop radix refinement when the boundary bucket's population equals the residual
selection rank. Every key in that bucket is then required; lower radix bits cannot
change membership. This avoids subsequent histogram and threshold passes.

In `solution/topk.cuh`, append `unsigned stop` to `State`, initialize it to zero in
`select`, and set it in the unique winning branch of `choose` using
`counts[i] == left`. After the existing pass-end barrier and updates to `splitter`
and `final_bit`, break when `sm.state.stop` is nonzero. The snippets give all edits.

Keep the barrier before `choose`: histogram atomics must finish before its reads.
Keep the barrier before `reset` so histogram readers finish, and the barrier after
reset so the next histogram sees zeros and every thread sees the same decision.
Keep the barrier after the pass loop. Do not move the branch ahead of the pass-end
barrier or threshold updates; every CTA thread must leave together.

Retain `emit`'s `(ordered(value) >> bit) << bit` comparison using `final_bit`.
An early exit leaves unexamined low bits, so comparing full keys would incorrectly
exclude part of the boundary bucket. The existing `selected` and `tied` counters
reserve disjoint ranges: the latter starts at `kSelected - sm.state.remaining`.
When stopping early, exactly `remaining` keys occupy the boundary bucket. When
all bits are exhausted, the existing output guard handles excess threshold ties.
Keep `out < kSelected`, chunk bounds, and head/tail handling unchanged.

Preserve the input problem and its numerical requirements: copy original score
bits, retain the existing ordered-key transform, and widen original token indices.
Output order and tied indices remain unspecified. No launch, ownership, input
layout, arithmetic, or output-allocation changes are needed.

## Example configuration

This replay selects 2048 of 131072 FP32 scores in one row and writes FP32 values
and int64 indices. `State` keeps `alignas(8)`; its added unsigned field occupies
existing padding, so `kStaticBytes = 8216` and `sizeof(Shared)` stay unchanged.

Retain one CTA, 512 threads, `__launch_bounds__(kThreads, 1)`, 2048 histogram
buckets, four contiguous buckets per thread in `choose`, and the direct scalar
prefix scan. The 32-bit ordered key uses 11-bit radix groups and at most three
passes, with `start_bit` values 21, 10, and 0. These pass limits stay unchanged;
only the data-dependent exit is restored.

Retain striped scalar global loads, 16 KiB chunks, 128-byte head/body/tail
partitioning, later-pass prefix filtering, shared histogram atomics, and the two
shared output counters. Preserve `config.toml`, the pybind11 destination-passing
ABI, caller device/stream, grid `(1, kBlocks, 1)`, block `(kThreads, 1, 1)`, zero
dynamic shared memory, and the host's 25-percent preferred shared carveout.

# Precondition

- Data types: radix-prefix order must match the selection comparator. Bucket
  populations, prefix sums, and residual ranks must be exact and fit their count
  representation. Otherwise equality can falsely certify a complete selection.
  No particular score dtype is imposed by the exit itself; the retained key
  transform must continue to implement the operator's numerical ordering.
- Layout: one owner must decide the boundary bucket for each selection group,
  with one threshold shared by all participants. Multiple incompatible decisions
  could cause different threads to refine different buckets. No additional
  contiguity or alignment is required because the transformation remaps no access.
- Storage: the owner must have the completed histogram and residual rank, and its
  selection state must be CTA-visible. These supply the equality test and allow
  its outcome to reach every thread without rereading scores.
- Pipeline: histogram updates must complete and become visible before selection.
  The winning owner must publish state before any thread tests termination, and
  the whole CTA must take the same exit to preserve barrier participation.
  Histogram reads must complete before reset, and reset must complete before the
  buffer is reused; otherwise readers or later passes can observe invalid counts.
- Hardware: the existing CTA shared memory and barriers provide decision storage
  and visibility. The state extended with a stop flag must fit the CTA allocation;
  otherwise the decision cannot be shared. No additional accelerator feature is
  required.

# Scope

- Cases: topk
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Each labeled fragment replaces the corresponding fragment in
`solution/topk.cuh`; preserve intervening code and all existing barriers.

## Before

```cuda
// State declaration.
struct alignas(8) State {
  unsigned candidates;
  unsigned remaining;
  unsigned bucket;
};

// select: inside the existing leader_thread initialization block.
sm.state = {kLength, kSelected, 0};

// choose: inside the existing winning-bucket branch.
const unsigned left = target - prefixes[i];
sm.state.candidates = counts[i];
sm.state.remaining = left;
sm.state.bucket = threadIdx.x * kBucketsPerThread + i;

// select: end of each radix pass, after the optional histogram reset.
__syncthreads();
const unsigned bucket = sm.state.bucket;
splitter |= bucket << bit;
final_bit = bit;
```

## After

```cuda
// State declaration; the flag occupies existing alignment padding.
struct alignas(8) State {
  unsigned candidates;
  unsigned remaining;
  unsigned bucket;
  unsigned stop;
};

// select: inside the existing leader_thread initialization block.
sm.state = {kLength, kSelected, 0, 0};

// choose: inside the existing winning-bucket branch.
const unsigned left = target - prefixes[i];
sm.state.candidates = counts[i];
sm.state.remaining = left;
sm.state.bucket = threadIdx.x * kBucketsPerThread + i;
sm.state.stop = counts[i] == left;

// select: publish state, update the threshold, then exit uniformly.
__syncthreads();
const unsigned bucket = sm.state.bucket;
splitter |= bucket << bit;
final_bit = bit;
if (sm.state.stop) break;
```

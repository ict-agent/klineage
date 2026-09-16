---
skill_id: topk.cluster-cta-output-reservation
intent: Reserve disjoint output intervals per CTA to replace contended cluster-wide
  allocation atomics.
preconditions:
- 'Data types: integer counts and allocation offsets must not overflow; reservation
  sums must be exact. No additional key dtype requirement applies because only output
  allocation changes. Output order and boundary-tie identity must be unspecified because
  reservations change both.'
- 'Layout: CTAs must own disjoint input subsets for one output set, with matching
  histogram/emission ownership and predicates, consistent bucket order, and known
  rank/shared-field mappings; otherwise local counts cannot define disjoint output
  intervals or address their destination counters. No additional input alignment or
  contiguity is required by reservation.'
- 'Storage: contributing nonleader CTAs must retain their local bucket histograms
  before reset, and cooperating CTAs must have cluster-visible shared allocation state;
  reservations need local counts and a place to exchange offsets.'
- 'Pipeline: histogram production and reset must allow a read-complete barrier between
  them, and the chosen bucket must be publishable before dependent reads. All CTA
  threads must be able to wait for allocation-state initialization and remote updates
  before emission; shared owners must stay live until remote users finish. These ordering
  and reuse boundaries prevent stale counts, overlapping writes, and expired-storage
  access.'
- 'Hardware: CUDA cluster shared-memory mapping, integer remote atomic additions,
  local integer atomics, and cluster/CTA barriers must be supported. The cluster must
  be resident and each CTA must accommodate its existing shared storage plus two counter-sized
  tallies; otherwise offset exchange or storage allocation fails.'
scope:
  cases:
  - topk
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Partition Top-K output allocation across CTAs instead of issuing every allocation
against rank zero's shared counters. In `solution/topk.cuh`, recover each nonleader
CTA's counts below and inside the selected radix bucket, then seed disjoint output
intervals. `emit` uses CTA-local `atomicAdd` after this reservation phase.

For rank `r`, let `A[r]` count keys strictly before the final truncated splitter
and `T[r]` count keys equal to it. With `F = kSelected - remaining`, initialize:

- `selected[r] = sum(A[q] for q > r)`;
- `tied[r] = F + sum(T[q] for q > r)`.

Each nonleader rank pushes its counts to lower ranks. Local increments then fill
disjoint intervals, ordered by descending CTA rank. Rank zero needs no local
histogram scan: its merged histogram chooses the global bucket, and its local
emission occupies the remainder after all higher ranks. Keep the output bound
`out < kSelected`; it truncates the boundary group to the required number.

## Replay locations and ordering

Apply the disjoint fragments below at their named locations; all other code stays
as supplied in the deoptimized bundle.

1. Append `local_selected` and `local_tied` to `Shared`, update `kStaticBytes`,
   and initialize both with the other counters. Add `tid` in `select`.
2. After the cluster barrier following `merge`, scan nonleader histograms into
   `counts_local` and `prefixes`. Rank zero still calls `choose`. Retain the CTA
   barrier before `reset` and the following cluster barrier: the register copies
   survive histogram reuse, and the chosen bucket becomes visible to every CTA.
3. After loading the chosen bucket, accumulate that bucket's local prefix into
   `local_selected` and replace `local_tied` with its count. Do this before the
   early-stop test, including the final pass. Prefixes from successive refinement
   passes describe disjoint accepted groups; summing them yields `A[r]`.
4. After the loop's CTA barrier, seed `tied` on every CTA and push each nonleader's
   tallies to lower ranks. Use atomic addition for both seeding and pushes because
   they can overlap. Keep the cluster barrier and CTA barrier before emission.
5. Change only the allocation expression in `emit` to local `atomicAdd`. Keep
   the value load, `ordered`, truncation, selection predicate, output guard,
   original value store, and widened original index unchanged.
6. Remove the terminal `native::sync()` added solely to keep rank zero's counters
   alive during remote emission. The preceding reservation barrier now completes
   all remote counter accesses. Delete the unused `native::fetch_add_remote`
   function from `solution/native.cuh`; retain `remote`, `add_remote`,
   `load_remote`, and all other native helpers. Update the decomposition comments
   describing central allocation to describe CTA reservation.

No launch or host ABI changes are needed. Preserve `solution/kernel.cu`,
`config.toml`, the caller stream, and all existing histogram/threshold barriers.
Keep every CTA participating through reservation completion. Input head/tail
handling and striped chunk ownership remain unchanged. Values keep their original
FP32 bits; indices identify their original input positions. Output order and tied
indices remain unspecified, and the problem's numerical requirements stay intact.

Rebuild from the edited bundle and validate with `klineage.harness.evaluate` using
the unchanged problem. Require acceptance and keep evidence outside this card.

## Example configuration

Preserve `B=1`, `S=131072`, `K=2048`; FP32 input/output values and int64 indices;
contiguous row-major tensors with strides `(S,1)` and `(K,1)`. Counts are unsigned
32-bit integers. `State` remains 8-byte aligned and packed into pairs of unsigned
fields for its existing 64-bit remote loads.

Keep one cluster of 16 CTAs, grid/cluster dimensions `(1,16,1)`, 512 threads per
CTA, `__launch_bounds__(kThreads,1)`, zero dynamic shared memory, nonportable cluster
size permission, and the 25% preferred shared carveout. Keep the SM90 CUDA target
and `TORCH_CUDA_ARCH_LIST=9.0a` build setting.

Retain 11-bit radix digits over 32-bit ordered keys, at most three passes, 2048
histogram buckets, and four consecutive buckets per scan thread. The bucket owner
is `bucket / kBucketsPerThread`; its register slot is the remainder. Preserve the
scalar prefix loop, shared histogram atomics, remote histogram reductions,
blocking cluster barriers, and elected initialization thread. There is no staged
input cache or register key batching.

Retain the 16 KiB chunks (4096 keys), two chunks per CTA, the 128-byte alignment
peel, and head/tail guards. Rank `r` owns chunks with offsets
`head + (r + chunk*kBlocks)*kChunkItems`; rank zero handles the head, and the last
rank handles the tail. These subsets feed both histograms and emission.

`Shared` grows from 8216 to 8224 bytes. Keep `selected` immediately before `tied`;
this snippet addresses the latter as `remote_selected + sizeof(unsigned)`.
The push maps thread `tid` to lower rank `tid`; this configuration has enough
threads to cover all lower ranks. Retain four-element per-thread scan arrays.
Other configurations require adapting these mappings and resource settings.

# Precondition

- Data types: counts and offsets must fit their integer representation, since
  wraparound would corrupt interval boundaries. No additional key dtype is needed:
  reservation changes addresses, not key representation or arithmetic. The
  contract must permit reordered outputs and different boundary-tie identities;
  CTA intervals alter allocation order and which equal candidates survive.
- Layout: one cluster's CTAs must cover disjoint input subsets for the same output
  set, using the same bucket order and matching histogram/emission ownership
  and predicates. Otherwise local tallies omit, double-count, or misclassify
  candidates, making intervals overlap or leave holes. Rank identities and shared
  field offsets must be known so pushes reach the intended counters. Reservation
  adds no input alignment or contiguity requirement; existing key loaders retain
  their own layout rules.
- Storage: nonleader local histograms must remain available before reset, because
  their prefixes and boundary counts supply the reservations without another input
  traversal. Allocation state must already be visible within the cooperating
  cluster so offset contributions can reach each destination CTA.
- Pipeline: the existing histogram production/reset boundaries must permit
  completing reads before storage reuse, and the chosen bucket must be publishable
  before dependent reads. Every CTA thread must be able to wait for allocation-state
  initialization and remote updates before emission; otherwise local atomics can
  consume incomplete starting offsets. Shared owners must remain live through their
  last remote users. These are ordering, visibility, and lifetime capabilities,
  with no required stage count.
- Hardware: the device must support CUDA cluster shared-memory mapping, remote
  integer atomic additions, CTA-local integer atomics, and the cluster/CTA barriers
  used to publish reservations. The chosen cluster must be resident together.
  Per-CTA shared capacity must cover existing storage plus two counter-sized
  tallies, including alignment; otherwise the reservation state cannot be allocated.

# Scope

- Cases: topk
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

```cuda
// solution/topk.cuh: Shared and its size assertion's constant.
struct Shared {
  unsigned hist[kBuckets];
  State state;
  unsigned selected;
  unsigned tied;
};
constexpr int kStaticBytes = 8216;

// select: after rank acquisition.
const bool leader_thread = native::elect();

// select: initialization.
if (leader_thread) {
  sm.state = {kLength, kSelected, 0, 0};
  sm.selected = 0;
  sm.tied = 0;
}

// select: each pass, immediately after merge(sm, rank, leader_hist).
native::sync();
if (rank == 0) choose(sm);

if (pass + 1 < kPasses) {
  __syncthreads();
  reset(sm);
}
native::sync();
const uint64_t result = native::load_remote(leader_state + 2 * sizeof(unsigned));
const unsigned bucket = static_cast<unsigned>(result);
splitter |= bucket << bit;
final_bit = bit;
if (result >> kKeyBits) break;

// select: after the pass loop's __syncthreads(), before emission.
const uint64_t size = native::load_remote(leader_state);
const unsigned selected = kSelected - static_cast<unsigned>(size >> kKeyBits);
if (rank == 0 && leader_thread) atomicAdd(&sm.tied, selected);
native::sync();
__syncthreads();

// emit: allocation and stores inside its existing key loop.
unsigned* counter = bits == splitter ? &sm.tied : &sm.selected;
const unsigned out = native::fetch_add_remote(native::remote(counter, 0), 1u);
if (out >= kSelected) continue;
values[out] = value;
indices[out] = static_cast<int64_t>(global_base + index);

// select: after its final emit call.
native::sync();
```

## After

```cuda
// solution/topk.cuh: add the two local tallies.
struct Shared {
  unsigned hist[kBuckets];
  State state;
  unsigned selected;
  unsigned tied;
  unsigned local_selected;
  unsigned local_tied;
};
constexpr int kStaticBytes = 8224;

// select: after rank acquisition.
const bool leader_thread = native::elect();
const int tid = threadIdx.x;

// select: initialize before the retained startup barriers.
if (leader_thread) {
  sm.state = {kLength, kSelected, 0, 0};
  sm.selected = 0;
  sm.tied = 0;
  sm.local_selected = 0;
  sm.local_tied = 0;
}

// select: preserve local bucket counts before histogram reset.
native::sync();
unsigned counts_local[kBucketsPerThread] = {};
unsigned prefixes[kBucketsPerThread] = {};
if (rank != 0) scan(sm, counts_local, prefixes);
if (rank == 0) choose(sm);

if (pass + 1 < kPasses) {
  __syncthreads();
  reset(sm);
}
native::sync();
const uint64_t result = native::load_remote(leader_state + 2 * sizeof(unsigned));
const unsigned bucket = static_cast<unsigned>(result);
splitter |= bucket << bit;
final_bit = bit;
if (rank != 0 && tid == bucket / kBucketsPerThread) {
  const int slot = bucket % kBucketsPerThread;
  atomicAdd(&sm.local_selected, prefixes[slot]);
  sm.local_tied = counts_local[slot];
}
if (result >> kKeyBits) break;

// select: after the pass loop's __syncthreads(), reserve descending-rank intervals.
const uint64_t size = native::load_remote(leader_state);
const unsigned selected = kSelected - static_cast<unsigned>(size >> kKeyBits);
if (leader_thread) atomicAdd(&sm.tied, selected);
if (sm.local_selected != 0 || sm.local_tied != 0) {
  if (tid < rank) {
    const unsigned remote_selected = native::remote(&sm.selected, tid);
    native::add_remote(remote_selected, sm.local_selected);
    native::add_remote(remote_selected + sizeof(unsigned), sm.local_tied);
  }
}
native::sync();
__syncthreads();

// emit: each CTA now allocates only within its reserved intervals.
unsigned* counter = bits == splitter ? &sm.tied : &sm.selected;
const unsigned out = atomicAdd(counter, 1u);
if (out >= kSelected) continue;
values[out] = value;
indices[out] = static_cast<int64_t>(global_base + index);

// select: return after the final emit; no remote counter users remain.
// solution/native.cuh: delete the now-unused fetch_add_remote function.
```

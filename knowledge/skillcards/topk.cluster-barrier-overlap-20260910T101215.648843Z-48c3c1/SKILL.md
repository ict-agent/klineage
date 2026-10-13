---
skill_id: topk.cluster-barrier-overlap
intent: Overlap CTA-local histogram work with split-phase cluster barriers.
preconditions:
- 'Data types: no additional representation or arithmetic requirement; splitting barriers
  neither moves values nor changes arithmetic.'
- 'Layout: overlap work must use CTA-owned data independent of pending cluster contributions;
  scanning a remote reduction destination before completion would read incomplete
  counts. No new contiguity or alignment requirement.'
- 'Storage: histograms scanned during overlap must be disjoint from the remote merge
  destination and already visible to their local readers; otherwise concurrent merging
  and scanning could race. No new buffer is required.'
- 'Pipeline: every participating cluster thread must execute matching aligned arrivals
  and waits uniformly. Local producers must finish before local reads; remote users
  must wait for initialization or merged results, and all readers must finish before
  reset, reuse, or CTA exit.'
- 'Hardware: a supported resident CUDA thread-block cluster with split release/acquire
  cluster barriers; unsupported cluster resources or missing barrier instructions
  prevent collective completion. No additional shared-memory capacity is required.'
scope:
  cases:
  - topk
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Overlap CTA-local histogram work with cluster synchronization by separating
`native::sync()` into `native::arrive()` and a later `native::wait()`.
Apply both changes below in `solution/topk.cuh::select`; together they restore
one scheduling mechanism. `solution/native.cuh` already provides the wrappers:
`sync()` calls `arrive()` followed immediately by `wait()`.

At initialization, replace the synchronization after `reset(sm)` with arrival.
Keep the following `__syncthreads()` and all first-pass histogram calls unchanged.
Insert `if (pass == 0) native::wait();` after the pass loop's histogram-completion
`__syncthreads()` and immediately before `merge(sm, rank, leader_hist)`.
The intervening work only updates each CTA's own histogram. The wait prevents
remote merges from accessing another CTA's shared storage before initialization
has completed across the cluster.

At each merge, replace synchronization with arrival, retain both zero-initialized
local arrays and the nonleader `scan`, then wait immediately before the leader's
`choose(sm)`. Nonleaders scan their own completed histograms while remote additions
update only the leader's histogram. The leader does not scan until after the wait.
Keep `counts_local` and `prefixes` live for the existing output-region bookkeeping.

Keep all other CTA barriers and cluster synchronizations, including the barrier
before histogram reset and the synchronization before reading the leader's state.
Local scans must finish before their histogram storage is reset. Cluster barriers
must remain outside divergent rank/thread branches; the first-pass wait condition
is uniform across the cluster. No launch, ownership, address mapping, bounds check,
allocation, or numerical operation changes. Preserve output value bits and indices;
output order and the choice among equal-valued boundary indices are unspecified.

## Example configuration

Replay with the existing FP32 input `[1, 131072]`, FP32 values `[1, 2048]`, and int64
indices `[1, 2048]`; preserve the complete problem and its numerical requirements.
Keep the `ordered` bit transform, three radix passes of 11/11/10 bits, early
termination, 2048 unsigned histogram bins, and direct scalar prefix summation.

Launch one cluster with grid and cluster dimensions `(1, 16, 1)`, block dimensions
`(512, 1, 1)`, zero dynamic shared memory, and 8224 static shared-memory bytes per
CTA. Retain `__launch_bounds__(kThreads, 1)`, the nonportable cluster-size attribute,
25% preferred shared-memory carveout, and the caller's CUDA stream. Build for the
existing `9.0a` target with CUDA 13.

Keep eight-key register batches, two 16 KiB chunks per CTA, the 128-byte alignment
peel, head/tail guards, striped chunk ownership, repeated global input reads,
remote unsigned histogram reductions, and descending CTA output-prefix pushes.
Each thread scans four adjacent histogram buckets; the selected bucket's owner
retains the corresponding count and prefix. These are replay settings and retained
mechanisms, not additional prerequisites for split barriers.

# Precondition

- Data types: no additional representation or arithmetic requirement. Only barrier
  placement changes; key bits, integer counts, and numerical operations are untouched.
- Layout: work moved between arrival and wait must depend only on CTA-owned data,
  not pending cluster contributions. A scan of the remote reduction destination
  would observe incomplete counts. Existing addresses remain valid; no additional
  contiguity or alignment is needed.
- Storage: histograms scanned during overlap must be disjoint from the remote merge
  destination and already visible to their local readers. This permits simultaneous local reads
  and remote additions without racing on a scanned histogram. No new buffer is used.
- Pipeline: all cluster threads must execute matching aligned arrivals and waits
  uniformly. Local producer completion must precede local histogram reads. Remote
  users must wait for cluster initialization or merged results as appropriate.
  Readers must finish before reset, buffer reuse, or CTA exit; otherwise splitting
  a barrier could expose uninitialized, stale, overwritten, or expired shared data.
- Hardware: the selected resident CUDA thread-block cluster must support split
  release/acquire cluster barriers. Missing instructions or unsupported cluster
  resources prevent collective completion. Existing shared allocations suffice;
  the transformation requires no additional shared-memory capacity.

# Scope

- Cases: topk
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The snippets show two separate regions of `select`. Comments marking intervening
code mean to preserve that code verbatim, including all histogram calls and CTA
barriers. Existing constants, helpers, and function signatures remain unchanged.

## Before

```cuda
// Initialization region, after shared state initialization.
reset(sm);
native::sync();
__syncthreads();

// Keep first-pass histogram construction and pass-loop setup.
// Inside the pass loop, after any later-pass histogram construction:
__syncthreads();
merge(sm, rank, leader_hist);

// Finish cluster reductions before scanning local histograms.
native::sync();
unsigned counts_local[kBucketsPerThread] = {};
unsigned prefixes[kBucketsPerThread] = {};
if (rank != 0) scan(sm, counts_local, prefixes);
if (rank == 0) choose(sm);
```

## After

```cuda
// Initialization region, after shared state initialization.
reset(sm);
native::arrive();
__syncthreads();

// Keep first-pass histogram construction and pass-loop setup.
// Inside the pass loop, after any later-pass histogram construction:
__syncthreads();
if (pass == 0) native::wait();
merge(sm, rank, leader_hist);

// Nonleaders scan locally while cluster reductions finish.
native::arrive();
unsigned counts_local[kBucketsPerThread] = {};
unsigned prefixes[kBucketsPerThread] = {};
if (rank != 0) scan(sm, counts_local, prefixes);
native::wait();
if (rank == 0) choose(sm);
```

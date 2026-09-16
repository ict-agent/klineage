---
skill_id: fmha.adjacent-offset-shuffle
intent: Share adjacent sequence offsets through a warp shuffle to avoid redundant
  global loads.
preconditions:
- 'Data types: no additional numerical restriction; endpoint sharing must copy the
  offset representation exactly and preserve the existing length arithmetic.'
- 'Layout: each consumer needs the offset owned by its next lane in the same warp,
  including a valid final endpoint. For S consecutive sequence owners starting at
  lane zero, S + 1 <= W, where W is the warp width; otherwise the last consumer has
  no in-warp producer. No additional memory alignment is needed.'
- 'Storage: neighboring threads currently reload the same global offsets, which must
  remain immutable while read; a shared register value must equal the global load
  it replaces.'
- 'Pipeline: offset production must complete before reads; every lane needed by the
  shuffle mask can remain active through a common exchange, including endpoint-only
  lanes, so no source is absent. Metadata consumers must wait for preparation, and
  offset or metadata storage cannot be overwritten before its readers finish.'
- 'Hardware: CUDA synchronized warp shuffle must support bit-exact transport of the
  offset representation; communication is confined to one warp and needs no shared-memory
  allocation.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Share adjacent sequence endpoints through a warp shuffle in
`solution/scheduler.cuh::prepare`. Each sequence thread currently loads
`qo[lane]` and `qo[lane + 1]`. Load each endpoint once into its owning lane,
then obtain the second endpoint from the next lane. For S sequences, this
reduces requested scalar endpoint loads from 2*S to S+1; cache-line traffic
need not decrease proportionally.

Replace only the endpoint-loading block shown below. Move the non-owner
return after the exchange. Lane `kBatch` loads the final endpoint and supplies
it to lane `kBatch - 1`; higher lanes supply zero without reading out of bounds.
All lanes execute the same shuffle with the existing `kFullWarpMask`.
Keep the three metadata stores and their integer arithmetic unchanged.

Keep `prepare`'s ABI, launch, sequence ownership, and metadata layout. Offsets
remain in global memory; the reused endpoint travels directly between registers.
The caller stream orders offset production before preparation and preparation
before `attention`. Shuffle synchronization transfers register values; it does
not replace these memory-ordering dependencies. No shared buffer or additional
barrier is introduced. Scratch metadata remains unavailable for overwrite until
attention finishes reading it; offsets remain stable through their consumers.

No attention arithmetic changes: preserve accumulation order, FP16 rounding,
tail masking, online softmax, and output ownership. Keep existing compiler flags;
no compiler controls are needed. Inspect `prepare` in generated device code:
the forward change should replace its two scalar endpoint-load instructions
with one guarded scalar load and a downward shuffle. Use the existing evaluator
on the unchanged problem for correctness and timing.

## Example configuration

Replay uses eight sequences and nine contiguous int32 cumulative Q offsets.
`prepare<<<1, 32, 0, stream>>>` has one full warp; lanes 0 through 7 write
sequence metadata and lane 8 supplies the final endpoint. Metadata retains
three `kBatch`-element int arrays: split count, Q-tile count, and batch index.
Keep `kSplitOffset`, `kBlocksOffset`, `kBatchOffset`, and the unused `ko` argument.

The retained attention specialization has FP16 packed `[16384,64,128]` Q/K/V
and output, FP32 accumulators, a row stride of 8192 half elements, Q tiles of
128 rows, and KV tiles of 176 rows. Its launch keeps 256 threads, 8640 CTAs,
and `sizeof(Shared)` dynamic shared bytes (123904). It retains two 128-thread
math groups, QK `m64n176k16` and PV `m64n128k16` WGMMA, per-thread arrays of
88 score, 64 output, and 44 packed probability registers, synchronous unswizzled
Q/K/V staging, Q reload per KV tile, descending KV traversal, shared-memory
softmax exchanges, scalar round-to-nearest half conversions, and scalar output
stores. Proxy fences, WGMMA waits, warp barriers, and CTA barriers remain intact.
These settings describe the unchanged computation, not shuffle prerequisites.

# Precondition

- Data types: no additional numerical restriction. Transfer the existing offset
  bits exactly and retain length subtraction and tile-count arithmetic. Changing
  attention operand types does not affect this register-transfer technique.
- Layout: the next lane must own the endpoint the consumer would load, including
  the final valid endpoint. With S consecutive owners from lane zero and warp
  width W, require S+1 <= W: a shuffle cannot fetch the next warp's register.
  Existing scalar-load alignment is sufficient; no vector alignment is required.
- Storage: adjacent owners currently reread common global offsets. Those offsets
  must stay immutable during reading so the neighbor's register equals the
  replaced load; no shared-memory placement is required.
- Pipeline: offset producers finish before endpoint reads. Every mask-named lane
  can remain active through a common exchange, including lanes that only supply
  an endpoint; otherwise a required source is absent. Preparation completes before
  metadata readers start, and input or metadata storage is reused only after
  its readers finish. Shuffle synchronization alone does not order global memory.
- Hardware: synchronized CUDA warp shuffles must transport the offset bits
  exactly within the warp. Cross-warp register exchange is unsupported by this
  instruction; no shared-memory capacity is needed.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Replace this block immediately after `const int lane = threadIdx.x;` in
`prepare`. Retain the following metadata stores. `kFullWarpMask` already exists
in `solution/ops.cuh`; `kBatch` is the number of sequences.

## Before

```cuda
if (lane >= kBatch) return;

// Load both endpoints without sharing offsets between lanes.
const int q = qo[lane];
const int qnext = qo[lane + 1];
```

## After

```cuda
// Load the final endpoint in its own lane before non-owners return.
const int q = lane <= kBatch ? qo[lane] : 0;
const int qnext = __shfl_down_sync(kFullWarpMask, q, 1);
if (lane >= kBatch) return;
```

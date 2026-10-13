---
skill_id: fmha.query-tile-shared-retention
intent: Retain each query tile in shared memory across KV iterations.
preconditions:
- 'Data types: no additional type requirement; the existing copy must preserve the
  same query representation and padding because only its frequency changes.'
- 'Layout: the query source coordinates, valid-row bounds, destination layout, and
  consumer ownership must be invariant across KV iterations; otherwise a retained
  tile supplies the wrong queries.'
- 'Storage: the query source is immutable during the CTA, and its distinct shared
  tile remains allocated without other writers throughout the KV loop; otherwise retained
  values become stale or are overwritten.'
- 'Pipeline: the query copy must complete and become visible to all consumers before
  their first read, and the tile must not be reused until their last read completes;
  these dependencies prevent uninitialized reads and overwrites.'
- 'Hardware: no additional feature or capacity is required; the preceding implementation
  already has the shared query tile and its publication mechanism, and hoisting changes
  neither allocation nor access instructions.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

In `solution/attention.cuh::consumer`, load the CTA's query tile once and reuse
it for every KV tile. Move the existing `load<kM>(...)` call from the KV loop to
immediately after `uint32_t prob[kProbRegs];`, before output initialization.
Remove the comment about reloading Q. All other code stays unchanged.

Each CTA owns one query tile and head. Its threads cooperatively copy Q into
`s.q`; both math groups read their respective rows from that shared tile.
The Q address, valid-row count, panel layout, and ownership do not depend on `n`.
Hoisting eliminates repeated global reads and shared writes for the same values.
Keep `load` unchanged, including its invalid-row zero fill. Keep K/V loads inside
the descending KV loop and retain score masking for the partial KV tile.

Retain `proxy_fence()` and `__syncthreads()` after the K/V loads: on the first
iteration they also publish the hoisted Q copy to the asynchronous MMA consumers.
Retain MMA fences, commits, waits, operand guards, and the loop-end CTA barrier.
Waits complete reads before buffer reuse; `s.q` has no writer after initialization
and remains live until the CTA finishes. No launch, ABI, workspace, or shared
allocation changes are needed. Keep the caller stream and metadata preparation.

Only copy placement changes. Preserve query bits, zero padding, MMA accumulation
order, online softmax, probability rounding, output rounding, and output bounds.

## Example configuration

This bundle uses FP16 Q/K/V and output, FP32 scores and accumulators, packed
`[16384,64,128]` inputs, eight sequences, and `kRow = 8192` half elements.
The supplied workload has 2048 tokens per sequence. Keep its complete ProblemSpec.
Each CTA has 256 threads in two 128-thread math groups, a 128-row Q tile, and
176-row K/V tiles. The launch is `grid = kMaxQTiles * kHeads`,
`block = kThreads`, with `sizeof(Shared)` dynamic shared bytes.

The Q tile occupies 32768 bytes. K and V each occupy 45056 bytes; reduction
scratch occupies 1024 bytes, totaling 123904 bytes. Preserve these allocations
and the existing shared alignment. `load<Rows>` assigns indices
`i = threadIdx.x + j*kThreads`, with
`row = (i / kInputPanelCols) % Rows` and
`col = (i / (Rows*kInputPanelCols))*kInputPanelCols + i % kInputPanelCols`.
Here `kInputPanelCols = 8`; shared panels have no swizzle. Global addresses use
`(qstart + row)*kRow + work.z*kDim + col` for Q. The existing guard reads only
`row < qvalid` and writes zero otherwise.

Keep SM90 WGMMA (`m64n176k16` for QK, `m64n128k16` for PV), their descriptors
and fragment layouts, synchronous operand copies, shared-memory lane exchanges,
scalar FP16 conversions and stores, direct CTA mapping, and launch bounds.
Keep `kLog2Scale = 0.127517432f`, `kScale = 0.0883883461f`, and all compile flags.
These settings describe this replay; the copy-hoisting technique does not require
these dimensions, dtypes, MMA instructions, or thread counts.

# Precondition

- Data types: no additional type requirement. Reuse the existing copy and padding
  so every consumer sees the same representation; this changes copy frequency,
  not arithmetic or conversion. A different element type only needs its existing
  correct copy operation.
- Layout: source coordinates, valid-row bounds, destination layout, and consumer
  ownership must remain invariant across the KV loop. Changing any of them between
  iterations can make the retained tile address or represent the wrong queries.
- Storage: the query source must remain immutable during the CTA, and its distinct
  shared allocation must remain live without other writers throughout the loop.
  Source mutation makes cached values stale; shared aliases or scratch writes
  destroy values needed by later iterations.
- Pipeline: complete and publish the query copy before the first consumer reads
  it. Finish the last consumer read before reusing its storage. Preserve the
  existing publication and completion synchronization; these dependencies prevent
  uninitialized reads and overwrites. No particular stage count is required.
- Hardware: no additional feature or capacity requirement. The preceding kernel
  already allocates the shared query tile and supplies its publication mechanism;
  hoisting changes neither the allocation nor the access instructions.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

In `consumer`, replace the shown initialization and loop prefix. Keep the entire
remaining loop body, including all fences, waits, and its final CTA barrier.
The trailing comment marks omitted, unchanged code.

## Before

```cuda
uint32_t prob[kProbRegs];
#pragma unroll
for (int i = 0; i < kPvRegs; ++i) out[i] = 0.f;

// Finish each tile before loading another into the same storage.
#pragma unroll 1
for (int n = last; n >= 0; --n) {
    const int base = (kstart + n * kN) * kRow + work.z * kDim;
    const int valid = length - n * kN;
    // Reload Q for this KV tile.
    load<kM>(p.q + qstart * kRow + work.z * kDim, s.q, qvalid);
    load<kN>(p.k + base, s.k, valid);
    load<kN>(p.v + base, s.v, valid);
    proxy_fence();
    __syncthreads();
    // Remaining loop body unchanged.
```

## After

```cuda
uint32_t prob[kProbRegs];
load<kM>(p.q + qstart * kRow + work.z * kDim, s.q, qvalid);
#pragma unroll
for (int i = 0; i < kPvRegs; ++i) out[i] = 0.f;

// Finish each tile before loading another into the same storage.
#pragma unroll 1
for (int n = last; n >= 0; --n) {
    const int base = (kstart + n * kN) * kRow + work.z * kDim;
    const int valid = length - n * kN;
    load<kN>(p.k + base, s.k, valid);
    load<kN>(p.v + base, s.v, valid);
    proxy_fence();
    __syncthreads();
    // Remaining loop body unchanged.
```

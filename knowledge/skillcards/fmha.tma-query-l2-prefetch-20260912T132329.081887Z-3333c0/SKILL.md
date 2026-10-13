---
skill_id: fmha.tma-query-l2-prefetch
intent: Prefetch query tiles into L2 before their demand transfers.
preconditions:
- 'Data types: No additional dtype or arithmetic requirement; prefetch uses the existing
  tensor-map representation without converting or recomputing values.'
- 'Layout: A valid TMA tensor map must describe the source dimensions, strides, and
  alignment, and the issuing thread must know the future tile coordinates; otherwise
  the prefetch cannot address the intended data. No additional contiguity or compute-fragment
  mapping is required.'
- 'Storage: The source is accessible global memory, and its tensor map is accessible
  in parameter, constant, or global memory through the tensormap proxy; these are
  the instruction source spaces.'
- 'Pipeline: Source and descriptor publication must precede early access, with their
  storage remaining valid. Future demand-load coordinates must be known early enough
  to issue prefetch ahead of demand. Preserve demand-copy completion and reader visibility,
  and reader completion before shared-buffer reuse; prefetch provides none of these
  guarantees and needs no collective participation.'
- 'Hardware: SM90 or newer and PTX 8.0 or newer support tensor L2 prefetch. No additional
  shared-memory capacity or whole-tile L2 residency is required; eviction can reduce
  the benefit without affecting correctness.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Warm the Q tile in L2 while its producer approaches the K-stage and Q-buffer
waits. Insert the After loop in `solution/scheduler.cuh`, inside `producer()`'s
`while (work.w < kBatch)`, at the start of the first `if (lane == 0)` block,
immediately before `load_k(p, s, pos, kstart + n * kN, work.z)`.
Only producer lane zero issues these requests, once per assigned work tile.

Reuse `p.qmap`, `qstart`, `work.z`, `kDim`, and `kPanelCols`. Each request names
`{col, qstart, work.z, 0}` in the existing tensor map. The map supplies dimensions,
strides, and tile extents; the loop advances through all Q column panels.
The later `load<kM, kEvictFirst>` remains the demand transfer into `s.q`.
No ownership, layout, launch, allocation, or ABI changes are needed.

The instruction is a nonblocking cache request. It cannot replace the demand
copy or its synchronization. Retain `QueryEmpty` before writing `s.q`, the
`q_full` transaction barrier before QK reads, and the consumers' completed QK
reads before their next `QueryEmpty` arrival. Keep source and descriptor
publication ordered by the caller stream; the host-built map remains in the
kernel's grid-constant parameters. Add no barrier, transaction count, or wait
for the prefetch. See the [PTX instruction specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch-tensor).

Retain the work-validity loop and use exactly the subsequent demand load's
coordinates. Partial tiles retain the tensor map's existing global boundary
handling, K-score mask, and output-row guards. Prefetch adds no output writes.
Preserve every arithmetic operation, reduction order, FP16 rounding, scale,
and the complete supplied problem and numerical gate.

## Example configuration

This replay uses packed contiguous Q/K/V/output `[16384,64,128]`, FP16 values,
FP32 accumulators, and two int32 offset arrays of length nine. The workload has
eight sequences of 2048 tokens. Attention is noncausal, without dropout, scaled
by `1/sqrt(128)`; keep the existing `kScale` and `kLog2Scale` constants.

The Q tensor map in `solution/kernel.cu::encode` has shape
`{kDim,kTokens,kHeads,1}`, byte strides
`{kHeads*kDim*sizeof(__half),kDim*sizeof(__half),0}`, box
`{kPanelCols,kM,1,1}`, and unit element strides. Its descriptor is 64-byte aligned
in `Params`. Here `kPanelCols=64`, `kM=128`, `kN=176`, and `kDim=128`:
the loop issues two requests, at columns 0 and 64, each describing 16384 bytes.
The token/head byte strides are 16384/256. Keep the map's SW128 and L2-128B
promotion settings and its existing out-of-bounds mode.

Retain the 384-thread CTA: one 128-thread producer warpgroup (only its first
32 threads run `producer`) and two 128-thread consumer warpgroups. Keep
`kActiveThreads=288`, the 24/240 register budgets, two K/V stages, Q/K/V shared
storage, and V/output buffer reuse. Keep WGMMA, online softmax, FP16 probability
packing, STSM/vectorized output, tensor-map descriptor prefetch, cache eviction
policies, batch sorting, atomic persistent scheduling, and programmatic launch
dependencies. The grid remains one CTA per SM, with `sizeof(Shared)` dynamic
shared memory and the caller CUDA stream. Preserve `config.toml`, the SM90a
build target, and all serialized compiler flags.

# Precondition

- Data types: No additional dtype or arithmetic requirement. The existing tensor
  map describes the stored representation; this cache request performs no
  arithmetic or conversion.
- Layout: The existing TMA map must correctly describe source dimensions,
  physical strides, and legal alignment. The issuer must know the tile's future
  coordinates to select the data later demanded. Incorrect mapping fetches the
  wrong region or violates TMA addressing rules. No new contiguity, alignment,
  or compute-fragment ownership restriction is added to the valid demand map.
- Storage: Source data must be accessible in global memory. The tensor map must
  be accessible through the tensormap proxy in parameter, constant, or global
  memory, as required by the instruction operands.
- Pipeline: Publish source data and descriptor before the earlier access and
  keep their storage valid. Know demand coordinates before demand so prefetch
  can run ahead. Keep the demand transfer's completion and visibility ordering
  and finish shared-buffer readers before reuse: prefetch does not make shared
  values ready or release their storage. It requires no collective participation.
- Hardware: Tensor L2 prefetch requires SM90 or newer and PTX 8.0 or newer.
  It adds no shared storage and requires no guaranteed whole-tile L2 capacity;
  cache eviction changes its benefit, not correctness.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Replace only the first lane-zero block in `producer()`'s work loop. The trailing
Q synchronization and demand load show unchanged context. All names already
exist in the deoptimized bundle.

## Before

```cuda
if (lane == 0) {
    load_k(p, s, pos, kstart + n * kN, work.z);
}
sync(Named::QueryEmpty, kActiveThreads);
if (lane == 0) load<kM, kEvictFirst>(p.qmap, s.q, &s.q_full, qstart, work.z);
```

## After

```cuda
if (lane == 0) {
    // Prefetch both Q panels into L2 before acquiring the first K stage.
    #pragma unroll
    for (int col = 0; col < kDim; col += kPanelCols) {
        asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global [%0, {%1, %2, %3, 0}];"
                     :: "l"(&p.qmap), "r"(col), "r"(qstart), "r"(work.z) : "memory");
    }
    load_k(p, s, pos, kstart + n * kN, work.z);
}
sync(Named::QueryEmpty, kActiveThreads);
if (lane == 0) load<kM, kEvictFirst>(p.qmap, s.q, &s.q_full, qstart, work.z);
```

---
skill_id: cuda.tma-descriptor-prefetch
intent: Prefetch tensor-map descriptors before TMA loads.
preconditions:
- 'Data types: no additional tensor dtype or arithmetic requirement; descriptor prefetch
  does not read tensor elements or change their representation.'
- 'Layout: descriptor addresses must be available before their TMA uses so the correct
  cache lines can be targeted; no additional tensor contiguity, alignment, swizzle,
  or thread-ownership constraint is introduced.'
- 'Storage: the descriptors must be accessible in constant or parameter memory through
  valid addresses, as required by prefetch.tensormap; no new operand or shared-memory
  allocation is needed.'
- 'Pipeline: descriptors must be initialized and visible before early access and remain
  valid through TMA use; preserve operand-copy completion, reader visibility, and
  completion before buffer reuse because prefetch establishes none of those guarantees.
  No collective participation or fixed stage count is required.'
- 'Hardware: sm_90 or newer and PTX 8.0 or newer support for prefetch.tensormap; unsupported
  targets cannot execute the instruction. No additional managed-buffer capacity is
  required; cache residency is opportunistic.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Prefetch tensor-map descriptor cache lines ahead of the first TMA loads. Issue
`prefetch.tensormap` for `p.qmap`, `p.kmap`, and `p.vmap` in
`solution/attention.cuh`, inside `native::attention`'s existing
`if (threadIdx.x == 0)` block, immediately before `init(&s.q_full, 1)`.
The existing barrier initialization and work setup provide lead time before
`ops.cuh::load` consumes these descriptors. This hides some descriptor-fetch
latency when their cache lines would otherwise be absent.

The [PTX prefetch specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prefetch-prefetchu)
defines this descriptor cache operation and its target support.

Only insert the three instructions shown below. Thread zero issues each once
per CTA; other threads need no new instruction or collective. Keep `Params`
and its `__grid_constant__` argument, descriptor encoding in `kernel.cu`,
all launch parameters, and caller-stream dispatch unchanged. Descriptor addresses
remain the same addresses passed to subsequent TMA loads. No tensor ownership,
layout, copy size, cache eviction policy, or shared-memory allocation changes.

Host encoding finishes before the launch; descriptors remain immutable during
execution. Preserve barrier initialization, its release fence, the CTA barrier,
full-barrier waits before shared-tile reads, and consumer completion arrivals
before stage reuse. Prefetch provides no operand completion signal and adds no
wait, fence, or stage. Retain sequence masks and guarded output stores. Even
CTAs beyond the valid tile range may prefetch: each receives valid descriptors,
and these instructions do not use token coordinates.

Preserve all arithmetic and rounding. Check the source diff contains only the
insertions, then use the existing evaluator with the unchanged problem for
correctness and latency; cache warming does not guarantee a speedup.

## Example configuration

Preserve this instance's settings:

- Packed NHD Q/K/V and output: FP16 `[16384, 64, 128]`, element strides
  `[8192, 128, 1]`; int32 cumulative offsets `[9]` describe eight sequences.
- `Params` contains three `alignas(64) CUtensorMap` members, encoded as rank-four
  maps. Tensor-map dimensions are `[128, 16384, 64, 1]`; byte strides are
  `[16384, 256, 0]`. Boxes use eight contiguous half elements and tile heights
  128 for Q and 176 for K/V, with unit element steps and no swizzle.
- Shared panels have address
  `(col / kInputPanelCols) * Rows * kInputPanelCols
  + row * kInputPanelCols + col % kInputPanelCols` in half elements.
  Retain one Q tile and two stages each of K and V, the existing `Shared`
  controls/reduction slots, and `config.dynamicSmemBytes = sizeof(Shared)`.
- Launch `prepare` with one 32-thread CTA; launch `attention` with 8640 CTAs of
  384 threads. A producer warp loads operands; two 128-thread math groups own
  separate 64-row Q tiles. Keep the direct batch/head/query-tile mapping and
  the producer's staggered K/V order.
- Retain QK `m64n176k16` and PV `m64n128k16` WGMMA, their fragment adapters,
  88 score registers, 64 output registers, and 44 packed-probability registers
  per math thread. Retain FP32 accumulation, reverse KV traversal, online
  softmax, scalar round-to-nearest FP16 conversions, and scalar output stores.
- Keep the `9.0a` compilation target, original compile flags, tensor-map L2
  promotion, operand eviction priorities, ABI, workload, oracle, and numerical
  tolerances. These retained settings are not dependencies of descriptor prefetch.

# Precondition

- Data types: no additional tensor dtype or arithmetic constraint. The instruction
  takes a descriptor address; it neither fetches tensor values into registers nor
  converts them. Changing tensor precision does not invalidate this technique.
- Layout: descriptor addresses must be known before their TMA uses, otherwise
  early accesses cannot target the needed cache lines. No extra tensor stride,
  alignment, swizzle, or thread-ownership requirement applies; those remain
  properties of the existing copies, not descriptor prefetch.
- Storage: descriptors must reside in accessible constant or parameter memory
  with valid addresses, the memory spaces supported by this instruction. An
  address in another space does not provide this descriptor-prefetch operation.
  No tensor or shared-memory storage is added.
- Pipeline: descriptor initialization and visibility must precede early access,
  and descriptors must remain valid through their TMA consumers. Otherwise the
  early access can target invalid or stale descriptor storage. Keep operand
  producer completion and visibility before reads and reader completion before
  buffer reuse: prefetch supplies none of these guarantees. It needs no
  collective participation or particular number of stages.
- Hardware: the instruction requires `sm_90` or newer and PTX 8.0 or newer support;
  earlier targets cannot execute it. It allocates no managed buffer, so it adds
  no shared-memory capacity constraint. Cache residency is opportunistic and
  does not require the entire tensor or descriptor working set to fit.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Replace this statement only inside the existing thread-zero initialization block
of `native::attention` in `solution/attention.cuh`. Keep the surrounding block and
all following initialization and synchronization statements unchanged.

## Before

```cuda
init(&s.q_full, 1);
```

## After

```cuda
asm volatile("prefetch.tensormap [%0];" :: "l"(&p.qmap));
asm volatile("prefetch.tensormap [%0];" :: "l"(&p.kmap));
asm volatile("prefetch.tensormap [%0];" :: "l"(&p.vmap));
init(&s.q_full, 1);
```

---
skill_id: cuda.paired-shared-transfers
intent: Combine adjacent shared-memory element transfers into aligned pairs.
preconditions:
- 'Data types: paired transfers must preserve each element''s storage bits without
  conversion; no particular numeric format is required because arithmetic is unchanged.'
- 'Layout: each transferred pair contains two adjacent, valid elements accessed by
  one thread, with its address aligned to 2*sizeof(element); no pair crosses a physical
  layout discontinuity, and store pairs have unique writers.'
- 'Storage: the scalar accesses target ordinary CTA-shared scratch; both elements
  must already reside in the addressed buffer because a wider access cannot span separate
  allocations or preserve externally meaningful per-element volatile accesses.'
- 'Pipeline: both elements must be produced and visible before their paired load,
  stores must be visible before consumer reads, and scratch cannot be overwritten
  until those reads finish; pairing must preserve these common visibility intervals.'
- 'Hardware: CUDA shared-memory instructions must support the aligned pair widths
  used; no additional shared-memory capacity is required because the same buffers
  are reused.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Restore paired shared-memory transfers in `solution/prepare.cuh`, inside
`prepare`'s decayed-matrix phase. Replace its scalar gather loop with `float2`
loads for `s.g` and `s.gt`, and packed `uint32_t` loads for adjacent BF16 values
in `s.q` and `s.k`. Unpack the loaded bits into the existing register arrays.
Replace its scalar scatter loop with `uint32_t` stores of the existing `Pair`
values to `s.qd`, `s.kd`, `s.ki`, and `s.kr`.

Use the two replacement regions below; remove the scalar-only gather comment.
Keep their enclosing `m`/`n` loops,
indices, register arrays, and intervening BF16 arithmetic. Remove both
`ld_scalar` overloads and `st_scalar`, including their introductory comment,
from `solution/native.cuh` once their calls are gone. Their volatile scalar
PTX only prevents repacking in the deoptimized implementation; the scratch has
no externally observable volatile semantics.

Each thread retains ownership of its existing adjacent element pairs. The
gather sources `q`, `k`, and `g` use row-major indexing; `gt` is a vector.
Destinations retain `offset<Rows>(r,c) = r*8 + (c&7) + (c/8)*Rows*8`.
For the existing even columns, `offset(r,c+1) == offset(r,c)+1`; no pair crosses
an eight-column slab boundary. Do not change tensor maps or shared layouts.

Retain all barriers. Normalization and gate producers finish at the existing
CTA barriers before gathering. The barrier between gathering and scattering
remains; the barrier after scattering publishes the decayed matrices to the
triangular-product readers. Keep later barriers and async proxy fences before
TMA output stores, and the existing store completion waits. Pairing does not
authorize earlier buffer reuse or remove producer/consumer ordering.

No ABI, grid, block, shared allocation, or caller-stream change is needed.
The supplied tiles are complete and every pair is in bounds. An adapted partial
tile needs a scalar tail or a guard covering both elements. Copy bits only:
preserve every BF16 conversion, multiplication, addition, FP32 accumulation,
normalization reduction order, approximate activation, and rounding point.
Check that no scalar-helper calls remain, then use the existing Kernel evaluator
with the supplied problem and unchanged correctness and timing policy.

## Example configuration

The supplied workload is BATCH=1, TOKENS=4096, HEADS=96, HEAD_DIM=128.
`kChunk=16` gives `kTiles=256`. Keep the preparation launch
`grid=(kTiles,kHeads)`, `block=kPrepareThreads=256`, its launch bounds
`(kPrepareThreads,8)`, and `sizeof(PrepareShared)=42368` dynamic shared bytes.

Keep `lane=tid%32`, `warp=tid/32`, `group=lane/4`, `pair=lane%4`, and
`m,n` in `[0,2)`. The gather indices are `t=m*2+n`, `r=m*8+warp`, and
`c=n*64+group*8+pair*2`. Registers remain `rg[4][2]`, `rgt[4][2]`,
`rq[4][2]`, and `rk[4][2]`. Eight warps cover all sixteen rows. Each warp
accesses two rows; threads share read-only `gt` values but uniquely write their
decayed-matrix pairs.

Here `float2` loads transfer 8 bytes, and BF16 pairs transfer 4 bytes through
`Pair.u`. `Pair.h[0]` and `Pair.h[1]` preserve the low/high element order.
Row strides are 512 bytes for `g` and 256 bytes for `q`/`k`; `gt` has a
4-byte element stride. Destination slabs contain eight BF16 columns, with
16 bytes per row and 256 bytes per slab. Existing array alignment and even
columns satisfy each pair's alignment. No memory allocation is added.

Retain the preparation TMA transfers, eight-column unswizzled layouts, shared
normalization/pivot exchanges, MMA fragment adapters, and tensor-core products.
The recurrence stays at `grid=(1,96)`, 192 threads: four compute warps, one load
warp, and one store warp. Preserve its three input stages, two output stages,
register intermediates, prefetching, and state reuse. `InputShared` remains
18048 bytes; `RecurShared` remains 160768 bytes. Keep separate preparation and
recurrence scratch, all host checks, compiler flags, and SM90 CUDA build settings.
These retained mechanisms are not prerequisites for pairing element transfers.

# Precondition

- Data types: preserve the storage bits of both elements without numerical
  conversion. Pairing changes memory operations, not arithmetic, so it requires
  no particular numeric format. A conversion while packing would change the
  values consumed by the unchanged computation.
- Layout: one thread must access both adjacent, valid elements at an address
  aligned to `2*sizeof(element)`. Each store pair needs a unique writer, and
  neither load nor store may cross a physical layout discontinuity. Otherwise a
  wider access is misaligned, addresses the wrong neighbor, or introduces a race.
- Storage: the preceding scalar accesses must address ordinary CTA-shared
  scratch, with both elements in the same addressed buffer. A pair cannot span
  separate allocations. Replacing scalar volatile operations is valid only when
  their individual accesses have no externally meaningful volatile semantics.
- Pipeline: both elements must be complete and visible before their paired load;
  paired stores must be visible before consumption. The scratch must remain
  unchanged until its last reader finishes. Pairing requires these common
  visibility intervals; existing barriers, fences, and completion waits still
  enforce production, consumption, and safe reuse.
- Hardware: CUDA shared-memory instructions must support the chosen aligned
  pair widths, or the paired accesses cannot execute as intended. No additional
  shared-memory capacity is needed; pairing uses the existing buffers.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The first region replaces the gather body immediately after `t`, `r`, and `c`.
The second replaces the scatter body immediately after `dst`. Keep their
surrounding loops, computation, and barriers unchanged; the snippets show only
these two regions.

## Before

```cuda
// Gather region.
#pragma unroll
for (int j = 0; j < 2; ++j) {
    rg[t][j] = ld_scalar(s.g+r*kDim+c+j);
    rgt[t][j] = ld_scalar(s.gt+c+j);
    rq[t][j] = ld_scalar(s.q+r*kDim+c+j);
    rk[t][j] = ld_scalar(s.k+r*kDim+c+j);
}

// Scatter region, after the unchanged BF16 arithmetic and dst calculation.
#pragma unroll
for (int j = 0; j < 2; ++j) {
    st_scalar(s.qd+dst+j,qd.h[j]);
    st_scalar(s.kd+dst+j,kd.h[j]);
    st_scalar(s.ki+dst+j,ki.h[j]);
    st_scalar(s.kr+dst+j,kr.h[j]);
}
```

## After

```cuda
// Gather region.
float2 g = *reinterpret_cast<float2*>(s.g+r*kDim+c);
float2 gt = *reinterpret_cast<float2*>(s.gt+c);
Pair q{*reinterpret_cast<uint32_t*>(s.q+r*kDim+c)};
Pair k{*reinterpret_cast<uint32_t*>(s.k+r*kDim+c)};
rg[t][0] = g.x; rg[t][1] = g.y;
rgt[t][0] = gt.x; rgt[t][1] = gt.y;
rq[t][0] = q.h[0]; rq[t][1] = q.h[1];
rk[t][0] = k.h[0]; rk[t][1] = k.h[1];

// Scatter region, after the unchanged BF16 arithmetic and dst calculation.
*reinterpret_cast<uint32_t*>(s.qd+dst) = qd.u;
*reinterpret_cast<uint32_t*>(s.kd+dst) = kd.u;
*reinterpret_cast<uint32_t*>(s.ki+dst) = ki.u;
*reinterpret_cast<uint32_t*>(s.kr+dst) = kr.u;
```

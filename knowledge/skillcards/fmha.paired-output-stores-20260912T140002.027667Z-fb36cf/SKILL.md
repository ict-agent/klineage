---
skill_id: fmha.paired-output-stores
intent: Combine adjacent output elements into aligned word stores.
preconditions:
- 'Data types: two 16-bit output representations must occupy the low and high halves
  of a 32-bit word in address order, with required rounding already complete; the
  store must preserve those bits, but needs no floating-point arithmetic type.'
- 'Layout: one thread owns both adjacent output elements, both are valid, and the
  pair address is 4-byte aligned; otherwise a single aligned 32-bit store cannot safely
  replace the two stores.'
- 'Storage: the final pair is available to its owning thread and its destination is
  writable global memory; the replacement writes the same bytes and requires no shared-memory
  staging.'
- 'Pipeline: values are finalized before storing, no observer requires the intermediate
  single-element update, and output reads or reuse follow an established completion
  boundary; these conditions permit combining the writes without changing visibility
  or lifetime.'
- 'Hardware: aligned 32-bit global stores and low-half-first byte representation are
  required to write the adjacent 16-bit values in one instruction; no additional shared-memory
  capacity is needed.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Combine each lane's two adjacent scalar output stores into one 32-bit global
store. In `solution/attention.cuh::epilogue`, keep `packed[i]` and replace the
two `store_half` calls with the word store shown below. Remove the unused
`kHalfBits` constant and `solution/ops.cuh::store_half` helper. Restore the
loop comment to `Store each lane's rounded pairs directly from its accumulator layout.`
No launch, ownership, indexing, or synchronization changes are needed.

`packed[i]` already contains `out[2*i]` in its low half and `out[2*i+1]` in its
high half, rounded by `half_pair` using `cvt.rn.f16x2.f32`. Store this word
unchanged. Keep the conversion, accumulation order, softmax, normalization,
probability conversion, and FP16 output representation.

Each consumer thread writes its own register fragment directly to packed-NHD
global output. The existing loop maps each pair to `(row, col)`; the second
element is `(row, col+1)`. Keep `if (row >= length) continue`. Every generated
column pair is within the head dimension. For other mappings, only combine
fully valid, aligned pairs; retain scalar handling for an unmatched tail.

The final `mma_wait<0>()`, `operands(out)`, and `rescale(out, scale)` finish
and normalize the values before the epilogue. Keep the LSE writes in place and
`signal(&s.o_empty)` after output stores. The handshake remains local to the
producer/consumer protocol; downstream output readers and reuse rely on caller
stream completion. No new shared buffer, barrier, or cross-thread participation
is introduced. Existing Q/K/V readiness and buffer-release ordering remains.

## Example configuration

Preserve `[TOKENS, HEADS, HEAD_DIM] = [16384, 64, 128]`, eight sequences,
and row stride `kRow = 8192` half elements (16384 bytes). Output is contiguous
packed-NHD. The pair address is
`p.output + (start + row)*kRow + work.z*kDim + col`.
The tensor base is aligned, row/head strides are multiples of four bytes,
and `col` is even, so the word address is aligned.

The tile is `kM=128` by `kN=176`, with two K/V stages. Preserve the
`kMaxQTiles*kHeads = 8640` CTA grid, `kThreads=384`, and dynamic allocation
`sizeof(Shared)`. The producer group has 128 threads, with one active warp;
256 consumer threads form two 128-thread math groups. Register budgets remain
24 for the producer group and 240 for each consumer thread.

In the epilogue, `tid=threadIdx.x-kGroupSize`, `lane=tid%32`, and `warp=tid/32`.
Each consumer owns 64 FP32 accumulators and 32 rounded pairs.
Preserve `kRowsPerLane=2`, `kLanesPerRow=4`, `kPairElems=2`,
`kGroupRows=8`, and `kGroupCols=8`. The mapping is:

```cuda
int row = work.y * kM + warp * kRowsPerLane * kGroupRows
          + lane / kLanesPerRow + (i % kRowsPerLane) * kGroupRows;
int col = (i / kRowsPerLane) * kGroupCols + (lane % kLanesPerRow) * kPairElems;
```

Retain SW128 TMA Q/K/V staging, its cache policies and barriers, WGMMA
QK/PV instructions and fragment adapters, online softmax, overlapping math
groups, register redistribution, direct CTA mapping, and programmatic stream
dependencies. Keep the host ABI, caller stream, build configuration, compile
flags, and complete problem. These are replay settings, not store prerequisites.

Check correctness and latency with the registered Kernel evaluator on the
unchanged problem. Keep results outside this card.

# Precondition

- Data types: the final values occupy the low and high 16-bit halves of a
  32-bit word in increasing address order. Required rounding is already
  complete, so writing the word preserves both representations. The store
  requires no particular floating-point arithmetic type; changing the bit
  order or rounding again could change the output.
- Layout: the writing thread owns both adjacent elements, both are in bounds,
  and their first address is aligned to four bytes. An aligned word store
  otherwise risks an invalid address, an out-of-bounds write, or overwriting
  another thread's output. Whole-tensor contiguity and a particular warp
  mapping are unnecessary when these pair-local conditions hold.
- Storage: the final pair is available to its owner and targets writable
  global memory. The optimization replaces writes to those same bytes;
  it needs no shared staging or communication with another owner.
- Pipeline: both values must be finalized before the word store. No observer
  may depend on seeing only the first scalar write, and output readers or
  reusers must follow an established completion boundary. Otherwise merging the writes can
  change observable ordering or race with access/reuse. No new barrier or
  stage-count requirement is imposed.
- Hardware: the target supports aligned 32-bit global stores and represents
  the lower 16 bits at the lower address. These rules make the word store
  equivalent to the two scalar stores. No additional shared-memory capacity
  is required because the packed value already exists.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The first Before block is the scalar helper in `solution/ops.cuh`; delete it.
The second is the changed part of the existing epilogue loop. Keep its
unroll directive, pair mapping, rounding above the loop, and completion
handshake below it. The After block replaces the second block.

## Before

```cuda
// Keep scalar output stores from being combined by the compiler.
__device__ __forceinline__ void store_half(__half* dst, uint16_t bits) {
    asm volatile("st.global.u16 [%0], %1;" :: "l"(dst), "h"(bits) : "memory");
}
```

```cuda
// kHalfBits = 16 is declared beside the loop's mapping constants.
if (row >= length) continue;
__half* dst = p.output + (start + row) * kRow + work.z * kDim + col;
store_half(dst, uint16_t(packed[i]));
store_half(dst + 1, uint16_t(packed[i] >> kHalfBits));
```

## After

```cuda
if (row >= length) continue;
*reinterpret_cast<uint32_t*>(p.output + (start + row) * kRow + work.z * kDim + col) = packed[i];
```

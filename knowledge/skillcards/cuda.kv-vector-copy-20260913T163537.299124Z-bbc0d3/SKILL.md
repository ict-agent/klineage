---
skill_id: cuda.kv-vector-copy
intent: Vectorize synchronous KV transfers from global to shared memory.
preconditions:
- 'Data types: no arithmetic dtype requirement; pack and unpack the same bits without
  conversion, because the transfer must preserve operand representation.'
- 'Layout: each enabled load and every store accesses a contiguous, naturally 16-byte-aligned
  span, with exclusive destination ownership and uniform validity across that span;
  v4.b32 accesses all 16 bytes and cannot mask individual elements.'
- 'Storage: valid source spans reside in global memory and all destinations in CTA
  shared memory, accessible for the full span; these are the address spaces of the
  paired load and store, and no additional shared allocation is needed.'
- 'Pipeline: sources remain stable through the copy, consumers wait for completed
  stores with the required visibility, and all readers finish before destination reuse;
  no observer may depend on intermediate scalar stores or volatile access semantics,
  since widening replaces those accesses.'
- 'Hardware: support for ld.global.v4.b32 and st.shared.v4.b32 with four 32-bit register
  operands; those operations implement the widened transfer without new shared-memory
  capacity.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Vectorize the synchronous global-to-shared KV copy. Replace only `copy_kv`
in `solution/hopper.cuh` with the After function. It moves the same 16 bytes
using one `ld.global.v4.b32` and one `st.shared.v4.b32` per thread instead of
eight scalar load/store pairs. Four integer registers preserve every operand bit.

Remove the scalar loop and its PTX `.volatile` qualifiers; those qualifiers
prevent compiler packing in the Before implementation. Retain `asm volatile`
and the memory clobbers in After. No build-flag change is needed. Leave
`row_pointer`, `copy_tiles`, `produce`, descriptors, layouts, launch, and ABI intact.

`copy_tiles` in `solution/attention.cuh` already assigns each producer lane
an aligned contiguous chunk. Its call passes `bytes=16` for valid rows and
`bytes=0` otherwise. Initialize all four registers to zero, predicate the entire
load on `bytes != 0`, and always store. Never dereference an invalid source.
The existing validity mask still suppresses padded scores. Partial chunks would
require a separate scalar tail; this workload has none.

Keep producer `shared_fence()` before the first Stage barrier: generic shared
stores must become visible to the retained asynchronous WGMMA readers. All
producer and consumer threads rendezvous there before consumption. Retain the
consumer WGMMA waits and the second Stage barrier so neither KV buffer is reused
while readers remain. Preserve the later probability writes, proxy fences,
local/peer barriers, and their existing buffer lifetimes.

The transfer performs no arithmetic. Preserve QK accumulation order, softmax
scaling and reduction grouping, BF16 probability rounding, output rounding,
and max-logit/LSE semantics. Keep the complete problem, workload, tolerances,
and caller device/stream. Check correctness and latency with the Kernel evaluator;
inspect generated instructions to confirm the widened copies. Store evidence
outside this card.

## Example configuration

The fixed problem uses 8192 tokens, 128 heads, QK width 576, value width 512,
and 2048 selected indices. Inputs and output are BF16; scores, accumulators,
maxima, and LSE use FP32; indices use signed int32. Preserve the softmax scale
`0.1352337788608801f` and all existing compiler flags, including fast math.

Each CTA handles one token and 64 heads. Keep 16384 CTAs, 384 threads,
one CTA per cluster, and 231296 dynamic shared-memory bytes. One 128-thread
warpgroup produces KV; two consume. Keep resident Q, two 64-row KV buffers,
unswizzled WGMMA cores/descriptors, QK/PV WGMMA, register accumulators, shared
reductions, and the probability tile that reuses KV0's final QK tile.

In `copy_tiles`, `group=lane/8` and `column=(lane%8)*8`. Each lane owns eight
contiguous BF16 elements per tile at rows `group + row*16` for four row steps.
The global element offset is `selected_index*params.kv_stride + column + tile*64`;
`params.kv_stride=576`, giving a 1152-byte row stride. The destination is
`sm.kv[Buffer] + tile*4096 + kv_index(group,column) + row*16*8`.
All starts preserve 16-byte alignment. Keep the four `copy_tiles` calls in
order: buffer 0 tiles [0,4), buffer 1 [4,9), buffer 0 [4,9), buffer 1 [0,4).
The enclosing loop advances two 64-row blocks at a time. These sizes and
scheduling choices are replay settings, not prerequisites of vector copying.

# Precondition

- Data types: no arithmetic dtype requirement. Copy the representation unchanged;
  integer packing must not convert or round operands.
- Layout: each enabled load and every store accesses a contiguous 16-byte span
  with natural 16-byte alignment. Each destination span has one owning thread.
  Address offsets and strides must preserve that alignment. Validity applies to
  the whole span: the vector instruction accesses all bytes without element masks.
  Misalignment violates the instruction rule; holes, overlap, or partial validity
  would copy wrong bytes or require a different tail path.
- Storage: valid source spans are accessible in global memory; all destination
  spans are accessible in CTA shared memory. These address spaces are required by the
  selected instructions. Reuse the existing destination allocation; the technique
  adds no shared-memory capacity requirement.
- Pipeline: inputs must be stable until copied, completed stores must be visible
  before consumption, and every reader must finish before overwrite. Preserve the
  barriers, waits, and visibility operations implementing those dependencies.
  Intermediate scalar writes must be unobservable, and the scalar volatile
  accesses must serve only compiler control; otherwise replacing them changes
  communication semantics. No particular buffer or stage count is required.
- Hardware: the target must support `ld.global.v4.b32` and `st.shared.v4.b32`
  using four 32-bit register operands; these are the widened transfer operations.
  No extra shared storage is allocated. Tensor-core features belong to the
  retained computation and impose no additional vector-copy requirement.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Both functions use the existing `Bf16` alias and `shared_addr` helper. Replace
the complete function in place; callers and all synchronization stay unchanged.

## Before

```cuda
__device__ __forceinline__ void copy_kv(
    const Bf16* src, Bf16* dst, int bytes) {
    constexpr int kCopyBytes = 16;
    constexpr int kCopyElems = kCopyBytes / sizeof(Bf16);

    // Volatile scalar transfers prevent packing; invalid rows still zero-fill.
#pragma unroll
    for (int i = 0; i < kCopyElems; ++i) {
        uint16_t value = 0;
        if (bytes != 0) {
            asm volatile("ld.volatile.global.b16 %0, [%1];"
                         : "=h"(value) : "l"(src + i) : "memory");
        }
        asm volatile("st.volatile.shared.b16 [%0], %1;"
                     :: "r"(shared_addr(dst + i)), "h"(value) : "memory");
    }
}
```

## After

```cuda
__device__ __forceinline__ void copy_kv(
    const Bf16* src, Bf16* dst, int bytes) {
    uint32_t x = 0, y = 0, z = 0, w = 0;

    // Keep the aligned vector width and zero-fill without an invalid load.
    if (bytes != 0) {
        asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(src) : "memory");
    }
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(shared_addr(dst)), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}
```

---
skill_id: cuda.packed-fp32-bf16-conversion
intent: Convert FP32 register pairs to BF16 with one packed conversion instruction.
preconditions:
- 'Data types: independent FP32 values require BF16 conversion with round-to-nearest,
  ties-to-even; the packed instruction must match both scalar conversions without
  moving their rounding points.'
- 'Layout: each pair is owned by one thread and consumed as a 32-bit word with the
  first BF16 in the low half and the second in the high half; reversed packing changes
  element identity. No memory alignment or global contiguity requirement is added.'
- 'Storage: both scalar inputs are available in the converting thread''s registers
  and its result is a register word; otherwise operand gathering would be an additional
  transformation. No shared-memory allocation is required.'
- 'Pipeline: both producers finish before conversion, and consumers use the completed
  result before register reuse; conversion stays at the existing rounding point. No
  additional collective participation or buffer ordering is required because the change
  is thread-local.'
- 'Hardware: native packed FP32-to-BF16 round-to-nearest conversion and a CUDA toolchain
  exposing __floats2bfloat162_rn are required to combine the work; CUDA targets SM80
  or newer support this path. No additional shared-memory capacity is needed.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Pack two independent FP32-to-BF16 conversions into one instruction in
`solution/native.cuh::quantize`. Replace that function with After below.
Remove its scalar conversion assembly, shift/OR packing, and rolled inner loop;
retain the unrolled fragment loop. The rolled loop prevents compiler pairing in
the preceding implementation and disappears with packed conversion.

For fragment `j`, convert `a.x[2*j]` to bits 0–15 and `a.x[2*j+1]` to bits 16–31
of `r.x[j]`. `__floats2bfloat162_rn(low, high)` preserves this order and rounds
each element independently to nearest, ties to even. Keep `Acc`, `Reg`, `Pair`,
all call sites, and every conversion's position relative to arithmetic unchanged.
Do not combine the conversion with preceding arithmetic or change BF16 rounding.

Ownership remains within each calling thread. No launch, addressing, shared-memory,
or synchronization changes are needed. Keep all producer-visibility and buffer-reuse
barriers in the callers. Every accumulator element is valid here; the complete
pairs require no new bounds handling. Preserve existing input and output guards.

Build with the existing flags. Inspect source-correlated machine code: `quantize`
should use packed conversion with two live FP32 inputs instead of scalar conversions
inside two-iteration loops. Other conversion sites are outside this change.
Use the existing Kernel evaluator and unchanged problem for correctness and timing.

## Example configuration

The supplied workload is KDA prefill with batch 1, 4096 tokens, 96 heads, dimension
128, and chunks of 16 tokens. Keep all these settings. Each `Acc` contains eight
FP32 values; each `Reg` contains four 32-bit words, each holding two BF16 values.
The existing `Pair` union exposes the same word as `u`, `b`, and `h[2]`.

Retain 256 threads per prepare and recurrence CTA; prepare grid `(256,96)` and
recurrence grid `(1,96)`. Prepare uses 42368 dynamic shared bytes; recurrence uses
124672, including its 18048-byte input structure. Retain the separate transpose
scratch, eight-column shared slabs, and caller CUDA stream.

Keep synchronous BF16 `mma.sync.aligned.m16n8k16` products with FP32 accumulators,
the existing reduction grouping, scalar shared transfers, shared transpose and
correction exchange, single-buffer chunk sequencing, and scalar `add_pair`.
Preserve normalization epsilon, gate approximations, BF16 intermediates, and all
build flags, including fast math and the SM90a target. These retained settings
describe this replay; they are not prerequisites for packed conversion.

# Precondition

- Data types: the scalar operations convert independent FP32 values to BF16 using
  round-to-nearest, ties-to-even. The packed operation must reproduce both results
  at the same rounding points; another format or rounding rule changes values.
- Layout: one thread owns both inputs and the destination word. Consumers expect
  the first BF16 in its low half and the second in its high half; exchanging those
  halves changes element identity. No additional memory alignment or global
  contiguity is required because this operation packs registers.
- Storage: the preceding scalar inputs and packed destination belong to the
  converting thread's registers. Values owned elsewhere would need a separate
  gathering mechanism. No shared-memory allocation is required.
- Pipeline: both input producers must complete before conversion, and consumers
  must receive its completed result before register reuse. Retain the existing
  rounding point. No additional collective participation or buffer ordering is
  required: this thread-local replacement leaves shared publication, visibility,
  and completion-before-reuse barriers in place.
- Hardware: native packed FP32-to-BF16 round-to-nearest conversion and the CUDA
  intrinsic `__floats2bfloat162_rn` are needed to combine the conversions. The native
  path supports SM80 and newer; a scalar fallback does not provide this mechanism.
  No additional shared-memory capacity is needed.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Replace only `quantize`; its surrounding types and callers already exist.

## Before

```cuda
__device__ __forceinline__ Reg quantize(const Acc& a) {
    constexpr int kElemBits = 16;
    constexpr int kPairElems = sizeof(uint32_t) / sizeof(BF16);
    constexpr int kFragments = sizeof(Reg) / sizeof(uint32_t);
    Reg r;

    #pragma unroll
    for (int j = 0; j < kFragments; ++j) {
        uint32_t packed = 0;

        // Keep conversions scalar; the rolled loop prevents compiler pairing.
        #pragma unroll 1
        for (int i = 0; i < kPairElems; ++i) {
            const float value = i == 0 ? a.x[kPairElems*j] : a.x[kPairElems*j+1];
            uint16_t bits;
            asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(value));
            packed |= uint32_t(bits) << (i * kElemBits);
        }
        r.x[j] = packed;
    }
    return r;
}
```

## After

```cuda
__device__ __forceinline__ Reg quantize(const Acc& a) {
    constexpr int kPairElems = sizeof(uint32_t) / sizeof(BF16);
    constexpr int kFragments = sizeof(Reg) / sizeof(uint32_t);
    Reg r;

    // Convert each pair together, preserving low/high element ownership.
    #pragma unroll
    for (int j = 0; j < kFragments; ++j) {
        Pair p;
        p.b = __floats2bfloat162_rn(a.x[kPairElems*j], a.x[kPairElems*j+1]);
        r.x[j] = p.u;
    }
    return r;
}
```

---
skill_id: cuda.packed-fp32-to-fp16-conversion
intent: Combine independent FP32-to-FP16 conversions into one packed instruction.
preconditions:
- 'Data types: each pair contains FP32 values requiring independent round-to-nearest-even
  conversion to IEEE binary16, without saturation or flush-to-zero modifiers; these
  are the selected instruction semantics.'
- 'Layout: one thread owns both independent conversions and can supply their results
  in the packed order its consumer expects; the instruction cannot gather values from
  other lanes. No memory contiguity or alignment requirement applies to register conversion.'
- 'Storage: both operands are available in thread registers and the consumer accepts
  their packed register result; no additional shared or global storage is required.'
- 'Pipeline: producers must finish computing both operands before conversion, consumers
  must read the converted result afterward, and asynchronous readers must finish before
  result registers are reused. This register-local replacement adds no barriers.'
- 'Hardware: the assembler and target must support cvt.rn.f16x2.f32 (PTX ISA 7.0,
  sm_80 or newer); unsupported targets cannot encode this paired conversion. No additional
  shared-memory capacity is needed.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Replace the two scalar conversions and integer packing in
`solution/ops.cuh::half_pair` with one packed conversion. Replace the
`scalar_half` and `half_pair` definitions with the After snippet; keep
the `half_pair` signature and every call site. The preceding scalar helper
is non-inlined to prevent compilation from merging its two conversions;
remove that helper and its call boundary when restoring packed conversion.
The change combines independent conversions within each thread. It preserves
the packed fragment adapter required by its consumers.

The helper returns `a` in bits 15:0 and `b` in bits 31:16. PTX writes its first
source to the upper half, so the instruction deliberately takes `%2, %1`.
Both halves retain round-to-nearest-even conversion, including subnormal and
overflow behavior; add no saturation, activation, or flush-to-zero modifiers.
No reduction, accumulation, scaling, or rounding point moves. See the
[PTX conversion specification](https://docs.nvidia.com/cuda/archive/13.1.2/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)
for packed source order and target support.

`attention.cuh::convert` calls the helper for adjacent probability values;
`attention.cuh::epilogue` calls it for adjacent output accumulator values.
Keep both mappings. Output bounds remain guarded by `row < length`, with two
scalar half stores per pair. The conversion introduces no memory accesses,
thread remapping, shared-memory layout change, or launch change.

Keep existing completion ordering: QK results are ready before softmax and
probability conversion; PV completes before probability registers are
replaced; final PV and rescaling complete before output conversion. Retain
operand fences, WGMMA waits, TMA transaction barriers, stage-reuse handshakes,
and output-completion signaling.

## Example configuration

Replay uses packed NHD FP16 Q/K/V and output `[16384,64,128]`, with FP32
accumulators and softmax state. Preserve the nine offsets delimiting eight
sequences and the head stride of 128 halves and token stride of 8192 halves.
Keep tiles `kM=128`, `kN=176`, `kDim=128`, two K/V stages, 384 CTA threads,
and 256 math threads in two 128-thread warp groups.

Each math thread converts 88 scores into 44 packed probability registers
and 64 output values into 32 packed registers. Retain WGMMA QK
`m64n176k16` and PV `m64n128k16`, descending key-tile traversal, online
softmax, `kLog2Scale=0.127517432f`, and `kScale=0.0883883461f`.
Keep the TMA SW128 layouts, shared reduction exchanges, scalar output stores,
one-tile-per-CTA scheduling, metadata preparation, `Shared` allocation,
`grid=kMaxQTiles*kHeads`, caller stream, host ABI, and compile flags.
These settings describe this replay; paired conversion does not depend on them.

Use the existing Kernel evaluator and unchanged problem for correctness and
latency. Confirm the helper preserves low/high ordering at both call sites.
Keep measurement evidence outside this card.

# Precondition

- Data types: the two FP32 inputs independently require IEEE binary16 output
  with round-to-nearest-even and no saturation or flush-to-zero modifiers.
  The paired instruction must match the original conversion semantics;
  a different rounding or representation needs a different instruction.
- Layout: both conversions belong to the same thread and are independent.
  The instruction receives only that thread's registers. Its packed result
  must follow the consumer's ordering. Register conversion imposes no
  memory contiguity or alignment requirement.
- Storage: both operands already reside in thread registers, and their
  consumer accepts a packed register result. The instruction neither loads
  operands from memory nor allocates shared or global storage.
- Pipeline: both input producers must complete before conversion, and
  consumers must read afterward. Any asynchronous reader must complete
  before its result registers are reused. These existing lifetime rules
  prevent stale inputs or overwritten operands; this register-local
  replacement introduces no barriers or stage-count requirement.
- Hardware: the assembler and target support `cvt.rn.f16x2.f32`, introduced
  in PTX ISA 7.0 and requiring `sm_80` or newer. Otherwise the instruction
  cannot be encoded. No additional shared-memory capacity is required.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

```cuda
// Keep scalar conversions separate through device compilation.
__device__ __noinline__ uint16_t scalar_half(float value) {
    uint16_t bits;
    asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(bits) : "f"(value));
    return bits;
}

__device__ __forceinline__ uint32_t half_pair(float a, float b) {
    constexpr int kHalfBits = 16;

    // Round each value separately, preserving the consumers' packed bit order.
    const uint16_t low = scalar_half(a);
    const uint16_t high = scalar_half(b);
    return uint32_t(low) | (uint32_t(high) << kHalfBits);
}
```

## After

```cuda
__device__ __forceinline__ uint32_t half_pair(float a, float b) {
    uint32_t out;
    asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(out) : "f"(a), "f"(b));
    return out;
}
```

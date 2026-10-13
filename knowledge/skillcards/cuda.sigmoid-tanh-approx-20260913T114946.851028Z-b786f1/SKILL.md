---
skill_id: cuda.sigmoid-tanh-approx
intent: Evaluate sigmoid through the native approximate hyperbolic-tangent instruction.
preconditions:
- 'Data types: FP32 sigmoid operands and results, with an error budget permitting
  approximate tanh and affine rounding; the instruction does not guarantee bitwise
  equivalence to the exponential expression.'
- 'Layout: no additional requirement; each calling thread transforms one scalar, without
  collective ownership, alignment, or contiguity constraints.'
- 'Storage: no additional requirement; the helper receives its operand by value and
  adds no memory access or storage allocation.'
- 'Pipeline: no additional requirement; ordinary scalar dependencies order operand
  production and result consumption, and the helper introduces no shared visibility
  or buffer-reuse obligation.'
- 'Hardware: the CUDA target must support tanh.approx.f32; this scalar instruction
  requires no additional shared-memory capacity.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Replace the exponential-and-reciprocal sigmoid in `solution/native.cuh::sigmoid`
with `sigmoid(x) = (tanh(x/2) + 1)/2`, using CUDA's `tanh.approx.f32`.
The native instruction shortens scalar activation evaluation. Replace only this
helper with After; all callers inherit the change.

`prepare.cuh::prepare` uses it for bounded gates and triangular-system beta
scaling. `recurrence.cuh::recur_tile` uses it for both beta fragments. Each caller
keeps its scalar operand and result in its own thread; no ownership, layout,
launch, bounds, or synchronization change is needed. Keep the existing barriers
that publish shared inputs and finish reads before buffer reuse, and retain
caller-stream ordering between preparation and recurrence.

The identity is exact over real numbers, but the instruction approximation and
FP32 affine rounding can change computed values, particularly near saturation.
Keep every surrounding BF16 conversion, arithmetic grouping, state update, and
output conversion. Validate against the complete supplied oracle and its existing
tolerances; this recipe does not promise bitwise sigmoid equivalence.

## Example configuration

Preserve BATCH=1, TOKENS=4096, HEADS=96, HEAD_DIM=128 and 16-token chunks.
Inputs q/k/v/g and output are BF16 BTHD; beta, gate parameters, and state ABI
are FP32. Shared intermediates retain their existing BF16/FP32 representations.
Global rows remain contiguous; shared matrix indexing remains
`row*8 + (col&7) + (col/8)*Rows*8`.

Keep 256 threads for each launch: beta preparation uses 1536 blocks,
preparation uses grid `(256,96)`, and recurrence uses `(1,96)`.
Retain 42368 preparation and 124672 recurrence dynamic shared bytes, the
1024-byte transpose scratch, single input buffer, and all allocation formulas.
Each recurrence warp continues to own one 16-column value tile across sequential
chunks. Keep MMA instructions and fragment adapters, shared staging, register
accumulators, normalization reduction grouping, scalar volatile transfers,
scalar BF16 addition/conversion controls, and `exp2_fast` unchanged.
Keep all compile flags, including `--use_fast_math`, launch bounds, and register
settings. No compiler-control change is needed for this replacement.

# Precondition

- Data types: the supplied instruction takes and returns FP32. The numerical
  contract must tolerate its approximation and affine rounding; algebraic
  equivalence alone cannot guarantee identical outputs. Check the actual operand
  range and propagated error under the existing contract, including saturation.
- Layout: no additional requirement. A call operates on one scalar owned by its
  calling thread; it neither groups lanes nor derives addresses from tensor layout.
- Storage: no additional requirement. The scalar argument is already available
  to the helper, which performs no loads, stores, or allocations.
- Pipeline: no additional requirement. Scalar data dependencies ensure the operand
  is produced before evaluation and the result before consumption. There are no
  new asynchronous producers, shared readers, or reusable buffers; existing caller
  synchronization still governs their surrounding memory accesses.
- Hardware: the target must support `tanh.approx.f32`, otherwise After cannot
  execute as specified. This operation adds no shared allocation or capacity
  requirement.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

```cuda
__device__ __forceinline__ float sigmoid(float x) {
    // Evaluate the logistic expression directly.
    return 1.0f / (1.0f + expf(-x));
}
```

## After

```cuda
__device__ __forceinline__ float sigmoid(float x) {
    constexpr float kHalf = 0.5f;
    float y;

    // Use the tanh identity to shorten scalar sigmoid evaluation.
    asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x * kHalf));
    return y * kHalf + kHalf;
}
```

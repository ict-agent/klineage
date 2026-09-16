---
skill_id: fmha.hoist-softmax-row-offset
intent: Reuse a hoisted scaled row maximum across softmax score evaluations.
preconditions:
- 'Data types: hoisting must preserve the product representation, rounding, and subnormal
  behavior; changed intermediate semantics can change softmax probabilities.'
- 'Layout: each thread has multiple scores using the same row maximum and scale; one
  private product can serve only those scores. No extra contiguity or alignment is
  required.'
- 'Storage: the maximum and scale are already visible to the executing thread; their
  product has no externally observed per-score side effect, so repeated evaluation
  can be removed.'
- 'Pipeline: the row-maximum reduction must finish before the product is computed;
  its operands remain invariant until all corresponding scores consume it. Finish
  those reads before reusing the private temporary for another row.'
- 'Hardware: no additional feature or memory capacity is required; the existing scalar
  arithmetic supports computing the same product once.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

In `solution/attention.cuh::softmax`, compute the scaled row maximum once
per thread and row, then reuse it in every score exponential. The preceding
kernel recomputes this invariant product through `score_offset` for each score.

Replace the block immediately after `maximum[row] = m` as shown below.
Remove `score_offset`, its per-score call, and `finite_max`. Define `scaled`
after `sum[row] *= scale[row]`, outside the column loop. Keep the negative-infinity
fallback: an entirely masked row uses a zero offset. Preserve score indexing,
column order, exponentiation, sum accumulation, and all other arithmetic.

The helper's `__noinline__` and volatile PTX prevent the compiler from sharing
repeated products. Remove both with the helper when applying this optimization.
The restored C++ multiplication uses the retained `--use_fast_math` settings:
FP32 round-to-nearest with subnormal flushing. Keep the product rounded separately
from the score expression; preserve the existing score multiply/subtract fusion.

Each thread reuses only its own row offset. There are no launch, ownership,
layout, allocation, or synchronization changes. The existing reduction completes
before the offset is computed. Its warp barriers still protect scratch publication
and reuse. Keep both CTA barriers around probability consumption and reuse, and
keep caller-stream ordering between attention and its separate epilogue.

Validate with the supplied problem and unchanged harness policy. Inspect generated
code to confirm one offset product per thread-row instead of per-score helper
calls; source-level loop movement alone does not establish this change.

## Example configuration

Preserve CUDA SM90a, the native pybind11 destination-passing ABI, caller device
and stream, and all compile flags: `-O3`, `-std=c++17`, `--use_fast_math`,
`--resource-usage`, `-lineinfo`, and `-DNDEBUG`.

The workload has FP16 packed-NHD Q/K/V/output shaped `[16384,64,128]`, strides
`[8192,128,1]`, and two int32 offset arrays of length nine. There are eight
sequences; the supplied workload uses 2048 tokens each. Arithmetic accumulates
in FP32. Retain `kLog2Scale = 0.127517432f`, `kScale = 0.0883883461f`, scalar
round-to-nearest FP16 conversions, and the existing approximate exponentials
and reciprocals.

Keep query/key tiles `kM=128`, `kN=176`, 256 threads per CTA, and the grid
`kMaxQTiles*kHeads`. Each thread handles two rows and 44 scores per row:
`kQkRegs=88`, indexed by `(col/2)*4 + row*2 + col%2`. Four lanes collectively
cover a row. Retain descending key-tile traversal and increasing scalar QK/PV
reduction order, online softmax, scalar output stores, and the separate epilogue.
Global probability panels, global reduction scratch, and epilogue state remain
unchanged. Keep the last-key-tile mask and ragged query/output bounds checks.

# Precondition

- Data types: the hoisted multiplication must retain the original representation,
  rounding, and subnormal handling. Moving it must not remove a required
  intermediate rounding; otherwise the exponential arguments can change.
- Layout: a thread must own multiple score evaluations with a common row maximum
  and scale. Reuse is confined to that group; sharing across different maxima
  would compute incorrect offsets. No extra physical contiguity or alignment
  is needed because score addresses do not change.
- Storage: both operands must already be visible to the thread. The product must
  have no externally observed per-score side effect; eliminating repeated
  evaluation would otherwise change behavior. No shared-memory placement is
  required.
- Pipeline: finish producing and reducing the maximum before computing the
  shared product. Its operands must stay invariant through their score loop,
  and every consumer must finish before the private temporary serves another
  row. These dependencies prevent stale or overwritten offsets; no particular
  stage count is required.
- Hardware: no additional hardware feature or memory capacity is required.
  The same existing scalar arithmetic computes fewer products; no new collective
  instruction or storage allocation is introduced.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The Before block includes the helper immediately preceding `softmax` and the
changed block inside its row loop, after `maximum[row] = m`. The After block
replaces that loop block; delete the helper entirely. Keep the surrounding
function and other code unchanged.

## Before

```cuda
// Keep repeated row-offset products from being shared by the compiler.
__device__ __noinline__ float score_offset(float maximum) {
    float scaled;
    asm volatile("mul.rn.ftz.f32 %0, %1, %2;"
                 : "=f"(scaled) : "f"(maximum), "f"(kLog2Scale));
    return scaled;
}

// Inside softmax, after maximum[row] = m:
sum[row] *= scale[row];
const float finite_max = m == -INFINITY ? 0.f : m;
#pragma unroll
for (int col = 0; col < kQkRegs / 2; ++col) {
    int i = (col / 2) * 4 + row * 2 + col % 2;

    // Recompute the row offset for every score.
    const float scaled = score_offset(finite_max);
    score[i] = exp2f(score[i] * kLog2Scale - scaled);
    sum[row] += score[i];
}
```

## After

```cuda
// Inside softmax, after maximum[row] = m:
sum[row] *= scale[row];
float scaled = (m == -INFINITY ? 0.f : m) * kLog2Scale;
#pragma unroll
for (int col = 0; col < kQkRegs / 2; ++col) {
    int i = (col / 2) * 4 + row * 2 + col % 2;
    score[i] = exp2f(score[i] * kLog2Scale - scaled);
    sum[row] += score[i];
}
```

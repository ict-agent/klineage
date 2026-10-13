---
skill_id: rmsnorm.aligned-global-vectorization
intent: Combine adjacent scalar global-memory accesses into aligned vector transfers.
preconditions:
- 'Data types: packed transfers must preserve scalar object representations; arithmetic
  and required rounding remain separate and unchanged, so bit packing needs no particular
  arithmetic dtype.'
- 'Layout: each thread owns contiguous groups with nonoverlapping writes; every vector
  address satisfies its transfer alignment and the whole vector is in bounds, because
  wide accesses cannot tolerate misalignment or mask trailing elements.'
- 'Storage: scalar groups already reside in global allocations addressable by the
  vector pointer; wider accesses reuse those allocations and require no new staging
  or shared capacity.'
- 'Pipeline: every group element must be produced and visible before its packed read,
  and all readers must finish before overlapping writes or buffer reuse; no new synchronization
  is required when these dependencies hold for the whole vector.'
- 'Hardware: the CUDA target/compiler must support the chosen aligned wide global
  access; the shown int4 operations require 16-byte alignment and no new shared-memory
  capability.'
scope:
  cases:
  - rmsnorm
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Restore aligned global-memory vectorization in
`solution/fused_norm.cuh`, `native_norm::fused_norm`. Replace the scalar BF16
loads and stores in both `round` loops with `int4` transfers. Each pack moves
bits for eight adjacent BF16 elements; its scalar views preserve the existing
FP32 arithmetic and BF16 rounding. The fragments below replace only global
buffer declarations, global loads/stores, and their scalar views.

Retain the `float4 sums` declarations, shared transfers, `sum_values` views,
all arithmetic loops, and their positions. In the first loop, add the input and
residual pack views after their loads, before the FP32 add loop. In the second,
add the weight/output pack views after the load block, before normalization.
The residual and output stores remain inside their existing `column < width`
guards. All other code, including `config.toml` and `solution/kernel.cu`, stays
unchanged.

Thread `t = lane + warp * kWarp` still owns the contiguous group beginning at
`column = (round * threads + t) * kVector`. No remapping, launch change, new
allocation, or new synchronization is needed. The host copies the inputs to
output buffers before in-place device computation on the caller stream.
Global activation and residual loads finish before residual overwrite. Shared
FP32 sums remain available until their owner reloads them after the reduction.
Preserve the CTA barriers and grid dependency instructions.

Keep zero initialization for inactive groups and the original full-group guards.
The supplied width contains complete groups. Other widths require a scalar tail
or separately proved padding: `column < width` alone does not protect a partial
vector. For transfer alignment `A`, global bases and row-stride byte offsets
must be aligned to `A`; a row stride is its element extent times element size.

Do not change increasing-`j` FP32 square accumulation, reduction grouping,
inverse-RMS computation, multiplication order, or BF16 round-to-nearest
conversions. Normalize the retained unrounded FP32 sums, not the rounded residual.
Validate with `klineage.harness.evaluate` and the unchanged problem; keep evidence
outside this card.

## Example configuration

- CUDA, `nvidia-sm90a-cuda13`; `[8192, 7168]` activations/residuals and `[7168]`
  weights, all BF16. FP32 sums and arithmetic produce BF16 outputs.
- `kVector = 8`, `kFloat4 = 4`, `kWarp = 32`, `kWarps = 28`; one CTA per row,
  `grid = dim3(8192)`, `block = dim3(32, 28)`, 896 threads, one round.
  Retain eight elements per thread and the same local reduction grouping.
- Contiguous rows have stride 7168 elements (14336 bytes). Existing tensor checks
  require 16-byte alignment; one `int4` contains eight BF16 values.
- Retain two `float4` shared stores and two loads per thread, `kReduction = 28`, the
  112-byte offset of `summed`, and `kSharedBytes = 28784` with its launch attribute.
  Shared memory retains unrounded sums plus the padded warp-reduction scratch.
- Retain both CTA barriers, two-level 32-lane XOR butterfly reduction,
  `rsqrt.approx.ftz.f32`, epsilon `1e-6f`, zero weight bias, and
  `__float2bfloat16_rn` conversions.
- Retain destination-passing ABI, tensor checks, two device copies, caller
  device/stream, launch error handling, programmatic stream serialization,
  `griddepcontrol.wait`, and `griddepcontrol.launch_dependents`.

# Precondition

- Data types: packed transfers must copy the original scalar object
  representations without conversion. Arithmetic and required rounding remain
  separate and unchanged; bit packing itself needs no particular arithmetic
  dtype. Representation-changing loads would alter the values being computed.
- Layout: each thread owns contiguous groups with nonoverlapping writes. Every
  vector address must satisfy its transfer alignment, and the whole vector must
  be in bounds. Misalignment violates wide-access requirements; a vector access
  cannot independently mask its trailing elements.
- Storage: the scalar groups already reside in global allocations addressable
  by the vector pointer. The transformation widens accesses within those same
  allocations. There is no additional staging or shared-capacity requirement;
  shared storage and its accesses are untouched.
- Pipeline: every group element must be produced and visible before its packed
  read. All readers must finish before overlapping writes or buffer reuse.
  These existing dependencies must hold for the whole vector because its
  elements are accessed together. No new synchronization is required when they
  already hold; in-place stores must not overwrite unread source elements.
- Hardware: the CUDA target/compiler must support the chosen aligned wide
  global-memory access. The shown `int4` operations require 16-byte alignment;
  without that support and alignment, the pointer operations cannot implement
  these vector transfers safely. No new shared-memory capability is required.

# Scope

- Cases: rmsnorm
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These are separate fragments at the named locations, not one replacement block.
Keep intervening shared-memory declarations/transfers and arithmetic unchanged.
The scalar `sum_values` views still refer to the existing `float4 sums` arrays.

## Before

```cuda
// First round loop: global buffer declarations.
__nv_bfloat16 input_values[kVector] = {};
__nv_bfloat16 residual_values[kVector] = {};

// First round loop: global loads inside the existing column guard.
#pragma unroll
for (uint32_t j = 0; j < kVector; ++j) {
  input_values[j] = input[row * input_stride + column + j];
  residual_values[j] = residual[row * residual_stride + column + j];
}

// First round loop: global store inside the existing column guard.
#pragma unroll
for (uint32_t j = 0; j < kVector; ++j) {
  residual[row * residual_stride + column + j] = residual_values[j];
}

// Second round loop: global buffer declarations.
__nv_bfloat16 output_values[kVector] = {};
__nv_bfloat16 weight_values[kVector] = {};

// Second round loop: global loads inside the existing column guard.
#pragma unroll
for (uint32_t j = 0; j < kVector; ++j) {
  weight_values[j] = weight[column + j];
}

// Second round loop: global store inside the existing column guard.
#pragma unroll
for (uint32_t j = 0; j < kVector; ++j) {
  input[row * input_stride + column + j] = output_values[j];
}
```

## After

```cuda
// First round loop: global buffer declarations.
int4 input_pack = make_int4(0, 0, 0, 0);
int4 residual_pack = make_int4(0, 0, 0, 0);

// First round loop: global loads inside the existing column guard.
input_pack = *reinterpret_cast<const int4*>(input + row * input_stride + column);
residual_pack = *reinterpret_cast<const int4*>(residual + row * residual_stride + column);

// First round loop: insert outside the guard, before the add loop.
const auto* input_values = reinterpret_cast<const __nv_bfloat16*>(&input_pack);
auto* residual_values = reinterpret_cast<__nv_bfloat16*>(&residual_pack);

// First round loop: global store inside the existing column guard.
*reinterpret_cast<int4*>(residual + row * residual_stride + column) = residual_pack;

// Second round loop: global buffer declarations.
int4 output_pack = make_int4(0, 0, 0, 0);
int4 weight_pack = make_int4(0, 0, 0, 0);

// Second round loop: global load inside the existing column guard.
weight_pack = *reinterpret_cast<const int4*>(weight + column);

// Second round loop: insert outside the guard, before normalization.
const auto* weight_values = reinterpret_cast<const __nv_bfloat16*>(&weight_pack);
auto* output_values = reinterpret_cast<__nv_bfloat16*>(&output_pack);

// Second round loop: global store inside the existing column guard.
*reinterpret_cast<int4*>(input + row * input_stride + column) = output_pack;
```

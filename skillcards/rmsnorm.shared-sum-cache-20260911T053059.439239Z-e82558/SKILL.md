---
skill_id: rmsnorm.shared-sum-cache
intent: Cache residual-add sums in shared memory to avoid rereading and recomputing
  them.
preconditions:
- 'Data types: cached sums must retain the precision and representation required by
  their later consumers; narrowing them can change normalization. Input dtype adds
  no caching requirement.'
- 'Layout: known element ownership must place each sum producer and its later consumer
  in the same CTA. Contiguous groups used by vector accesses must satisfy their natural
  alignment and bounds, or those accesses are invalid.'
- 'Storage: first-pass sums are transient arithmetic values, while later consumers
  recompute them from operands; these repeated intermediates supply the work to cache.'
- 'Pipeline: execution must permit input-producer completion before reading, sum-producer
  completion and visibility before consumption, and no cached-slot reuse before its
  last read. Earlier residual stores must not destroy another reader''s operands.'
- 'Hardware: CTA-local shared memory must have capacity for the live reduction scratch
  plus width * sizeof(sum_element) bytes; otherwise the row cache cannot coexist with
  the reduction.'
scope:
  cases:
  - rmsnorm
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Cache the first pass's unrounded sums in CTA shared memory, then reload them
for normalization. This removes the second global input/residual read and add.
Apply the changes in `submission/solution/fused_norm.cuh`, in `fused_norm` and
`kSharedBytes`; keep `solution/kernel.cu` and `config.toml` unchanged.

1. Extend dynamic shared memory after the reduction scratch with one sum per
   row column. Define `summed` after `shared`, using the padded reduction offset.
   Increasing `kSharedBytes` updates both `config.dynamicSmemBytes` and
   `cudaFuncSetAttribute`; retain both uses and all launch dimensions.
2. In the first pass, retain input loads and accumulation. Add a `float4` scratch
   array, save each unrounded FP32 sum, and convert the residual output with
   `__float2bfloat16_rn`. Store residual output here and write the FP32 cache.
3. In the second pass, replace recomputation with cache reads. Remove its residual
   store. Keep weight loads, normalization arithmetic, and output stores.

Each CTA owns row `blockIdx.x`. Thread
`threadIdx.x + threadIdx.y * kWarp` owns `kVector` consecutive columns per round,
starting at `(round * threads + thread) * kVector`. Store column `c` at
`summed[c]`; the same thread reloads it. Reduction scratch and row cache never
overlap. The cache uses aligned `float4` accesses, each moving four FP32 values.
The reduction offset is rounded to `kFloat4` elements; column offsets are also
multiples of that group, preserving 16-byte alignment.

Retain `column < width` guards and zero-initialized invalid lanes. This replay's
width is divisible by `kVector`, so every admitted group is complete. An adapted
partial group needs element guards and a scalar tail instead of an out-of-bounds
vector access; preserve zero contribution from invalid columns.

Keep both existing `__syncthreads()` calls. The first follows all cache writes;
the second follows the row reduction. Thus all cache data is ready before the
output pass. Each column has one owner, no thread needs another owner's original
residual, and the cache remains allocated without reuse until CTA completion.
Earlier residual stores are therefore safe. Preserve caller-stream copies,
`griddepcontrol.wait`, and `griddepcontrol.launch_dependents` so upstream data is
ready before reading and downstream consumers retain the existing ordering.

Cache the FP32 add result, never the rounded BF16 residual. Preserve the order of
per-thread square accumulation, both butterfly reductions, division and epsilon,
`rsqrt.approx.ftz.f32`, scale multiplication, and final BF16 rounding.

## Example configuration

Retain `kTokens = 8192`, `kHidden = 7168`, `kVector = 8`, `kWarp = 32`,
`kWarps = 28`, `kFloat4 = 4`, and `kReduction = 28`. Launch 8192 CTAs with
`block = (32, 28)`: 896 threads, eight elements each, one round per row.
The reduction region stays 112 bytes; adding the 28672-byte FP32 row makes
`kSharedBytes = 28784` instead of 112. Use two `float4` values per thread.

Activations and residuals are contiguous BF16 `[8192, 7168]` arrays with element
strides `(7168, 1)` and row stride 14336 bytes. Weight is contiguous BF16 `[7168]`.
The wrapper requires 16-byte tensor alignment. Inputs are copied into distinct
output buffers before in-place computation. Retain both device copies, the
pybind11 destination-passing ABI, device guard, error checks, and caller stream.

Keep FP32 addition/accumulation, BF16 round-to-nearest-even outputs,
`kEpsilon = 1.0e-6f`, `kWeightBias = 0.0f`, scalar BF16 global accesses,
unroll directives, and the existing two-level warp-shuffle reduction.
Retain programmatic stream serialization with `kPdlEnabled = 1` and the
SM90-guarded dependency instructions. These are retained mechanisms, not
requirements of caching. Keep the complete problem and its numerical policy.
Check correctness and latency with the registered Kernel evaluator; keep evidence
outside this card.

# Precondition

- Data types: cached sums must retain the precision and representation required
  by normalization. Introducing a narrowing conversion can change the output;
  moving the already-computed value must not add rounding. The input dtype itself
  imposes no additional requirement on this cache.
- Layout: ownership must identify a producer and later consumer in the same CTA
  for every sum, because another CTA cannot address its shared storage. Vector
  transfers require contiguous, in-bounds groups at their natural alignment;
  violating either condition makes the grouped access invalid.
- Storage: first-pass sums exist as transient arithmetic values. Later consumers
  recompute them from operands, supplying repeated work that caching can replace.
  Without these reusable intermediates there is no recomputation to eliminate.
- Pipeline: execution must permit input producers to finish before reading, and
  sum producers to finish with visible results before consumption. It must permit
  retaining each cached slot until its last reader completes. Moving residual
  stores earlier must not overwrite operands another reader still needs. These
  dependencies prevent stale values and lost inputs; they do not require a
  particular number of stages or barriers.
- Hardware: CTA-local shared memory must hold the live reduction scratch and
  `width * sizeof(sum_element)` bytes simultaneously. Insufficient capacity makes
  this full-row cache incompatible with the retained reduction.

# Scope

- Cases: rmsnorm
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The fragments below occupy separate locations in `fused_norm.cuh`. First-pass
input arrays and guarded loads stay unchanged. Replace its accumulation block,
then replace the second pass's `sum_values` declaration and guarded loading block.
In the After first pass, declare `sums` alongside the existing input arrays;
place `sum_values` after the guarded input loads. All omitted reduction,
normalization, weight/output declarations, and output stores remain unchanged.

## Before

```cuda
// File scope.
constexpr uint32_t kSharedBytes = kReduction * sizeof(float);

// fused_norm shared-memory declaration.
extern __shared__ float shared[];

// First pass: after the existing guarded input/residual loads.
#pragma unroll
for (uint32_t j = 0; j < kVector; ++j) {
  float value = __bfloat162float(input_values[j]);
  value += __bfloat162float(residual_values[j]);
  sum_sq += value * value;
}

// Second pass: before the unchanged normalization loop.
float sum_values[kVector] = {};
if (column < width) {
#pragma unroll
  for (uint32_t j = 0; j < kVector; ++j) {
    weight_values[j] = weight[column + j];
    float value = __bfloat162float(input[row * input_stride + column + j]);
    value += __bfloat162float(residual[row * residual_stride + column + j]);
    residual[row * residual_stride + column + j] = __float2bfloat16_rn(value);
    sum_values[j] = value;
  }
}
```

## After

```cuda
// File scope: retain both existing launch uses of this constant.
constexpr uint32_t kSharedBytes = (kReduction + kHidden) * sizeof(float);

// fused_norm: reduction scratch precedes the row cache.
extern __shared__ float shared[];
float* summed = shared + ((warps + kFloat4 - 1) / kFloat4) * kFloat4;

// First pass: declare beside input_values and residual_values.
float4 sums[kVector / kFloat4];
// The existing guarded input/residual loads remain here.
auto* sum_values = reinterpret_cast<float*>(sums);
#pragma unroll
for (uint32_t j = 0; j < kVector; ++j) {
  float value = __bfloat162float(input_values[j]);
  value += __bfloat162float(residual_values[j]);
  sum_sq += value * value;
  residual_values[j] = __float2bfloat16_rn(value);
  sum_values[j] = value;
}
if (column < width) {
#pragma unroll
  for (uint32_t j = 0; j < kVector; ++j) {
    residual[row * residual_stride + column + j] = residual_values[j];
  }
#pragma unroll
  for (uint32_t j = 0; j < kVector / kFloat4; ++j) {
    reinterpret_cast<float4*>(summed + column)[j] = sums[j];
  }
}

// Second pass: before the unchanged normalization loop.
float4 sums[kVector / kFloat4] = {};
if (column < width) {
#pragma unroll
  for (uint32_t j = 0; j < kVector; ++j) {
    weight_values[j] = weight[column + j];
  }
#pragma unroll
  for (uint32_t j = 0; j < kVector / kFloat4; ++j) {
    sums[j] = reinterpret_cast<const float4*>(summed + column)[j];
  }
}
const auto* sum_values = reinterpret_cast<const float*>(sums);
```

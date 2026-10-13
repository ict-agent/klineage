---
skill_id: rmsnorm.cta-row-cooperation
intent: Distribute each RMSNorm row across a CTA for cooperative reduction and normalization.
preconditions:
- 'Data types: No additional input dtype restriction; preserve partial-sum representation
  and the required arithmetic tree because distributing additions must not change
  rounding.'
- 'Layout: Rows have known strides and disjoint output regions; row elements must
  admit a unique participant assignment preserving reduction groups, or in-place writes
  race and sums change.'
- 'Storage: Row operands reside in device storage visible to every prospective CTA
  participant; unpublished thread-private sources would be unavailable to reassigned
  readers.'
- 'Pipeline: Source producers complete and their writes are visible before consumption.
  Every participating warp and CTA can reach the required barriers before partial-sum
  reads, scratch reuse, and output overwrites; otherwise consumers can read incomplete
  or overwritten values.'
- 'Hardware: CUDA warp and CTA barriers with full 32-lane warps. For G row groups
  and W warp lanes, G <= W and G*W must fit the CTA thread limit; shared capacity
  must cover (P + G*W)*sizeof(partial), with P >= G, for disjoint reduction and exchange
  storage.'
scope:
  cases:
  - rmsnorm
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Parallelize each row's residual addition, squared-sum reduction, and normalization
across one CTA. Apply this in `solution/fused_norm.cuh`; retain `sum_squares`,
the output-pass body, `solution/kernel.cu`, and `config.toml`.

Delete serial `row_sum` and replace `kRowThreads` with the shared-storage constants
below. Add `shared_xor`. Replace the beginning of `fused_norm` through
`mean_square` with the After prefix. Change the launch as shown.

One CTA owns a row. Thread `(lane, warp)` owns reduction group
`thread = lane + warp*kWarp` and columns
`(round*threads + thread)*kVector + j`. Reusing the output loop with these new
`thread`, `threads`, and `rounds` values distributes its scalar loads and stores
without changing arithmetic. Each column has one writer.

Operands start in device-global memory; `row_sum` uses thread-private arrays.
Move serial partials into per-thread accumulators. Exchange them through shared
memory in descending XOR steps, then reduce the warp totals with warp zero.
`shared[0:kReduction]` holds totals; the following `kWarps*kWarp` elements hold
lane-exchange scratch. Each warp owns a distinct scratch slice. Both warp
barriers are necessary: one publishes values, the other completes partner reads
before scratch reuse. The first CTA barrier publishes warp totals and completes
the input-reading pass. The second publishes the final sum before normalization.
All threads execute the first reduction; only the complete first warp executes
the second. Keep both CTA barriers outside that conditional.

Retain scalar global accesses, unrounded FP32 residual sums, the per-group
multiply-add sequence, descending reduction offsets, zero padding of unused
warp totals, division followed by epsilon, the existing reciprocal-square-root
instruction, multiplication order, and round-to-nearest BF16 stores. Do not reload
the rounded residual for normalization. Keep the two asynchronous device copies
and every operation on the caller stream. Validate the unchanged problem with
`klineage.harness.evaluate`; preserve its tolerances and timing policy.

## Example configuration

This replay uses 8192 rows of width 7168, BF16 inputs/outputs, FP32 intermediates,
`kVector=8`, `kWarp=32`, and `kWarps=28`. Row strides are 7168 elements
(14336 bytes); weights have unit stride. The wrapper retains its 16-byte alignment
checks, although this transformation uses scalar loads.

The serial launch is 64 CTAs of 128 independent row threads. Restore 8192 CTAs of
`dim3(32,28)` (896 threads), one round per thread. Allocate 924 FP32 shared elements
(3696 bytes): 28 totals and 896 exchange values. Preserve `kFloat4=4` and its
round-up formula as the original allocation convention; four-element rounding is
not required by the scalar exchange itself. No fixed SM placement is used.

The exact width is divisible by eight, so the retained `column < width` guards
cover complete groups. A different partial-group width needs per-element guards
and zero-filled reduction leaves. `gridDim.x=kTokens` gives every CTA a valid row;
remove the serial per-thread row return so it cannot strand barrier participants.
Keep unsigned 32-bit indexing, epsilon `1.0e-6f`, weight bias `0.0f`, and the
unchanged `rsqrt.approx.ftz.f32` instruction. The numerical grouping constants
remain in the serial bundle to preserve its addition tree.

# Precondition

- Data types: No additional input dtype restriction. Cooperation changes owners,
  not representations. Preserve the partial-sum type, per-group operations, and
  required reduction tree; extra casts or reassociation can alter rounding.
- Layout: Row strides must be known and output rows must not overlap. Assign each
  row element to one participant while preserving its reduction group; duplicate
  or misplaced owners would race on in-place stores or change the squared sum.
- Storage: Row operands must reside in device storage visible to all prospective
  CTA participants. Unpublished values in the original thread's private storage
  would be unavailable to reassigned readers.
- Pipeline: Source producers must complete and their writes must be visible before
  consumption. All participants must be able to reach their warp
  and CTA barriers. Publication must precede partial-sum reads, partner reads must
  finish before scratch reuse, and the input reduction must finish before output
  overwrites. Violating these orders exposes incomplete or overwritten values.
- Hardware: This exchange uses CUDA warp and CTA barriers with complete 32-lane
  warps. With `G` row groups and `W` lanes, `G <= W` lets one warp read every group
  total, and `G*W` must fit a CTA. Shared capacity must be at least
  `(P + G*W)*sizeof(partial)`, where `P >= G`; totals and every warp's exchange
  slice must coexist without overlap.

# Scope

- Cases: rmsnorm
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The snippets replace the namespace allocation constants, the kernel prefix ending
at `mean_square`, and the launch setup. Delete `row_sum`; keep `sum_squares`
unchanged. The output loop below `inverse_rms` remains unchanged and uses the
prefix's ownership variables.

## Before

```cuda
constexpr uint32_t kRowThreads = 128;
```

```cuda
const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
if (row >= kTokens) return;

const uint32_t thread = 0;
const uint32_t threads = 1;
const uint32_t span = kVector * threads;
const uint32_t rounds = (width + span - 1) / span;

const float sum_sq = row_sum(input, residual, row, width, input_stride, residual_stride);
const float mean_square = sum_sq / float(width) + epsilon;
```

```cuda
cudaLaunchConfig_t config{};
config.gridDim = dim3((kTokens + kRowThreads - 1) / kRowThreads);
config.blockDim = dim3(kRowThreads);
config.stream = stream;
```

## After

```cuda
constexpr uint32_t kFloat4 = 4;
constexpr uint32_t kReduction = ((kWarps + kFloat4 - 1) / kFloat4) * kFloat4;
constexpr uint32_t kExchange = kWarps * kWarp;
constexpr uint32_t kSharedBytes = (kReduction + kExchange) * sizeof(float);

__device__ __forceinline__ float shared_xor(float value, int mask, float* scratch) {
  const uint32_t thread = threadIdx.x + threadIdx.y * kWarp;

  // Publish each lane's value before its partner reads it.
  scratch[thread] = value;
  __syncwarp();
  float result = scratch[thread ^ mask];

  // Finish reads before the next butterfly step reuses scratch.
  __syncwarp();
  return result;
}
```

```cuda
const uint32_t row = blockIdx.x;
const uint32_t lane = threadIdx.x;
const uint32_t warp = threadIdx.y;
const uint32_t warps = blockDim.y;
const uint32_t thread = lane + warp * kWarp;
const uint32_t threads = warps * kWarp;
const uint32_t span = kVector * threads;
const uint32_t rounds = (width + span - 1) / span;
extern __shared__ float shared[];

float sum_sq = sum_squares(input, residual, row, width, input_stride,
                           residual_stride, thread, threads);

#pragma unroll
for (uint32_t offset = kWarp / 2; offset > 0; offset /= 2) {
  sum_sq += shared_xor(sum_sq, offset, shared + kReduction);
}
shared[warp] = sum_sq;
__syncthreads();

if (warp == 0) {
  sum_sq = lane < warps ? shared[lane] : 0.0f;
#pragma unroll
  for (uint32_t offset = kWarp / 2; offset > 0; offset /= 2) {
    sum_sq += shared_xor(sum_sq, offset, shared + kReduction);
  }
  shared[0] = sum_sq;
}
__syncthreads();

const float mean_square = shared[0] / float(width) + epsilon;
```

```cuda
cudaLaunchConfig_t config{};
config.gridDim = dim3(kTokens);
config.blockDim = dim3(kWarp, kWarps);
config.dynamicSmemBytes = kSharedBytes;
config.stream = stream;

status = cudaFuncSetAttribute(fused_norm, cudaFuncAttributeMaxDynamicSharedMemorySize,
                              kSharedBytes);
if (status != cudaSuccess) return status;
```

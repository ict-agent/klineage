---
skill_id: rmsnorm.warp-shuffle-exchange
intent: Replace shared-memory butterfly exchanges with warp register shuffles.
preconditions:
- 'Data types: each exchanged value must fit one 32-bit shuffle operand without changing
  its bits; the exchange must leave arithmetic and rounding unchanged.'
- 'Layout: each XOR partner must be an active lane in the same hardware warp; every
  lane named by the shuffle member mask must execute the matching exchange. No additional
  tensor contiguity or alignment is needed because this exchange addresses registers.'
- 'Storage: each producer retains its value in a register, and the shared scratch
  being removed serves only this warp-local exchange; other shared consumers must
  keep their storage.'
- 'Pipeline: producers finish before exchange, and consumers finish before their values
  are replaced; scratch barriers must serve only exchange visibility and reuse, while
  barriers publishing other data remain.'
- 'Hardware: CUDA must support synchronized 32-bit butterfly shuffles for the participating
  warp; no additional shared-memory capacity is required because exchange scratch
  is removed.'
scope:
  cases:
  - rmsnorm
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Replace shared-memory XOR exchanges with register shuffles in
`solution/fused_norm.cuh`. This removes one scratch write, one scratch read,
and two warp barriers per exchange. Keep the butterfly additions in their
existing order.

Apply these edits:

1. Delete `kExchange`; set `kSharedBytes = kReduction * sizeof(float)`.
2. Replace `shared_xor` with `shuffle_xor` below.
3. In both butterfly loops in `fused_norm`, replace
   `shared_xor(sum_sq, offset, shared + kReduction)` with
   `shuffle_xor(sum_sq, offset)`.

The first loop exchanges partial sums within every warp. After the first CTA
barrier, only warp zero executes the second loop. Both loops have full-warp
participation. XOR offsets select the same partner values as the shared version.
The shuffle's member mask names all 32 lanes; its clamp selects one whole warp.

Keep `shared[warp]`, `shared[0]`, both CTA barriers, and the shared prefix they
use. Only the appended exchange scratch and its two warp barriers disappear.
Each shuffle consumes the current register value; it neither publishes unrelated
shared data nor replaces either CTA barrier.

Keep grid/block dimensions and caller-stream ordering. The launch and
`cudaFuncSetAttribute` already use `kSharedBytes`; changing that constant updates
both shared-memory requests. No other host edits are needed.

Preserve input/output ownership, load guards, zero-filled inactive values,
increasing element order, butterfly order, FP32 sums, reciprocal square root,
and final BF16 round-to-nearest conversions. In particular, normalization uses
unrounded FP32 residual sums. No new bounds checks or early returns are needed.

## Example configuration

The supplied problem uses CUDA on `nvidia-sm90a-cuda13`: 8192 contiguous rows of
7168 BF16 elements and one contiguous BF16 weight vector. Row strides are
7168 elements; the wrapper requires 16-byte alignment. Each CTA owns one row.
The launch is `grid=(8192)` and `block=(32,28)`: 896 threads, with thread
`lane + 32*warp` owning eight consecutive columns starting at `8*thread`.
There is one load round for this workload.

The reduction runs offsets `16,8,4,2,1` twice: first within all 28 warps, then
within warp zero over 28 partial sums plus four zeros. Preserve all additions
and the existing two CTA barriers. `kReduction=28` reserves 112 shared bytes.
The preceding shared exchange appends 896 floats (3584 bytes), bringing the
allocation to 3696 bytes. The forward change restores 112 bytes.

Retain the two device-to-device input copies, two-pass sum recomputation,
eight-element thread work, loop unrolling, in-place device computation,
`epsilon=1e-6`, zero weight bias, `rsqrt.approx.ftz.f32`, and programmatic
stream serialization with its guarded `griddepcontrol` instructions.
Keep `config.toml` and `solution/kernel.cu` unchanged.

# Precondition

- Data types: each exchanged value must fit a 32-bit shuffle operand losslessly.
  The instruction moves bits; it does not require the tensor inputs to have any
  particular dtype. Keep all arithmetic and rounding outside the exchange
  unchanged, since reordering additions could change the operator result.
- Layout: the XOR source must be an active lane in the same hardware warp.
  A shuffle cannot fetch another warp's value. Every lane named by its member
  mask must execute the matching instruction; the shown full mask requires all
  32 lanes. The existing scratch indexing must identify those same partners.
  No additional tensor contiguity or alignment is needed: the exchange
  addresses registers, not tensor storage.
- Storage: producers must still hold the exchanged values in registers.
  Remove only scratch used exclusively for these exchanges; another shared
  reader would otherwise lose its input. Shared storage used by other
  communication remains allocated and visible.
- Pipeline: produce each register value before exchanging it and consume it
  before replacement. The removed warp barriers must protect only scratch
  publication and read completion before scratch reuse. Preserve barriers
  that publish other data;
  a shuffle supplies no shared-memory visibility guarantee for those accesses.
- Hardware: the target must implement synchronized 32-bit butterfly shuffles
  across the participating warp. Otherwise the replacement instruction is
  unavailable. No additional shared-memory capacity is needed; this change
  deletes the per-thread exchange allocation.

# Scope

- Cases: rmsnorm
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

These fragments are in `solution/fused_norm.cuh`. The final statement occurs
inside both existing butterfly loops; retain their surrounding control flow.

```cuda
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

// At both reduction sites in fused_norm:
sum_sq += shared_xor(sum_sq, offset, shared + kReduction);
```

## After

```cuda
constexpr uint32_t kSharedBytes = kReduction * sizeof(float);

__device__ __forceinline__ float shuffle_xor(float value, int mask) {
  float result;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
               : "=f"(result) : "f"(value), "r"(mask));
  return result;
}

// At both reduction sites in fused_norm:
sum_sq += shuffle_xor(sum_sq, offset);
```

---
skill_id: cuda.warp-register-matrix-transpose
intent: Transpose warp-distributed matrix fragments directly in registers.
preconditions:
- 'Data types: elements are 16-bit payloads packed two per 32-bit register; the b16
  transpose moves bits without arithmetic or rounding requirements.'
- 'Layout: each complete warp owns an 8x8 matrix; lane l holds row l/4, columns 2*(l%4)
  and 2*(l%4)+1, and consumers expect that same packing of the transposed matrix.
  A different fragment mapping would permute the wrong elements; global contiguity
  and address alignment add no requirement.'
- 'Storage: source and result fragments are register-resident; the shared exchange
  is private transpose scratch with no other consumers, so removing its stores cannot
  erase externally visible data.'
- 'Pipeline: all 32 lanes execute each transpose together with ready source registers,
  as required by sync.aligned. Existing producer visibility and consumer completion
  before underlying buffer reuse must remain; only barriers protecting the private
  transpose scratch may disappear.'
- 'Hardware: movmatrix requires sm_75 or newer and a toolchain accepting PTX 7.8 or
  newer. No additional shared-memory capacity is needed; the existing source/result
  register words suffice.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Replace the shared-memory permutation in `solution/native.cuh::transpose(Reg)`
with one warp-register `movmatrix` transpose per fragment word. Replace the whole
helper with After; its local constants, scratch, gathers, and two barriers per
word disappear. No other source edit or launch change is needed.

For lane `l`, let `r=l/4` and `c=2*(l%4)`. Each input word holds
`[A[r,c], A[r,c+1]]`, low half first. The result must hold
`[A[c,r], A[c+1,r]]`. The Before gather reads packed words from lanes
`4*c+r/2` and `4*(c+1)+r/2`, selecting half `r%2`. `movmatrix` performs
this same bit permutation. It neither converts BF16 nor changes accumulation.
The intrinsic shape, participation, and target rules follow
[NVIDIA's PTX specification](https://docs.nvidia.com/cuda/archive/12.6.3/parallel-thread-execution/index.html#warp-level-matrix-instructions-movmatrix).

Keep every caller: `prepare.cuh::invert` transposes `m` and `p`, and
`recurrence.cuh::recur_tile` transposes each quantized residual. The former runs
in the complete first warp; the latter runs in complete compute warps. Preserve
all surrounding shared-memory barriers and caller-stream launch ordering.
`movmatrix.sync` coordinates its register exchange; it does not publish unrelated
shared-memory writes. Removing private scratch also removes its reuse hazard.

Retain the matrix load/store adapters, shared layouts, numerical conversions,
MMA instruction order, reduction grouping, and all bounds and zero-fill checks.
The helper has no partial-tile path: each call covers a full warp-owned fragment.
Rebuild from the complete final bundle with its existing compile flags and use
the Kernel evaluator with the unchanged problem.

## Example configuration

This replay uses CUDA on `nvidia-sm90a-cuda13`, with BATCH=1, TOKENS=4096,
HEADS=96, HEAD_DIM=128, and 16-token chunks (256 chunks per head).
`Reg` contains four 32-bit words, each a separate warp-wide 8x8 BF16 fragment;
`Acc` holds eight FP32 values. Preserve all four word transposes and existing
BF16 rounding boundaries, FP32 arithmetic, and fast-math settings.

Keep `prepare` at grid `(256,96)`, 256 threads, and launch bounds `(256,8)`.
Keep `recurrence` at grid `(1,96)`, 192 threads, with four compute warps owning
32 value channels each. Preserve the sequential chunk loop, single input/output
buffers, register prefetches, `mma.sync.m16n8k16`, and `ldmatrix`/`stmatrix`.
Retain the beta conversion kernel and stream-specific workspace allocation.

Shared matrices retain eight-column slabs:
`offset<Rows>(r,c)=r*8+(c%8)+(c/8)*Rows*8`.
Dynamic shared sizes stay 42368 bytes for PrepareShared and 124672 bytes for
RecurShared; InputShared remains 18048 bytes. The Before helper reserves
`ceil(max(kPrepareThreads,kRecurThreads)/kWarp)*kWarp*sizeof(uint32_t)`
static scratch bytes: 1024 here. After removes only this reservation; preserve
the dynamic shared-memory attributes and launch byte counts.

# Precondition

- Data types: the instruction permutes 16-bit payloads packed into 32-bit words.
  Other element widths need another instruction or packing. BF16 is incidental;
  no floating-point interpretation, arithmetic order, or rounding is introduced.
- Layout: a complete warp owns each 8x8 fragment in the stated lane/halfword
  mapping, and its consumer expects the transposed mapping. Otherwise the fixed
  register routing selects different matrix entries. No global stride,
  contiguity, or address-alignment constraint is added by a register-only move.
- Storage: inputs and results are register fragments, with shared memory serving
  only as private transpose scratch. Another reader of those scratch stores
  would lose data if they were removed.
- Pipeline: all 32 lanes must reach the same instruction with completed source
  values; divergent participation violates `sync.aligned`. Preserve barriers
  that make underlying producers visible and finish consumers before their
  buffers are reused. Only scratch publication/reuse barriers are redundant
  after its private exchange is replaced; no stage-count requirement is added.
- Hardware: the target must support `movmatrix` (sm_75 or newer), and its
  assembler must accept PTX 7.8 or newer. Otherwise the instruction is unavailable.
  No extra shared-memory capacity or register storage beyond the existing
  source/result words is required.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The snippets replace only `transpose(Reg)`; retain `Reg`, `kWarp`, and all callers.

## Before

```cuda
__device__ __forceinline__ Reg transpose(Reg a) {
    constexpr int kElemBits = 16;
    constexpr int kPairElems = sizeof(uint32_t) / sizeof(uint16_t);
    constexpr int kMatrixSide = 8;
    constexpr int kRowLanes = kMatrixSide / kPairElems;
    constexpr int kMaxThreads = kPrepareThreads > kRecurThreads
        ? kPrepareThreads : kRecurThreads;
    constexpr int kScratchWarps = (kMaxThreads + kWarp - 1) / kWarp;
    constexpr uint32_t kElemMask = (uint32_t(1) << kElemBits) - 1;
    __shared__ uint32_t scratch[kScratchWarps][kWarp];
    const int warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
    const int row = lane / kRowLanes;
    const int col = lane % kRowLanes * kPairElems;
    const int shift = row % kPairElems * kElemBits;
    Reg b;

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        // Publish the row fragments before gathering the transposed pair.
        scratch[warp][lane] = a.x[j];
        __syncwarp(kAllLanes);
        const uint32_t lo = scratch[warp][col * kRowLanes + row / kPairElems];
        const uint32_t hi = scratch[warp][(col + 1) * kRowLanes + row / kPairElems];
        b.x[j] = ((lo >> shift) & kElemMask)
            | (((hi >> shift) & kElemMask) << kElemBits);

        // Finish every read before another fragment or call reuses scratch.
        __syncwarp(kAllLanes);
    }
    return b;
}
```

## After

```cuda
__device__ __forceinline__ Reg transpose(Reg a) {
    Reg b;
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(b.x[j]) : "r"(a.x[j]));
    return b;
}
```

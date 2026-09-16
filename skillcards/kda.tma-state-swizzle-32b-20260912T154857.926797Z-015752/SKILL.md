---
skill_id: kda.tma-state-swizzle-32b
intent: Permute shared state transfers with TMA 32-byte bank swizzling.
preconditions:
- 'Data types: the tensor-map type must support 32-byte swizzling and match the stored
  representation; the permutation moves intact 16-byte units, so it adds no arithmetic
  or numerical restriction.'
- 'Layout: known tensor strides and complete logical ownership must let every shared
  reader and writer use the same permutation. The noninterleaved box inner extent
  times element size must not exceed 32 bytes; TMA global and shared bases require
  128-byte alignment, and shared-base swizzle phase must be accounted for to avoid
  wrong addresses.'
- 'Storage: state already passes between global memory and an unswizzled CTA-shared
  buffer accessible to its converters; its allocation must contain the entire permuted
  footprint, including any padding, or addresses may escape the buffer.'
- 'Pipeline: input producers and TMA loads must finish before conversion reads; conversion
  writes must become visible before TMA stores. All participating converters must
  synchronize, and every reader must finish before overlapping storage is reused;
  otherwise the permutation exposes incomplete or overwritten data.'
- 'Hardware: NVIDIA TMA with 32-byte tensor-map swizzling and its synchronization
  support is required; available CTA-shared capacity must cover the maximum simultaneously
  live buffer footprint. The full-width permutation adds no storage.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Apply TMA's 32-byte bank permutation to the state transfer buffer. TMA moves
16-byte units to permuted shared addresses; the conversion threads use matching
addresses. Logical values, arithmetic, and global layouts stay unchanged.

Make both edits in the After snippet together:

- In `solution/native.cu`, `encode_map`, set `MapKind::State` to
  `CU_TENSOR_MAP_SWIZZLE_32B`. Both `recur.state_in` and `recur.state_out` use this
  branch. Keep the default swizzle for every other map.
- In `solution/native.cuh`, replace `fp32_offset`. Its callers in
  `solution/recurrence.cuh`, `state_in` and `state_out`, already route every
  conversion access through it. Keep the BF16 `offset<kDim>` layout unchanged.

For slab width `W`, rows `R`, and element bytes `e`, the preceding layout is
`p = row*W + col%W + (col/W)*R*W`. The full-width byte layout has `W*e = 32`.
For a zero-phase slab base, apply `a' = a ^ ((a & 128) >> 3)` to `a = e*p`:
byte-address bit 7 toggles bit 4. This swaps intact 16-byte units within each
32-byte span. Divide by `e` to index elements. The snippet specializes this to
FP32. A slab base on a 256-byte boundary has zero phase; otherwise include
`(shared_base / 128) % 2` in the source-bit parity. Smaller boxes need the
hardware's padded shared layout. [PTX swizzling rules](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#tensor-swizzling-modes)

Ownership remains in `state_in` and `state_out`: each warp walks its assigned
8-by-8 blocks; each lane converts two adjacent columns. TMA writes the permuted
FP32 buffer before `state_in` reads it. Conversely, `state_out` writes that
permutation before TMA reads it. No MMA fragment directly accesses this buffer.

Keep ordering in `recurrence`: `state_ready` tracks the full input transfer;
`wait_bar`, the proxy fence, and CTA participation precede conversion.
The CTA barriers after `state_in` protect its reads before the union becomes
pipeline storage. Drain the input/output pipeline and reach the final CTA barrier
before `state_out` overwrites that union. Preserve its proxy fence and CTA barrier
before the elected store thread issues the final transfer. The terminal state
buffer is never overwritten again in that invocation; retain the store commit,
terminal barrier, and caller-stream completion boundary. A version that reuses
that buffer in the same invocation must wait for the TMA source reads first.

No launch, allocation, bounds, or arithmetic changes are needed. Preserve input
normalization, gate functions, reduction/MMA order, every BF16 rounding point,
FP32 accumulation, and final conversion. Rebuild from the complete bundle and
check with `klineage.harness.evaluate` using the unchanged problem.

## Example configuration

Preserve these replay settings; they are not technique prerequisites:

- Workload: `BATCH=1`, `TOKENS=4096`, `HEADS=96`, `HEAD_DIM=128`;
  `kChunk=16`, `kTiles=256`. All state slabs and token chunks are complete.
- Global states are contiguous `[batch,head,value,key]` FP32. State maps have
  shape `{128,128,96}`, byte strides `{512,65536}`, box `{8,128,1}`,
  unit element strides, no interleave, and unchanged bounds-fill/L2 settings.
- `s.fp32` has 16384 FP32 elements, starts 32768 bytes after `raw`, and uses
  4096-byte slabs. Retain the shared declarations and build settings; this
  layout has zero swizzle phase. Do not infer zero phase from `alignas(128)`
  alone when adapting the layout.
- Recurrence uses grid `(1,96)`, 192 threads, four compute warps, one load warp,
  one store warp, three input stages, and two output stages. Shared bytes:
  `RecurShared=98432`, `InputShared=18048`; the pipeline union overlaps FP32
  conversion storage only across separated lifetimes.
- Preparation uses grid `(256,96)`, 256 threads, and 21248 shared bytes.
  Retain beta preprocessing, shared operand reuse, register tiling/prefetch,
  BF16 `mma.sync.m16n8k16`, and its matrix load/store/transpose adapters.
- Retain `config.toml`, pybind11 destination-passing ABI, device checks,
  caller stream, workspace cache, `sm_90a` target, and serialized compiler
  flags, including fast math and register-usage level 10. Scale, bounded gate,
  oracle, tolerances, workload files, seed, and timing policy remain unchanged.

# Precondition

- Data types: the encoded type must permit 32-byte swizzling and describe the
  actual stored bits. Otherwise TMA cannot encode or interpret the transfer.
  There is no additional arithmetic or numerical requirement: the permutation
  keeps each 16-byte unit intact. FP32 conversion storage and BF16 computation
  are example settings, not prerequisites for moving the bits.
- Layout: the tensor strides and logical ownership must be known so all shared
  writers/readers can address the same elements after permutation. A missed
  reader or writer accesses the wrong value. For the noninterleaved descriptor,
  `box_inner_extent * element_bytes <= 32` is the encoding limit. Global and
  shared bases must meet 128-byte TMA alignment. Shared-base phase must be
  included when nonzero; otherwise software and TMA choose different addresses.
  [CUDA TMA alignment and box rules](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#tma-swizzle)
- Storage: an unswizzled CTA-shared state buffer already connects global
  transfers and conversion threads. Those threads must be able to access the
  buffer because their addresses change with the transfer layout. Allocation
  must cover every permuted address, including padding for narrower boxes;
  otherwise a transfer or conversion escapes its storage.
- Pipeline: global input production and TMA load completion must precede shared
  conversion reads. Conversion writes must be visible to TMA before output
  transfer. Every converter must participate in the applicable synchronization.
  All readers must finish before any overlapping allocation is reused. These
  dependencies prevent stale reads and overwrite races; no particular pipeline
  depth is required.
- Hardware: the device and encoder must support NVIDIA TMA's 32-byte swizzle
  and its completion/visibility synchronization. Without them the descriptor
  or ordered transfer is unavailable. CTA-shared capacity must cover the
  maximum simultaneous buffer footprint, including required padding. Permuting
  full-width spans consumes no additional bytes.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Both snippets use the existing `offset`, tensor-map defaults, and `kDim`.
The State switch branch configures both input and output state descriptors.
The conversion callers and all other code remain unchanged.

## Before

```cuda
// solution/native.cu: encode_map switch
case MapKind::State:
    type = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    shape[1] = kDim; strides[0] = kDim*sizeof(float); strides[1] = kStateElems*sizeof(float);
    box[1] = kDim;
    break;

// solution/native.cuh
__device__ __forceinline__ int fp32_offset(int row, int col) {
    // Match the unswizzled state tensor maps.
    return offset<kDim>(row, col);
}
```

## After

```cuda
// solution/native.cu: encode_map switch
case MapKind::State:
    type = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    shape[1] = kDim; strides[0] = kDim*sizeof(float); strides[1] = kStateElems*sizeof(float);
    box[1] = kDim; swizzle = CU_TENSOR_MAP_SWIZZLE_32B;
    break;

// solution/native.cuh
__device__ __forceinline__ int fp32_offset(int row, int col) {
    constexpr int kSwizzleBit = 32;
    constexpr int kSwizzleShift = 3;
    const int p = offset<kDim>(row, col);

    // Toggle byte-address bit 4 from bit 7; indices count FP32 elements.
    return p ^ ((p & kSwizzleBit) >> kSwizzleShift);
}
```

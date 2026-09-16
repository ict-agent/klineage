---
skill_id: cuda.shared-column-slab-layout
intent: Reduce shared-memory bank contention with column-slab matrix storage.
preconditions:
- 'Data types: no additional arithmetic or dtype requirement; permute element addresses
  without converting stored bits, preserving each access''s element alignment.'
- 'Layout: known rectangular matrices and lane ownership; choose a slab width S dividing
  the column extent C. Every writer and reader must use the same logical coordinates,
  and any contiguous access bypassing the mapper must stay within one slab to avoid
  crossing a physical discontinuity.'
- 'Storage: the affected matrices already occupy CTA-local shared memory, and their
  physical arrangement is internal; otherwise changing the layout would break an unadapted
  consumer.'
- 'Pipeline: producers must finish and publish writes before consumers read, and all
  readers must finish before storage reuse. Preserve the participating threads and
  synchronization establishing those dependencies.'
- 'Hardware: banked CUDA shared memory; choose S so the mapped addresses reduce bank
  contention for the target lane accesses. The permutation uses the existing R*C*sizeof(element)
  bytes per matrix, requiring no additional capacity.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Store shared matrices in column slabs to reduce bank contention when lanes access
several rows over a narrow column span. For a logical matrix with `R` rows and `C`
columns, map `(r,c)` to `(c/S)*R*S + r*S + c%S` instead of `r*C+c`.
This changes physical addresses only; each lane retains its logical elements.

In `solution/native.cuh`, replace the `offset<Rows, Cols>` implementation with the
After snippet. Keep both template parameters and every existing call unchanged.
`Cols` remains useful shape metadata although the slab expression uses `Rows`.
`fp32_offset` already delegates to this helper, so the same edit changes FP32
state staging. No launch, allocation, compiler-flag, or synchronization change is
needed. Update comments describing these affected matrices as row-major.

All affected producers and consumers already route through the helper:

- `prepare.cuh`: decay stores; inversion writes/reads; triangular masking;
  matrix-product stores; and `store_prepare` exports.
- `recurrence.cuh`: `load_input`, state import/export and conversion,
  correction publication/loads, state updates, and output export.
- `native.cuh`: `load_frag`, `load_a`, `load_b`, `load_t`, `store_c`, and
  `store_t` preserve logical fragment ownership while translating addresses.

Keep global tensors and workspace rows unchanged. Leave direct indexing of
normalization, gate-prefix, triangular FP32, reduction, and transpose scratch
arrays unchanged: those arrays do not use `offset`.

The compact formula covers `0 <= r < R`, `0 <= c < C` when `C % S == 0`.
Keep existing bounds and beta zero fill. In decay preparation, stores using
`dst+j` begin at an even column and span two elements, so they remain inside the
chosen slab. Scalar fragment loads/stores otherwise map each element separately;
no collective matrix instruction imposes a shared-layout constraint here.

Preserve barriers after cooperative loads, before dependent products, and before
chunk-buffer reuse. Preserve warp synchronization after correction publication
and during transpose/reduction scratch exchanges. Preserve lane participation
and the warp's value-tile ownership. The permutation adds no concurrent accesses
or buffers. Preserve all arithmetic, FP32 accumulation order, BF16 conversions,
rounding points, and scalar volatile load/store controls.

Rebuild from the edited complete bundle and check it with
`klineage.harness.evaluate` using its unchanged problem. Inspect generated shared
addresses to confirm the slab strides; keep validation evidence outside this card.

## Example configuration

Replay with `S = 8`. Preserve `BATCH=1`, `TOKENS=4096`, `HEADS=96`, `HEAD_DIM=128`,
`kChunk=16`, and 256 preparation/recurrence threads. Preparation launches
`grid=(kTiles,kHeads)`; recurrence launches `grid=(1,kHeads)` and processes
256 chunks in order. Each recurrence warp owns 16 value columns.

The affected logical arrays and existing template arguments are:

| Arrays | Shape | Mapper arguments |
| --- | --- | --- |
| `PrepareShared.kd/qd/ki/kr` | `[kChunk,kDim]` | `<kChunk>` |
| `PrepareShared.inv/mqk` | `[kChunk,kChunk]` | `<kChunk,kChunk>` |
| `InputShared.v/kd/qd/kr` | `[kChunk,kDim]` | `<kChunk>` |
| `InputShared.inv/mqk` | `[kChunk,kChunk]` | `<kChunk,kChunk>` |
| `RecurShared.state/fp32` | `[kDim,kDim]` | `<kDim>` |
| `RecurShared.correction/out` | `[kChunk,kDim]` | `<kChunk>` |

These matrices use BF16 except FP32 state staging. Default `Cols=kDim` remains.
Keep the 42368-byte `PrepareShared`, 18048-byte `InputShared`, and 124672-byte
`RecurShared` structures and their alignments. Retain static transpose scratch,
scalar shared transfers, BF16 MMA and its register fragments, separate matrix
products, chunk algebra, and the caller CUDA stream and tensor ABI. Those retained
features do not impose additional prerequisites on the address permutation.

# Precondition

- Data types: no additional arithmetic or dtype requirement. Move element
  addresses without conversion, preserve stored bits, and maintain element
  alignment; otherwise addressing would alter values or invalidate accesses.
- Layout: matrix extents and lane ownership must be known. The compact mapping
  requires slab width `S` to divide column extent `C`; otherwise the final slab
  needs padding beyond the compact allocation. Every writer and reader must
  agree on logical coordinates. Any contiguous access bypassing the mapper must
  remain within a slab, because crossing its end is physically discontinuous.
- Storage: the matrices already reside in CTA-local shared memory, with an
  internal physical arrangement. All consumers of that arrangement must be
  covered by the mapper; an unadapted consumer would read different elements.
- Pipeline: completed producer writes must be visible before consumption, and
  readers must finish before storage reuse. Retain the synchronization and
  participating threads establishing these dependencies; address permutation
  does not establish ordering or make overwrite races safe.
- Hardware: CUDA shared memory must be banked, and `S` must reduce contention
  for the relevant lane addresses; otherwise this transformation has no bank
  contention benefit. Each matrix retains its existing `R*C*sizeof(element)`
  allocation, so no additional capacity is required.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Replace only this helper. All existing callers, including `fp32_offset` and
explicit square-matrix column extents, remain valid.

## Before

```cuda
template<int Rows, int Cols = kDim>
__device__ __forceinline__ int offset(int row, int col) {
    return row * Cols + col;
}
```

## After

```cuda
template<int Rows, int Cols = kDim>
__device__ __forceinline__ int offset(int row, int col) {
    constexpr int kSlabCols = 8;
    static_assert(Cols % kSlabCols == 0);
    return (col / kSlabCols) * Rows * kSlabCols
        + row * kSlabCols + col % kSlabCols;
}
```

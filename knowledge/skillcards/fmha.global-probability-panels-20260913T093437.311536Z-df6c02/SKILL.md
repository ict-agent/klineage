---
skill_id: fmha.global-probability-panels
intent: Coalesce global probability accesses with column-panel storage.
preconditions:
- 'Data types: preserve each element bit-for-bit and keep address arithmetic representable;
  the permutation requires no particular floating-point format or arithmetic change.'
- 'Layout: known row/column ownership must let neighboring warp lanes access adjacent
  rows and short column groups, so panels consolidate their addresses. For this unpadded
  mapping, positive panel width P divides column extent C, and each existing adjacent-element
  store group stays inside a panel; every writer and reader must use the same bijection.
  Scalar accesses add no alignment requirement beyond element alignment.'
- 'Storage: the intermediate is a private, contiguous global tile whose physical layout
  is internal to the updated writer and readers; an unchanged consumer or exposed
  row-major view would interpret permuted elements incorrectly.'
- 'Pipeline: all readers of the preceding aliased contents finish before tile writes;
  probability producers finish and publish before consumers read; all probability
  readers finish before buffer reuse. Address changes do not replace these ordering
  guarantees.'
- 'Hardware: ordinary CUDA global loads/stores with warp address coalescing provide
  the benefit; no additional feature or capacity is needed because the permutation
  retains the existing R*C element allocation.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Arrange the global probability tile in column panels to coalesce accesses across
neighboring query rows. In `solution/attention.cuh`, change both `store_prob`
(the writer) and `prob_value` (the reader). Add `panel` in `solution/ops.cuh`
after `reduce`, before `scalar_half`. Update the writer's layout comment to
column-panel storage. No launch or allocation changes are needed.

For logical tile coordinates `(row, col)`, replace row-major `row*C+col` with
`(col/P)*R*P + row*P + col%P`: physical order is
`[column_panel][query_row][column_within_panel]`. This is a compact bijection
when `P` divides `C`. Preserve the two adjacent half stores; their logical pairs
remain within one panel. Both accesses must change together.

Each CTA owns its `SoftmaxState.prob` region, obtained through `prob_state` by
reusing a consumed score slab. Keep scale/statistic placement and the alias
unchanged. `store_prob` retains each lane's logical row/column ownership;
`prob_value` retains the corresponding row and increasing-key traversal.
The layout packs short column groups from successive rows together, improving
the opportunity to combine a warp's global memory requests.

Keep the CTA barrier before score storage is overwritten, warp barriers in row
reductions, and caller-stream ordering between the softmax and PV launches.
Their producer visibility and buffer-lifetime guarantees remain necessary.
Masked keys still produce zero probabilities, invalid V keys still load zero,
and query/output bounds remain unchanged. Write the complete physical tile,
including padded coordinates. Move FP16 bits only; preserve probability rounding,
FP32 FMA/reduction order, online rescaling, normalization, and output rounding.

## Example configuration

Preserve `kM=128`, `kN=176`, and `kInputPanelCols=8`: the probability tile is
`[22][128][8]` after optimization, occupying 45,056 bytes in the existing
47,104-byte `SoftmaxState` region. Its backing score region remains 90,112 bytes
per CTA per key tile. There are no new buffers or shared-memory allocations.

Keep 256 threads per CTA, 32 lanes per warp, four lanes per logical row,
two adjacent halves per lane, and two query rows per lane. For probability pair
`i`, the existing writer owns
`row=warp*16+lane/4+(i%2)*8` and
`col=(i/2)*8+(lane%4)*2`. Thus a fixed pair index spans eight rows of one
8-column panel per warp. Retain 88 score components, 44 packed probability pairs,
and 64 output components per thread. The PV reader's existing fragment mapping
already identifies the same logical rows; only its probability address changes.

The workload is packed NHD FP16 `[16384,64,128]`, with eight sequences; Q/K/V
row stride is 8192 half elements. Keep scalar FP32 QK/PV arithmetic, online
softmax, the existing reduction grouping, per-element conversion/reciprocal
controls, all-tile score masking, and separate scalar output stores.

Keep the SM90 target, original compiler flags, tensor ABI, caller stream, and
four launches: `attention_scores`, `attention`, `attention_values`,
`attention_epilogue`. Each launch uses `kMaxQTiles*kHeads=8640` CTAs,
256 threads, and zero dynamic shared memory; surplus CTAs retain their bounds
return. Keep the 94 score-slab pointers and existing scratch lifetime.

# Precondition

- Data types: this is a bit-preserving permutation, so no particular floating-point
  format is required. Do not alter conversions or arithmetic. Address products
  and offsets must remain representable; overflow breaks the index bijection.
- Layout: known writer/reader ownership must expose neighboring rows and short
  column groups within a warp; otherwise this panel arrangement does not
  consolidate their addresses. For the unpadded mapping, positive panel width
  `P` divides logical column extent `C`. Existing adjacent-element store groups
  must not cross panels, or their consecutive physical addresses would target
  the wrong row. All writers and readers must apply the same coordinate map.
  Scalar accesses need only the existing element alignment; no vector alignment
  is introduced.
- Storage: a contiguous global intermediate tile is private to the implementation,
  and every consumer of its layout can be updated. An externally required
  row-major view or an unchanged reader would see a permuted matrix.
- Pipeline: finish reads of the preceding aliased contents before overwriting the
  tile. Finish and publish probability writes before consumers access them, and
  finish those reads before reusing the allocation. These prevent overwritten,
  stale, or incomplete data; changing addresses supplies none of these guarantees.
- Hardware: ordinary CUDA global loads/stores and warp address coalescing suffice.
  Coalescing is the hardware mechanism that benefits from closer lane addresses.
  No additional feature or capacity is required: the existing allocation of
  `R*C*sizeof(element)` bytes is unchanged.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The writer snippets replace its index and stores inside the existing loop;
retain the row/column expressions. The reader snippets replace the FMA line
inside the existing increasing-key loop. Keep the surrounding value load,
accumulator, and bounds handling. Add the helper only in the After version.

## Before

```cuda
// solution/attention.cuh: store_prob, inside its existing loop.
const int index = row * kN + col;
prob_tile[index] = __ushort_as_half(uint16_t(prob[i]));
prob_tile[index + 1] = __ushort_as_half(uint16_t(prob[i] >> kHalfBits));

// solution/attention.cuh: prob_value, inside its existing key loop.
acc = fmaf(__half2float(prob_tile[row * kN + key]), value, acc);
```

## After

```cuda
// solution/ops.cuh: add after reduce, before scalar_half.
template<int Rows>
__device__ __forceinline__ int panel(int row, int col) {
    return (col / kInputPanelCols) * Rows * kInputPanelCols
           + row * kInputPanelCols + col % kInputPanelCols;
}

// solution/attention.cuh: store_prob, inside its existing loop.
const int index = (col / kInputPanelCols) * kM * kInputPanelCols
                  + row * kInputPanelCols + col % kInputPanelCols;
prob_tile[index] = __ushort_as_half(uint16_t(prob[i]));
prob_tile[index + 1] = __ushort_as_half(uint16_t(prob[i] >> kHalfBits));

// solution/attention.cuh: prob_value, inside its existing key loop.
acc = fmaf(__half2float(prob_tile[panel<kM>(row, key)]), value, acc);
```

---
skill_id: cuda.collective-shared-matrix-loads
intent: Load shared matrix fragments with warp-collective ldmatrix instructions.
preconditions:
- 'Data types: 16-bit payloads packed two per 32-bit fragment register; m8n8.b16 loads
  preserve bits without arithmetic or conversion, so no floating-point format is required.'
- 'Layout: each matrix row contains eight contiguous 16-bit elements starting at a
  16-byte-aligned shared address; known row addresses and consumer fragment ownership
  must match the instruction, or it fetches misaligned or incorrect elements.'
- 'Storage: complete source matrices already reside in CTA shared memory and remain
  valid throughout the load; the instruction cannot read these operands directly from
  global memory.'
- 'Pipeline: all 32 warp lanes execute the same collective load, after producers publish
  its rows and before any writer reuses their storage; divergent participation or
  unordered writes makes the collective invalid or its data unsafe.'
- 'Hardware: NVIDIA SM75 or newer with ldmatrix.m8n8.x4.b16 and its transposed form;
  this replacement uses the existing shared allocation and four result registers per
  lane, without additional shared capacity.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Replace scalar shared-memory fragment gathers with warp-collective
`ldmatrix.sync.aligned.x4.m8n8.shared.b16` loads, using `.trans` for
transposed fragments. Each lane receives the same four packed registers.
Only loading changes; preserve every payload bit and all arithmetic.

In `solution/native.cuh`, replace the contiguous block from `FragOrder`
through `load_t` with the After block. This removes both enums and
`load_frag`, adds `ld_matrix`/`ld_trans`, and restores the three adapters.
In `solution/recurrence.cuh`, replace only `load_correction` with its After
version. Retain `ld_scalar`: decay preparation still uses it. Retain
`st_scalar`, scalar matrix stores, and all other helpers.

## Ownership and ordering

Keep `offset<Rows>(r,c) = r*8 + (c&7) + (c/8)*Rows*8`.
Each register `j` contains two elements: `e=0` in its low half and `e=1`
in its high half. The four matrices have these physical origins, with
both matrix offsets multiplied by eight:

| Adapter | Base row, column | Matrix row, column | View |
| --- | --- | --- | --- |
| `load_a` | `row,col` | `j%2,j/2` | Normal |
| `load_b` | `row,col` | `j/2,j%2` | Normal |
| `load_t` | `col,row` | `j/2,j%2` | Transposed |
| `load_correction` | `0,value` | `j%2,j/2` | Transposed |

Within each matrix, normal fragments read `(lane/4,2*(lane%4)+e)`;
transposed fragments read `(2*(lane%4)+e,lane/4)`. The After adapters
supply matrix-row addresses through consecutive groups of eight lanes.
The instruction redistributes those rows into this exact ownership.

Keep the existing producer and reuse barriers. Preparation publishes its
slabs with CTA barriers; inversion publishes its warp's stores with
`__syncwarp`. Recurrence publishes state and input copies with CTA
barriers, and each compute warp publishes its correction tile with
`__syncwarp(kAllLanes)`. Keep `transpose` scratch barriers. State updates
and output stores finish before the existing CTA barriers permit the
next chunk to overwrite buffers. The collective load is not a substitute
for these memory-ordering dependencies.

Keep all calls warp-uniform: preparation uses complete selected warps,
inversion uses the complete first warp, and recurrence assigns a complete
warp to each value tile. These fixed tiles have no partial matrix loads.
For adapted boundaries, initialize every addressed matrix element before
the load; never predicate individual lanes around it. Keep current beta
zero fill and all host bounds checks.

## Example configuration

Preserve batch 1, 4096 tokens, 96 heads, dimension 128, chunks of 16, and
256 chunks per head. Both kernels launch 256 threads. Preparation uses
`grid=(256,96)`, `__launch_bounds__(256,8)`; recurrence uses `grid=(1,96)`,
`__launch_bounds__(256)`, with eight compute warps, each owning 16 value
columns. Retain all caller-stream launches and the destination-passing ABI.

`q`, `k`, `v`, `g`, and output retain contiguous BTHD layout; state remains value-key.
Shared arrays retain eight-column slabs: physical row segments are 16
bytes for BF16 and 32 bytes for FP32; successive slabs start `Rows*8`
elements apart. `PrepareShared`, `InputShared`, and `RecurShared` stay
42368, 18048, and 124672 bytes. Retain the 1024-byte static transpose
scratch and one serialized recurrence input/output buffer. Add no storage.

Retain BF16 operands/intermediates, FP32 accumulators, existing
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, quantization points,
normalization reduction grouping, block inversion, and scalar BF16 adds.
Keep the interleaved k/state and q/state products, shared correction
materialization, and state-update order. Retain scale, gate constants,
approximate exponentials/sigmoid, epsilon, and the complete problem oracle.

Keep serialized compile flags, including `-O3`, `--use_fast_math`,
`--ptxas-options=-v,--register-usage-level=10,--warn-on-spills`, and
`-lineinfo`, with the existing SM90a build target. Remove scalar-load
compiler control only by replacing the matrix gather call sites above;
do not change the retained volatile scalar helper or scalar-add loop.

Rebuild from the final complete bundle. Use `klineage.harness.evaluate`
with the unchanged problem for correctness and timing. Inspect generated
code for collective matrix loads at these sites; preserve MMA counts and
fragment bits. Keep measurements outside this card.

# Precondition

- Data types: the instruction moves 16-bit payloads into paired 32-bit
  fragment registers. It performs no conversion or arithmetic; BF16 is
  an example representation, not a prerequisite. Other payload widths
  need another instruction or packing adapter.
- Layout: each addressed matrix row must provide eight contiguous
  16-bit elements at a 16-byte-aligned shared address. Known row addresses
  and consumer fragment ownership must match the instruction's row-pair
  distribution, optionally transposed; otherwise accesses are misaligned
  or consumers receive different elements. Whole matrices need not be
  contiguous, and row strides need not match this example.
- Storage: all matrix rows must already be present in CTA shared memory
  and remain allocated during their loads. This instruction's shared
  address operand cannot replace a direct global-memory read.
- Pipeline: all 32 lanes must execute the same collective instruction.
  Producers must complete and make their stores visible before it reads;
  writers must wait for all readers before overwriting the rows. Preserve
  existing publication/reuse barriers: instruction rendezvous alone does
  not establish the required memory ordering. No stage count is required.
- Hardware: SM75 or newer must support the normal and transposed
  `ldmatrix.m8n8.x4.b16` forms. Four result registers per lane already
  exist in the scalar fragments. No additional shared capacity is needed;
  the preceding allocation is reused.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

`solution/native.cuh`, from the enums through `load_t`:

```cuda
enum class FragOrder { Column, Row };
enum class FragView { Normal, Transposed };

template<int Rows, FragOrder Order, FragView View>
__device__ __forceinline__ Reg load_frag(const BF16* p, int row, int col, int lane) {
    constexpr int kMatrixSide = 8;
    constexpr int kMatricesPerAxis = 2;
    constexpr int kPairElems = sizeof(uint32_t) / sizeof(BF16);
    constexpr int kRowLanes = kMatrixSide / kPairElems;
    constexpr int kFragments = sizeof(Reg) / sizeof(uint32_t);
    Reg a;

    // Gather each fragment's elements without collective matrix loads.
    #pragma unroll
    for (int j = 0; j < kFragments; ++j) {
        const int mr = Order == FragOrder::Column ? j % kMatricesPerAxis : j / kMatricesPerAxis;
        const int mc = Order == FragOrder::Column ? j / kMatricesPerAxis : j % kMatricesPerAxis;
        Pair pair;
        #pragma unroll
        for (int e = 0; e < kPairElems; ++e) {
            const int r = View == FragView::Normal ? lane / kRowLanes : lane % kRowLanes * kPairElems + e;
            const int c = View == FragView::Normal ? lane % kRowLanes * kPairElems + e : lane / kRowLanes;
            pair.h[e] = ld_scalar(p + offset<Rows>(row + mr * kMatrixSide + r, col + mc * kMatrixSide + c));
        }
        a.x[j] = pair.u;
    }
    return a;
}
template<int Rows> __device__ __forceinline__ Reg load_a(const BF16* p, int row, int col, int lane) {
    return load_frag<Rows, FragOrder::Column, FragView::Normal>(p,row,col,lane);
}
template<int Rows> __device__ __forceinline__ Reg load_b(const BF16* p, int row, int col, int lane) {
    return load_frag<Rows, FragOrder::Row, FragView::Normal>(p,row,col,lane);
}
template<int Rows> __device__ __forceinline__ Reg load_t(const BF16* p, int row, int col, int lane) {
    return load_frag<Rows, FragOrder::Row, FragView::Transposed>(p,col,row,lane);
}
```

`solution/recurrence.cuh`:

```cuda
__device__ __forceinline__ Reg load_correction(const RecurShared& s, int value, int lane) {
    return load_frag<kChunk, FragOrder::Column, FragView::Transposed>(s.correction,0,value,lane);
}
```

## After

Replace the same `solution/native.cuh` block:

```cuda
__device__ __forceinline__ Reg ld_matrix(const BF16* p) {
    Reg a;
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(a.x[0]), "=r"(a.x[1]), "=r"(a.x[2]), "=r"(a.x[3]) : "r"(shared_addr(p)));
    return a;
}
__device__ __forceinline__ Reg ld_trans(const BF16* p) {
    Reg a;
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(a.x[0]), "=r"(a.x[1]), "=r"(a.x[2]), "=r"(a.x[3]) : "r"(shared_addr(p)));
    return a;
}
template<int Rows> __device__ __forceinline__ Reg load_a(const BF16* p, int row, int col, int lane) {
    return ld_matrix(p + offset<Rows>(row + lane % 16, col + lane / 16 * 8));
}
template<int Rows> __device__ __forceinline__ Reg load_b(const BF16* p, int row, int col, int lane) {
    return ld_matrix(p + offset<Rows>(row + lane % 8 + lane / 16 * 8, col + (lane / 8 % 2) * 8));
}
template<int Rows> __device__ __forceinline__ Reg load_t(const BF16* p, int row, int col, int lane) {
    return ld_trans(p + offset<Rows>(col + lane % 8 + lane / 16 * 8, row + (lane / 8 % 2) * 8));
}
```

Replace only `load_correction` in `solution/recurrence.cuh`:

```cuda
__device__ __forceinline__ Reg load_correction(const RecurShared& s, int value, int lane) {
    return ld_trans(s.correction + offset<kChunk>(lane % 16, value + lane / 16 * 8));
}
```

---
skill_id: cuda.collective-shared-matrix-stores
intent: Store distributed matrix fragments collectively into shared memory.
preconditions:
- 'Data types: elements occupy 16 bits, paired without conversion in 32-bit registers;
  the b16 store copies bits and imposes no arithmetic or floating-point-format requirement.'
- 'Layout: complete 8x8 fragments follow the instruction lane mapping; each addressed
  shared row contains eight contiguous elements and begins at a 16-byte-aligned address,
  so collective stores reach the coordinates expected by shared readers.'
- 'Storage: source fragments are ready in lane registers and destinations are allocated
  in the same CTA shared memory; stmatrix cannot target global memory.'
- 'Pipeline: all 32 warp lanes execute the same store with identical qualifiers after
  register production; synchronization must publish stores before consumers and finish
  readers before destination reuse, because collective rendezvous alone does not provide
  these memory dependencies.'
- 'Hardware: SM90 or later with assembler support for stmatrix.m8n8.b16; unsupported
  targets cannot encode it. No extra shared capacity is needed because destinations
  are unchanged.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Replace scalar fragment stores in `solution/native.cuh::store_c` and `store_t`
with collective `stmatrix` stores. Replace both complete functions with the After
snippet; its two helpers must precede the wrappers. Keep every call site unchanged.
Retain `st_scalar`: decay preparation still uses it independently.

Each call stores a 16x16 tile from four packed registers per lane. `store_c`
writes row fragments; `store_t` writes the transposed coordinates. The optimization
replaces eight volatile scalar stores per lane with one collective instruction.
It only moves existing bits; retain all quantization, arithmetic, and reduction order.
The volatile scalar helper prevents the deoptimized accesses from being repacked;
forward replay bypasses it only here and requires no compiler-flag changes.

## Ownership and layout

For lane `l`, register `j`, and halfword `e`, the existing row-store destination is
`(row + l/4 + 8*(j%2), col + 2*(l%4) + 8*(j/2) + e)`.
The transpose-store destination is
`(col + 2*(l%4) + 8*(j/2) + e, row + l/4 + 8*(j%2))`.
Here `j=0..3`, `e=0..1`. Both use the retained slab offset
`offset<Rows>(r,c) = 8*r + (c%8) + (c/8)*Rows*8` in elements.
Each slab row occupies `8*sizeof(BF16)` bytes; slab spacing derives from `Rows`.

For `.x4`, consecutive eight-lane groups supply row addresses for consecutive
register fragments. The After wrappers encode those addresses. Do not replace
these addresses with each lane's scalar destination: address ownership and data
ownership differ. All tiles here are complete and all addresses are in bounds;
no launch or bounds-check change is required. An adapted boundary tile needs
valid padded shared destinations and full-warp participation.

## Ordering and consumers

`prepare.cuh` uses `store_c` for `mqk` and the inverse. Their surrounding CTA
barriers publish them before masking, inversion consumers, and workspace export.
`recurrence.cuh` uses `store_c` for corrected values and output, and `store_t`
for state updates. Keep the correction `__syncwarp(kAllLanes)` before its
`ld_trans` readers. Each warp owns distinct value columns; all its correction
reads finish before the next chunk reuses that region.

Keep the CTA barriers after `recur_tile`, after output export, and around state
conversion. They publish state/output, finish readers, and protect chunk reuse.
Register-producing arithmetic remains before the stores. The instruction's
warp rendezvous does not replace these memory-ordering barriers.

## Example configuration

Preserve B=1, T=4096, H=96, D=128, 16-token chunks, and 256 chunks per head.
Preparation launches `grid=(256,96)`, block=256; recurrence launches `grid=(1,96)`,
block=256. Each recurrence warp owns 16 value channels for one head.
`PrepareShared`, `InputShared`, and `RecurShared` remain 42368, 18048, and 124672
bytes. Keep the separate 1024-byte register-transpose scratch and existing
shared-memory opt-in. No allocation or launch changes are needed.

Retain BF16 operands/state/quantization, FP32 accumulators and exported state,
`mma.sync.m16n8k16` arithmetic, `ldmatrix` loads, shared transpose/reduction
exchanges, scalar BF16 additions, normalization/gate calculations, the opening
interleaved products, one serialized input buffer, and CTA-local recurrent state.
Keep config, tensor ABI, caller stream, input contract, compiler flags, and
`sm_90a` compilation. These are replay settings, not additional store prerequisites.

For a subsequent application, use the existing evaluator with the complete
problem and inspect generated code for normal and transposed `STSM` stores.
Keep measurements outside this card.

# Precondition

- Data types: each element is a 16-bit payload packed with its neighbor in a
  32-bit register. The selected instruction transfers that representation;
  it needs neither BF16 arithmetic nor a particular accumulation precision.
- Layout: each complete 8x8 fragment must match the instruction's lane mapping.
  Every row address must name eight contiguous elements aligned to 16 bytes.
  Different row strides are legal when the supplied row addresses match them;
  incorrect ownership or alignment invalidates the collective store. Shared
  readers must interpret these same coordinates.
- Storage: the values already reside in lane registers, and the destination is
  allocated CTA-shared memory. Other address spaces cannot be used with this
  instruction.
- Pipeline: all 32 lanes remain active and reach the same instruction with
  identical qualifiers after producing their registers. Publish completed
  stores before readers and finish reads before overwrite using synchronization
  covering the participants. These prevent divergent collectives and shared
  memory races; no particular stage count is required.
- Hardware: SM90 or later and an assembler supporting `stmatrix.m8n8.b16` are
  needed to encode the instruction. The same destination allocation is reused,
  so there is no additional shared-memory capacity requirement.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

`BF16`, `Reg`, `Pair`, `st_scalar`, `shared_addr`, and `offset<Rows>` already exist
in `solution/native.cuh`. Replace the two Before functions with all four After
functions; preserve these shared definitions and all callers.

## Before

```cuda
template<int Rows> __device__ __forceinline__ void store_c(BF16* p, Reg a, int row, int col, int lane) {
    constexpr int kMatrixSide = 8;
    constexpr int kMatricesPerAxis = 2;
    constexpr int kPairElems = sizeof(uint32_t) / sizeof(BF16);
    constexpr int kRowLanes = kMatrixSide / kPairElems;
    constexpr int kFragments = sizeof(Reg) / sizeof(uint32_t);

    // Write each lane's row fragments without collective matrix stores.
    #pragma unroll
    for (int j = 0; j < kFragments; ++j) {
        const int r = row + lane / kRowLanes + (j % kMatricesPerAxis) * kMatrixSide;
        const int c = col + lane % kRowLanes * kPairElems + (j / kMatricesPerAxis) * kMatrixSide;
        Pair pair{a.x[j]};
        #pragma unroll
        for (int e = 0; e < kPairElems; ++e)
            st_scalar(p + offset<Rows>(r, c + e), pair.h[e]);
    }
}
template<int Rows> __device__ __forceinline__ void store_t(BF16* p, Reg a, int row, int col, int lane) {
    constexpr int kMatrixSide = 8;
    constexpr int kMatricesPerAxis = 2;
    constexpr int kPairElems = sizeof(uint32_t) / sizeof(BF16);
    constexpr int kRowLanes = kMatrixSide / kPairElems;
    constexpr int kFragments = sizeof(Reg) / sizeof(uint32_t);

    // Transpose the destination coordinates, preserving every element's bits.
    #pragma unroll
    for (int j = 0; j < kFragments; ++j) {
        const int r = col + lane % kRowLanes * kPairElems + (j / kMatricesPerAxis) * kMatrixSide;
        const int c = row + lane / kRowLanes + (j % kMatricesPerAxis) * kMatrixSide;
        Pair pair{a.x[j]};
        #pragma unroll
        for (int e = 0; e < kPairElems; ++e)
            st_scalar(p + offset<Rows>(r + e, c), pair.h[e]);
    }
}
```

## After

```cuda
__device__ __forceinline__ void st_matrix(BF16* p, Reg a) {
    asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1,%2,%3,%4};"
      :: "r"(shared_addr(p)), "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]));
}
__device__ __forceinline__ void st_trans(BF16* p, Reg a) {
    asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1,%2,%3,%4};"
      :: "r"(shared_addr(p)), "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]));
}
template<int Rows> __device__ __forceinline__ void store_c(BF16* p, Reg a, int row, int col, int lane) {
    st_matrix(p + offset<Rows>(row + lane % 16, col + lane / 16 * 8), a);
}
template<int Rows> __device__ __forceinline__ void store_t(BF16* p, Reg a, int row, int col, int lane) {
    st_trans(p + offset<Rows>(col + lane % 8 + lane / 16 * 8, row + (lane / 8 % 2) * 8), a);
}
```

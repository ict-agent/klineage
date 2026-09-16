---
skill_id: cuda.warp-cooperative-fragment-stores
intent: Store register fragments into shared memory with warp-cooperative matrix stores.
preconditions:
- 'Data types: values must have a bit-preserving 16-bit representation packable in
  pairs into 32-bit operands; stmatrix.b16 cannot directly encode other widths and
  requires no arithmetic or rounding change.'
- 'Layout: each warp fragment must match the non-transposed m8n8 register mapping;
  each destination matrix row contains eight contiguous 16-bit elements at a 16-byte-aligned
  address. Writer addresses must preserve the layout expected by readers, or values
  land at incorrect coordinates.'
- 'Storage: source fragments are ready in registers and destination tiles already
  occupy CTA shared memory; stmatrix.shared cannot target global memory or inaccessible
  storage.'
- 'Pipeline: all 32 live lanes execute each collective store with identical qualifiers
  after fragment production. Preserve visibility before consumers read, including
  proxy ordering for asynchronous readers, and complete old reads before overwriting
  shared storage; the warp collective alone does not establish these inter-warp or
  reuse dependencies.'
- 'Hardware: the target and assembler must support stmatrix.m8n8.b16 (SM90 or newer,
  PTX 7.8 or newer); existing shared capacity must cover the addressed tiles. This
  replacement allocates no additional shared storage.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Replace per-thread fragment stores with warp-cooperative
`stmatrix.sync.aligned.x4.m8n8.shared.b16`. Each instruction stores four
8-by-8 matrices from four packed registers per lane, reducing store and address
instructions while preserving every destination bit.

Add `store_matrix` inside namespace `hopper` in `solution/hopper.cuh`.
Replace `save_prob` in `solution/attention.cuh` with the After implementation.
Keep its calls in both `consume` specializations and `store_output`: the latter
passes its already rounded `converted` fragment. Inlining this helper restores
the same mechanism at all probability and output staging sites. No host ABI,
launch, allocation, reader, or compiler-flag change is needed.

## Ownership and layout

The existing `lane` is the thread index within its consumer warpgroup.
For register element `i`, the preceding implementation owns logical coordinates
`row_index((i % 4) / 2, lane)` and
`8 * (i / 4) + 2 * (lane % 4) + i % 2`.
Its destination remains `dest[swizzle(row, col)]`.

The collective separates data ownership from row-address ownership.
For warp-local lane `l`, register `j` holds the pair belonging to row `l / 4`
and columns `2 * (l % 4)` and `2 * (l % 4) + 1` of matrix `j`.
Lanes `8*j` through `8*j+7` supply that matrix's row addresses.
For each call here, the four matrices are arranged top-left, bottom-left,
top-right, bottom-right within a 16-by-16 region. Advance 16 columns and eight
source elements per lane between calls. Preserve the non-transposed instruction.

Keep the existing swizzle exactly: for element offset `x = row*kTile + col`,
return `x ^ ((x & (7 << 6)) >> 3)`. It permutes eight-element segments while
leaving each segment contiguous. Thus the collective writes the same physical
layout consumed by `pv_smem` descriptors and the output tensor map. This recipe
does not introduce or remove the swizzle.

## Ordering and bounds

Keep every existing fence, named barrier, WGMMA wait, and free/ready handshake.
The store changes no arithmetic: retain FP32 accumulation and softmax order,
BF16 round-to-nearest conversions before staging, output normalization, and
natural-log statistics.

P0 is written into `sm.kv[0]`'s last tile only after its QK reads complete.
Its existing proxy fence and `Prob0` handshake publish it to the peer PV reader;
that reader's completion precedes `free0[1]` and producer reuse. P1 is written
to `sm.prob`; the existing proxy fence and `Prob1` handshake precede its remote
PV read. The next iteration's `Max0` handshake follows completion of that read
before P1 is overwritten.

The final `Sum` barrier joins both consumers after their QK/PV work, allowing
`sm.q_o` to change from query storage to output staging. Preserve each output
tile's proxy fence and Store0/Store1 barrier before its elected TMA writer.
Output staging ranges are written once and remain intact while stores are
outstanding. Keep store commits and the existing kernel-completion behavior.

All staging tiles in this instance are full. Keep invalid-index zero filling
and score masking where they already occur; do not predicate individual lanes
of `stmatrix` or introduce early exits. A partial tile in another implementation
would need allocated padding and collective participation, or a separate tail
path.

## Example configuration

Preserve the supplied workload: 8192 tokens, 128 heads, one KV head, QK width
576, value width 512, and 2048 selected indices. Q/KV/output are BF16; indices
are int32; reductions and statistics are FP32. Keep scale
`0.1352337788608801f`, its log2 conversion, and all numerical requirements in
the problem. Contiguous global strides derive from each tensor's row extent
and element size; these stores do not access global memory.

Keep 64-by-64 tiles, `kScoreRegs=32`, two fragment rows per thread,
`kOutputRegs=128`, and four output tiles per consumer. Each CTA owns one token
and one 64-head slice; its two consumer warpgroups own separate 256-column
output halves. The third warpgroup gathers KV. Launch grid `(16384,1,1)`,
block `(384,1,1)`, and cluster `(1,1,1)` stay unchanged.

Retain two KV buffers, nine query tiles, 32 key blocks traversed in pairs,
216/72 consumer/producer register budgets, and `sizeof(Shared)=230864` bytes.
Keep the Q/O and K0-RoPE/P0 storage aliases. Retain WGMMA QK/PV instructions,
online softmax, asynchronous KV copies, query/output TMA, statistics bulk
stores, cache policies, and all scheduling. Compile for the supplied SM90a
target with the bundle's existing release flags.

Check the forward change with the existing Kernel evaluator and unchanged
problem. Inspect the actual compiled kernel for matrix-store instructions
(`stmatrix` in PTX, `STSM` in SASS) at both kinds of staging site.

# Precondition

- Data types: values must have a bit-preserving 16-bit representation that can be
  packed two per 32-bit operand without conversion. Another width cannot use
  this `.b16` mapping directly. No additional floating-point or reduction-order
  requirement comes from this bit movement; retain any required conversions
  before the store.
- Layout: register fragments must follow the non-transposed m8n8 mapping.
  Each row has eight contiguous 16-bit elements with a 16-byte-aligned start;
  the instruction groups four lanes into that row transaction. Different rows
  need not be contiguous. Row addresses must reproduce the physical layout
  expected by readers; incorrect packing or addresses misplace values.
- Storage: completed fragments are in registers and destination tiles are
  already allocated in CTA shared memory. The shared-space instruction cannot
  store to global memory or another inaccessible allocation.
- Pipeline: all 32 live warp lanes reach each store with identical qualifiers
  after producing their fragments; divergent participation is undefined.
  Preserve publication before readers, including proxy ordering for asynchronous
  consumers, and finish prior reads before storage reuse. The collective
  synchronizes its warp but supplies neither inter-warp publication nor buffer
  lifetime management.
- Hardware: SM90-or-newer hardware and a PTX-7.8-or-newer assembler must support
  `stmatrix.m8n8.b16`; otherwise the instruction is unavailable. Existing shared
  capacity must cover all addressed tiles. No additional capacity is required
  because their sizes and lifetimes do not change.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Both versions use the existing `Bf16`, `row_index`, `swizzle`, `shared_addr`,
`kScoreRegs`, and `kRowsPerThread`. Keep all callers and their synchronization.
The After block adds a low-level helper in `hopper.cuh` and replaces the
high-level store in `attention.cuh`.

## Before

```cuda
__device__ __forceinline__ void save_prob(const Bf16* p, Bf16* dest, int lane) {
    constexpr int kLanesPerRow = 4;
    constexpr int kPairElems = 2;
    constexpr int kColStep = kLanesPerRow * kPairElems;
    constexpr int kRegsPerStep = kRowsPerThread * kPairElems;

    // Store each lane's elements at the retained swizzled coordinates.
#pragma unroll
    for (int row = 0; row < kRowsPerThread; ++row) {
        const int real_row = row_index(row, lane);
#pragma unroll
        for (int i = row * kPairElems; i < kScoreRegs; i += kRegsPerStep) {
            const int col = kColStep * (i / kRegsPerStep) + (lane % kLanesPerRow) * kPairElems;
            dest[swizzle(real_row, col)] = p[i];
            dest[swizzle(real_row, col + 1)] = p[i + 1];
        }
    }
}
```

## After

```cuda
// solution/hopper.cuh, namespace hopper
__device__ __forceinline__ void store_matrix(Bf16* dst, const Bf16* src) {
    const auto* r = reinterpret_cast<const uint32_t*>(src);
    asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};"
                 :: "r"(shared_addr(dst)), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]));
}

// solution/attention.cuh, namespace spa
__device__ __forceinline__ void save_prob(const Bf16* p, Bf16* dest, int lane) {
    constexpr int kWarpLanes = 32;
    constexpr int kMatrixRows = 8;
    constexpr int kRegionRows = 2 * kMatrixRows;
    constexpr int kRegionCols = 2 * kMatrixRows;
    constexpr int kElemsPerStore = 8;
    const int row = (lane / kWarpLanes) * kRegionRows + lane % kRegionRows;
    const int col = ((lane % kWarpLanes) / kRegionRows) * kMatrixRows;
#pragma unroll
    for (int i = 0; i < kScoreRegs / kElemsPerStore; ++i)
        store_matrix(dest + swizzle(row, col + kRegionCols * i), p + kElemsPerStore * i);
}
```

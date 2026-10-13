---
skill_id: fmha.epilogue-shared-xor-swizzle
intent: Reduce shared-memory bank conflicts with an XOR permutation of output segments.
preconditions:
- 'Data types: no additional arithmetic dtype requirement; the permutation must move
  opaque 16-byte segments without changing their bits or conversion and reduction
  order.'
- 'Layout: writers and readers must agree on logical row and segment ownership in
  complete 128-byte rows; 16-byte-aligned transfers must stay within their segments
  so the XOR mapping preserves alignment and contiguous accesses.'
- 'Storage: intermediate rows already occupy CTA-shared memory visible to their writers
  and readers, with capacity at least total_rows * row_bytes across all panels; the
  permutation must stay within that allocation.'
- 'Pipeline: prior users of aliased storage must finish before writes, all writes
  must become visible before reads, and all readers must finish before reuse; address
  permutation supplies none of these ordering guarantees.'
- 'Hardware: banked CTA-shared memory with 32 banks of four bytes and integer XOR
  addressing; changing segment placement rotates the banks reached by different rows
  without requiring extra storage.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Apply a row-dependent XOR permutation to the epilogue's shared-memory output
staging. This spreads aligned segment accesses across shared-memory banks.
Change only `offset<Rows>` in `solution/ops.cuh`, using the After snippet.
Its two callers in `solution/attention.cuh::epilogue` automatically update
both the `stmatrix` writer and the `uint4` reader. Keep their logical indices.

The existing helper lays out columns as panels of `kPanelCols` elements:
`index = row * 64 + (col & 63) + (col / kPanelCols) * Rows * 64`.
Indices are in half elements. With a 128-byte row, eight 16-byte segments
share the row. XOR segment index with `(row + panel * Rows) & 7`, preserving
the eight elements within each segment. Equivalently, XOR index bits 3–5 with bits
6–8: `index ^ ((index >> 3) & 56)`. This mapping is its own inverse and
never moves an element outside its 128-byte row.

## Ownership and ordering

In `epilogue`, `tid = threadIdx.x - kGroupSize`, `lane = tid % 32`, and
`warp = tid / 32`. The `stmatrix.sync.aligned.x4.m8n8.shared.b16` row-address
provider uses `row = warp * 16 + lane % 16` and
`col = i * 16 + (lane / 16) * 8`. Each supplied row segment contains eight
16-bit values. The instruction's register-fragment mapping remains unchanged.
The readback uses `row = tid / 8 + r * 32` and
`col = (tid % 8) * 8 + c * 64`. Both accesses must call the same helper;
changing one side alone would return different output elements.

Output staging aliases `s.v`. Retain the first
`sync(Named::Epilogue, kMathThreads)` after each consumer's final
`mma_wait<0>()`, so every PV reader has finished before output overwrites V.
Retain the second epilogue barrier, which makes all `stmatrix` stores visible
before readback. Each consumer loads its segments into `values` before
`proxy_fence()` and `signal(&s.o_empty)`. The producer waits for all arrivals
on `o_empty` before reusing this storage for V. Global stores then use the
register copies. Keep every participating thread and all these operations.

Keep the host ABI, caller stream, launch, allocation, TMA descriptors,
WGMMA descriptors, and operand swizzles unchanged. The helper addresses only
output staging; Q/K/V TMA layouts do not use it. Keep output and LSE row
guards (`row < length`), KV-tail masking, sequence offsets, and tile traversal.
Keep FP32 arithmetic, FP16 probability conversion, and the existing
round-to-nearest FP16 output conversion before staging. The permutation
changes addresses only.

## Example configuration

Replay the supplied packed-NHD FMHA specialization: 16384 tokens, 64 heads,
head dimension 128, and eight sequences. Preserve the supplied problem and
workload, including the offset tensors. Inputs and output are FP16;
accumulators are FP32. Global row stride is `kRow = kHeads * kDim` elements
(16384 bytes here). Noncausal attention uses the existing scale constants.

Keep `kM = 128`, `kN = 176`, `kDim = 128`, `kPanelCols = 64`, and the two
K/V stages. A Q tile occupies 32768 bytes; each K or V stage occupies
45056 bytes. Output staging occupies the first 32768 bytes of `s.v`.
Preserve `Shared`, its alignment, and the `sizeof(Shared)` launch allocation.
The tile contains two output panels, each with 128 rows of 64 half elements.

Keep 384 threads per CTA: one 128-thread producer group, with its first
32 threads doing transfers, and two 128-thread consumer groups. Keep the
24/240 register redistribution, one persistent CTA per SM, atomic work
assignment, and sequence-length ordering. Keep the 88 QK, 64 PV, and 44
packed-probability registers per consumer thread. Retain WGMMA with K=16,
interleaved QK/PV and softmax work, TMA operand transfers with SW128 layouts,
two-stage K/V buffering, serialized query loading, and V/output aliasing.
Keep the STSM writer, vector readback and global stores, cache policies,
programmatic launch dependency, and existing build flags. These settings
describe this replay; they do not define the address-permutation technique.

Validate correctness and latency with `klineage.harness.evaluate` on the
unchanged problem. Keep measurements outside this card.

# Precondition

- Data types: no additional arithmetic dtype requirement. Permute opaque
  16-byte segments without changing their bits or conversion and reduction
  order; otherwise the transformation would also change numerical behavior.
  The example's half-element indexing is an address unit, not a requirement
  for FP16 arithmetic.
- Layout: writers and readers must agree on logical row and segment ownership
  in complete 128-byte rows. Their transfers must be aligned to 16 bytes and
  stay within each segment. XOR permutes the eight segments inside a row;
  crossing segment boundaries would invalidate an unchanged contiguous
  transfer, and unmatched writer/reader addresses would select wrong data.
- Storage: intermediate rows already occupy CTA-shared memory visible to
  their writers and readers. Capacity must cover `total_rows * row_bytes`
  across all panels.
  The permutation stays inside those rows, so it requires no added capacity;
  an incomplete row allocation could receive an out-of-bounds mapped address.
- Pipeline: prior readers of aliased storage must complete before output
  writes; those writes must become visible before output reads; every reader
  must complete before reuse. The permutation provides no synchronization,
  so removing any of these guarantees permits stale data or an overwrite.
- Hardware: banked CTA-shared memory with 32 four-byte banks and integer XOR
  addressing. A 16-byte segment spans four banks; rotating segment placement
  changes the bank groups reached by different rows. This bank geometry
  explains the chosen permutation; other geometries need an adapted mapping.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Replace only `offset<Rows>` and its comment in `solution/ops.cuh`.
The existing `kPanelCols` and both epilogue callers provide the shared context.
The constants below name the original permutation's bit fields.

## Before

```cuda
// Linear output addresses, in half elements; each panel occupies Rows * 64.
template<int Rows>
__device__ __forceinline__ int offset(int row, int col) {
    int index = row * 64 + (col & 63) + (col / kPanelCols) * Rows * 64;
    return index;
}
```

## After

```cuda
// Permute output segments to spread shared-memory bank accesses.
template<int Rows>
__device__ __forceinline__ int offset(int row, int col) {
    int index = row * 64 + (col & 63) + (col / kPanelCols) * Rows * 64;
    constexpr int kSwizzleShift = 3;
    constexpr int kSwizzleMask = 0b111000;

    // Preserve each aligned eight-half segment; XOR its row selector.
    return index ^ ((index >> kSwizzleShift) & kSwizzleMask);
}
```

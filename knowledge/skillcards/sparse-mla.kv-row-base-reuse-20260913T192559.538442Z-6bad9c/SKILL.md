---
skill_id: sparse-mla.kv-row-base-reuse
intent: Reuse one sparse KV row address across a group of scalar loads.
preconditions:
- 'Data types: no arithmetic dtype restriction; index and address types must represent
  the selected row and element offsets without overflow so hoisted and repeated addresses
  agree.'
- 'Layout: the grouped scalar accesses must share an invariant row base with known
  element offsets; otherwise one reused pointer cannot address every intended element.'
- 'Storage: repeated-address bookkeeping must be thread-private and unobservable to
  other threads; removing its temporary and reloads must not remove communication.
  No particular tensor memory space is required.'
- 'Pipeline: the row index, stride, and base must be ready and unchanged throughout
  the load group or hoisting changes addresses. No additional transfer ordering or
  buffer-reuse requirement applies; preserve existing producer visibility and reader-completion
  barriers because the transfers retain their order.'
- 'Hardware: no additional feature or capacity requirement; ordinary scalar pointer
  arithmetic already supports the loads, and one reused address replaces repeated
  address calculations.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Hoist the sparse KV row address out of the scalar transfer loop. In
`solution/attention.cuh`, `copy_tiles` already obtains a selected index and
checks its validity for each column tile. Compute its row pointer once,
then pass that pointer plus the column-tile offset to `copy_kv`.

In `solution/hopper.cuh`, remove `copy_kv`'s fourth `row_offset` argument,
the local volatile `offset`, and the per-element `row_pointer` call.
Each load then uses `src+i` from the shared row base. The volatile local
exists only to prevent the compiler from hoisting the repeated address in
the starting kernel; remove it when restoring reuse. Keep the existing
`row_pointer` helper and its opaque address constraint unchanged.

This reuses address arithmetic, not loaded tensor values. Retain every
scalar `ld.volatile.global.b16` and `st.volatile.global.b16`, the selected
index reload per column tile, the copy loop's unrolling, and invalid-row
zero filling. Do not vectorize or stage another buffer.

Lane ownership, KV/probability layouts, QK/PV arithmetic, register tiles,
masking, and launch configuration remain unchanged. The first consumer
still gathers both KV tiles. Keep the Stage barrier that publishes its
stores before both consumers read and the final Stage barrier that waits
for all readers before overwrite. Preserve all probability/statistics
barriers, caller-stream ordering, and allocation lifetimes.

All valid source addresses must stay within the selected KV row. Invalid
indices still perform no global load and write zero to each destination
element. Preserve all BF16 bits, FP32 reduction order, rounding points,
output semantics, problem metadata, and compiler flags.

After replay, use `klineage.harness.evaluate` on the complete problem.
Inspect generated copy instructions: one computed row base should supply
all scalar source offsets; per-element volatile local offset loads should
be absent. Keep measurements outside this card.

## Example configuration

Retain 8192 tokens, 128 heads, QK width 576, value width 512, and 2048
selected indices. Query/KV and probability scratch are BF16; sparse indices
are signed 32-bit. Row offsets and pointers use 64-bit addressing.
`kv_stride=576` is measured in BF16 elements; the source row stride is
1152 bytes. The row-offset multiplication remains once per column tile.

Each `copy_kv` transfers eight BF16 elements (16 bytes) using scalar loads
and stores. Its source base includes `(lane%8)*8 + tile*64` elements;
all eight elements use the same sparse row. `kCopyBytes=16` and
`kCopyElems=kCopyBytes/sizeof(Bf16)` remain unchanged. For invalid indices,
`bytes=0`; otherwise `bytes=16`.

Keep 64x64 tiles, 128 threads per consumer group, two consumer groups,
256 threads per CTA, and 16384 CTAs. The cluster contains one CTA.
The gather uses eight lanes per row, sixteen row groups, four rows per
lane group, nine QK column tiles, and the existing four transfer groups.
Each CTA keeps two global KV tiles of 64x576 elements (147456 bytes total)
and two global probability tiles of 64x64 elements (16384 bytes total).
Retain row-major KV tiles and the probability buffer's 8x8 core layout.

Keep 16 iterations over pairs of selected-key blocks, all named barriers,
shared masks/statistics/reduction scratch, scalar QK/PV FMAs, and online
softmax. Score/output register arrays remain 32/128 FP32 elements per lane.
Retain `kScale=0.1352337788608801f`, intermediate BF16 rounding, final BF16
output, and FP32 maximum/log-sum-exp. No numerical instruction changes.

# Precondition

- Data types: no additional arithmetic dtype requirement applies. Index
  and pointer arithmetic must represent the selected row and element
  offsets without overflow; otherwise hoisting can produce a different
  or invalid address. Tensor representations are merely copied.
- Layout: every grouped scalar access must share one invariant row base
  and have a known element offset. A different row per element would make
  reusing one pointer incorrect. Contiguity is this example's offset
  pattern, not an intrinsic requirement.
- Storage: address bookkeeping must be private and unobservable outside
  its thread. Removing its temporary/reloads must not remove another
  thread's communication. The technique imposes no tensor memory-space
  requirement; source and destination placement need not change.
- Pipeline: the row index, stride, and base must be available before the
  group and invariant through its loads, or a hoisted address becomes
  stale. There is no added transfer-ordering or reuse constraint. Preserve
  producer visibility before consumption and reader completion before
  overwrite through the existing barriers; transfer order is unchanged.
- Hardware: no additional feature or capacity is needed. Ordinary scalar
  address arithmetic already serves these loads. Reusing one address
  replaces repeated calculations without introducing another memory tile
  or specialized instruction.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The caller fragments replace the `copy_kv` call in `copy_tiles` after
`index` and `valid` are computed. The helper fragments replace `copy_kv`
in namespace `hopper`. All other helpers and constants remain available.

## Before

```cuda
// copy_tiles: pass the row displacement for per-element address calculation.
copy_kv(src + tile * kTile,
        dst + tile * kTileElems + row * kCopyGroups * kTile,
        valid ? 16 : 0, int64_t(index) * params.kv_stride);

// hopper.cuh
__device__ __forceinline__ void copy_kv(
    const Bf16* src, Bf16* dst, int bytes, int64_t row_offset) {
    constexpr int kCopyBytes = 16;
    constexpr int kCopyElems = kCopyBytes / sizeof(Bf16);
    volatile int64_t offset = row_offset;

#pragma unroll
    for (int i = 0; i < kCopyElems; ++i) {
        uint16_t value = 0;
        if (bytes != 0) {
            const Bf16* row_src = row_pointer(src, offset);
            asm volatile("ld.volatile.global.b16 %0, [%1];"
                         : "=h"(value) : "l"(row_src + i) : "memory");
        }
        asm volatile("st.volatile.global.b16 [%0], %1;"
                     :: "l"(dst + i), "h"(value) : "memory");
    }
}
```

## After

```cuda
// copy_tiles: reuse one row base across the scalar source offsets.
const Bf16* row_src = row_pointer(src, int64_t(index) * params.kv_stride);
copy_kv(row_src + tile * kTile,
        dst + tile * kTileElems + row * kCopyGroups * kTile,
        valid ? 16 : 0);

// hopper.cuh
__device__ __forceinline__ void copy_kv(
    const Bf16* src, Bf16* dst, int bytes) {
    constexpr int kCopyBytes = 16;
    constexpr int kCopyElems = kCopyBytes / sizeof(Bf16);

#pragma unroll
    for (int i = 0; i < kCopyElems; ++i) {
        uint16_t value = 0;
        if (bytes != 0) {
            asm volatile("ld.volatile.global.b16 %0, [%1];"
                         : "=h"(value) : "l"(src + i) : "memory");
        }
        asm volatile("st.volatile.global.b16 [%0], %1;"
                     :: "l"(dst + i), "h"(value) : "memory");
    }
}
```

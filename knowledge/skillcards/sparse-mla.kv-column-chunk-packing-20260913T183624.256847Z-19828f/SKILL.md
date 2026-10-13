---
skill_id: sparse-mla.kv-column-chunk-packing
intent: Pack gathered KV column chunks to shorten the row stride of cooperative QK
  loads.
preconditions:
- 'Data types: no additional dtype restriction; packing only relocates element bits
  and must preserve their representation, conversions, and arithmetic order.'
- 'Layout: known R-by-D tiles with chunk extent C dividing D and C < D; QK lanes vary
  the sparse row at a fixed feature column, and producers plus every QK/PV reader
  can use one permutation. These conditions shorten the row stride without mismatched
  addresses.'
- 'Storage: gathered KV tiles already occupy writable global scratch with nonoverlapping
  writer ownership and known consumers; otherwise permutation can race or leave a
  consumer reading the old layout.'
- 'Pipeline: producers and consumers can synchronize within the owning CTA; completed
  writes must be visible before QK/PV reads, and all reads must finish before tile
  reuse. Otherwise either layout can expose incomplete or overwritten operands.'
- 'Hardware: no additional requirement; the permutation uses existing scalar global
  loads/stores and CTA barriers, occupies the same R*D elements, and requires no new
  instruction, memory level, alignment mode, or capacity.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Pack column chunks of gathered KV tiles together so QK lanes accessing different
sparse rows at the same feature coordinate use a shorter physical row stride.
Change only `kv_index` in `solution/hopper.cuh` and the producer row increment in
`copy_tiles` in `solution/attention.cuh`. Retain global KV staging.

For a tile with `R` sparse rows, `D` feature columns, and chunk extent `C`, replace
`row*D + col` with `(col/C)*R*C + row*C + col%C`. This is a bijection when `C`
divides `D`. At a fixed column, adjacent rows become `C` elements apart instead
of `D`; each chunk's columns remain contiguous. This is column-chunk packing,
without an XOR swizzle.

## Replay

Set `R = D = kTile` and `C = kCoreElems` using existing constants. Restore the
packed `kv_index` formula below. In `copy_tiles`, change only the destination's
row increment from `row*kCopyGroups*kTile` to
`row*kCopyGroups*kCoreElems`. Keep its initial destination
`kv_buffer(params, Buffer) + kv_index(group, column)` and feature-tile increment
`tile*kTileElems`. A producer lane owns its existing contiguous column chunk;
its subsequent sparse rows are `kCopyGroups` rows apart.

Both consumers already call `kv_index`: `qk_tile` reads key row `col` and feature
`k`; `pv_smem` reads sparse row `k` and value column `col % kTile`, with the
existing feature-tile offset. They automatically adopt the new physical layout.
Do not change their logical coordinates, ownership, loops, or reduction order.
`prob_index`, probability storage, and final output addressing stay unchanged.

Keep `copy_kv` and `load_index` unchanged, including volatile scalar transfers.
Invalid indices still zero-fill each entire destination chunk and populate
`sm.valid`; `mask_scores` still sets invalid logits to negative infinity.
The supplied workload tiles exactly. A different partial tile needs bounds or
padding in both writer and reader mappings.

Keep both `sync<Barrier::Stage, kThreads>()` calls in `consume`: the first makes
all gathered stores and masks visible to QK/PV, and the second completes all
readers before either KV buffer is overwritten. Preserve local probability,
maximum, probability-exchange, and sum barriers. No launch, allocation, shared
storage, or caller-stream change is needed.

Use the existing evaluator for correctness and latency. Check generated gather
store addresses: after packing, successive producer row batches are separated
by `kCopyGroups*kCoreElems*sizeof(Bf16)` bytes. Preserve the scalar volatile
stores so compilation cannot silently repack transfers. Keep evidence outside
this card.

## Example configuration

The supplied problem has 8192 tokens, 128 heads, QK width 576, value width 512,
and 2048 sparse indices per token. Inputs and probabilities are BF16, indices
are int32, and score/output accumulators and statistics are FP32. Retain BF16
probability rounding, final BF16 conversion, softmax scale
`0.1352337788608801f`, and all existing exponential and normalization behavior.

Each KV feature tile has `R=D=64`, `C=8`, and two-byte elements: 8192 bytes per
tile. Packing changes the sparse-row stride from 128 to 16 bytes; a column chunk
spans 1024 bytes. Each KV buffer holds nine feature tiles, or 73,728 bytes.
There are two KV buffers per CTA, totaling 2.25 GiB across the launch. The two
global probability buffers total 256 MiB; shared masks, statistics, and reduction
scratch occupy 1920 bytes per CTA. All allocations remain unchanged.

The launch has 16,384 CTAs, each owning one token and 64 heads, with 256 threads
split into two 128-thread consumers. Each lane retains two logical rows,
32 score registers, and 128 FP32 output accumulators. Group 0 also gathers KV.
For copies, `kCopyGroup=8`, `kCopyGroups=16`, `kCopyRows=4`, and
`kVectorElems=8`; each helper performs eight scalar element transfers. The
producer row-batch increment changes from 2048 to 256 bytes. Retain the four
copy groups and both KV buffers. No dedicated producer or async copy is added.

Retain the block-pair traversal, QK tile order (group 0: 0 through 8; group 1:
4 through 8, then 0 through 3), increasing inner-K FMA order, output register
tiling, cooperative online softmax, probability layout, scalar left-operand
reloads, shared reductions, and direct output stores. Keep launch bounds,
single-CTA clusters, ABI checks, caller stream, and all build flags including
`--use_fast_math` unchanged.

# Precondition

- Data types: no additional dtype restriction. This permutation copies existing
  bits; it introduces no arithmetic. Preserve element representation, conversion,
  and reduction order so relocated operands have identical numerical meaning.
- Layout: tiles have known extents `R` and `D`; choose `C` dividing `D` with
  `C < D`. Divisibility makes the unpadded mapping bijective, and the inequality
  shortens the sparse-row stride. QK lanes vary sparse row at a fixed feature
  column, making that stride relevant. Producers and every QK/PV reader must
  agree on the permutation or they will address different logical elements.
- Storage: gathered KV tiles already reside in writable global scratch with
  nonoverlapping writer ownership and known consumers. Without this ownership,
  stores can race; an unconverted consumer would read the wrong layout.
- Pipeline: producers and consumers can synchronize within their owning CTA.
  Producer completion and visibility must precede QK/PV reads, and every reader
  must complete before storage is reused. These dependencies prevent incomplete
  and overwritten reads; the technique does not require a particular buffer count.
- Hardware: no additional requirement. The permutation uses the existing scalar
  global loads/stores and CTA barriers and occupies the same `R*D` elements.
  It requires no new instruction, memory level, alignment mode, or capacity.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The `copy_tiles` excerpts use its unchanged template parameters `Buffer`, `Begin`,
and `End`, function arguments, and local variables. All omitted code is retained.

## Before

```cuda
// solution/hopper.cuh
__device__ __forceinline__ int kv_index(int row, int col) {
    // Keep each gathered KV tile row-major for both scalar readers.
    return row * kTile + col;
}

// solution/attention.cuh: copy_tiles
Bf16* dst = kv_buffer(params, Buffer) + kv_index(group, column);
// Inside the unchanged row and feature-tile loops:
copy_kv(row_src + tile * kTile,
        dst + tile * kTileElems + row * kCopyGroups * kTile,
        valid ? 16 : 0);
```

## After

```cuda
// solution/hopper.cuh
__device__ __forceinline__ int kv_index(int row, int col) {
    // Pack KV core columns contiguously for both scalar readers.
    return (col / kCoreElems) * kTile * kCoreElems +
           row * kCoreElems + col % kCoreElems;
}

// solution/attention.cuh: copy_tiles
Bf16* dst = kv_buffer(params, Buffer) + kv_index(group, column);
// Inside the unchanged row and feature-tile loops:
copy_kv(row_src + tile * kTile,
        dst + tile * kTileElems + row * kCopyGroups * kCoreElems,
        valid ? 16 : 0);
```

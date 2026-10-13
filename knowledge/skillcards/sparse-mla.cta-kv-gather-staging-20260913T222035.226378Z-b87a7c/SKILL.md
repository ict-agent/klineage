---
skill_id: sparse-mla.cta-kv-gather-staging
intent: Stage selected KV rows once per CTA for reuse across QK and PV consumers.
preconditions:
- 'Data types: staging must preserve each operand representation and the existing
  arithmetic order; copying bits adds no arithmetic dtype requirement.'
- 'Layout: known source strides, selection offsets, and CTA consumer ownership must
  identify the same operand for every staged read; scalar transfers require natural
  element alignment.'
- 'Storage: source KV rows and selection indices are readable in global memory and
  remain unchanged throughout reuse; the reused consumer set is contained in one CTA
  because CTA-local publication cannot coordinate unrelated CTAs.'
- 'Pipeline: source producers must complete before gathering, and all CTA participants
  must reach publication and reuse barriers; consumers must see completed writes and
  finish before storage is overwritten.'
- 'Hardware: global loads/stores and CTA barriers are available, with free global
  capacity of at least N_CTA * B * R * D * sizeof(KV_element) for the staged rows
  in addition to existing storage.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Gather each CTA's selected KV rows once into private global scratch, then reuse
those rows across its heads and QK/PV operations. This removes repeated sparse
index/address resolution from the arithmetic loops. It does not introduce shared
KV storage, asynchronous transfers, or a new pipeline.

Apply the changes below in `solution/attention.cuh`, `solution/hopper.cuh`, and
`solution/kernel.cu`. Retain the configured export, caller stream, compile flags,
launch, validity masks, probability layout, and all existing barriers.

## Example configuration

The supplied problem is sparse MLA prefill with 8192 tokens, 128 heads, QK width
576, value width 512, and 2048 selected positions. Q/KV and probabilities are BF16;
QK/PV accumulators, maxima, and sums are FP32. Keep the scale
`0.1352337788608801f`, existing exponential/reciprocal helpers, probability
rounding, and final BF16 rounding unchanged. Invalid indices still supply zero KV
bits and receive negative-infinity logits.

Each CTA handles one token and 64 heads. Keep 16384 CTAs, 256 threads, two
128-thread consumer groups, singleton clusters, and 1920 shared bytes. Each lane
owns two head rows via `row_index` and retains 32 score and 128 output registers.
Group 0 owns output columns 0–255; group 1 owns 256–511. Keep 64-position tiles,
32 selected blocks, and the loop's paired-block traversal.

Restore two KV buffers per CTA, each holding `kTile * kWidth` BF16 elements.
Their allocation is 2415919104 bytes. Keep the separate 268435456-byte probability
allocation. These are global allocations, not shared-memory capacity requirements.

Source KV has shape `[kTokens, 1, kWidth]` with row stride `params.kv_stride` in
BF16 elements. The staged layout is
`[CTA][buffer][width_tile][selected_row][width_column]`, with 64×64 row-major
width tiles. Thus source column `c` maps to
`(c / kTile) * kTileElems + row * kTile + c % kTile`.
`row` selects `indices[token * kTopk + (block + buffer) * kTile + row]`.

Keep QK width-tile order 0–8 for group 0 and 4–8 then 0–3 for group 1.
Keep increasing inner K order, PV call order, online softmax, scalar FMAs,
probability recomputation, shared peer reductions, and volatile Q/probability
operand loads. Preserve the mask and probability publication/reuse barriers.
The synchronous copies below retain scalar transfers, repeated index reads,
recomputed row addresses, default cache policy, and the four copy groups.

## Replay edits

1. In `attention.cuh`, add `kKvBuffers = 2`,
   `kKvTileElems = kTile * kWidth`, `kKeyTiles = kWidth / kTile`, and
   `kVectorElems = 8` as `constexpr int` constants. Append `Bf16* kv_tiles` to
   `Params`. Replace `load_kv_value` with `kv_buffer` from After.
2. In `hopper.cuh`, add the three After helpers before the namespace closes.
   `kv_index` describes the staged row-major width tile. `copy_kv` copies bits;
   its volatile instructions and per-element offset reload preserve the existing
   scalar transfer behavior. Do not replace these with vector or async copies.
3. In `attention.cuh`, define `copy_tiles` before `load_valid`. Rename the
   `load_valid` declaration, definition, and call to `load_kv`. Prepend the four
   After copy calls to its body, before the unchanged validity-mask writes.
   Only group 0 calls it. Keep the following `Stage` barrier, which publishes
   the global tiles and shared masks to both consumer groups. Keep the final
   `Stage` barrier: every QK/PV reader must finish before the next pair is copied.
4. Adapt QK as shown in After: add its staged `key` pointer and replace both
   direct gather operands. Remove the now-unused `int block` parameter from
   `qk_tile`, `qk_left`, `qk_right`, and `qk_peer`; remove the matching argument
   from their calls. Keep every template specialization and accumulation mode.
5. Restore `pv_smem(const Bf16* s, const Bf16* v,
   float (&o)[kOutputRegs])`. Remove its Params/token/block/value_base parameters.
   Use the staged PV operands and call substitutions shown in After. Probability
   operands, loops, accumulator ownership, and all surrounding synchronization
   remain unchanged.
6. In the host wrapper, allocate `kv_tiles` as shown in After immediately after
   `probabilities`, and append its pointer to the Params initializer. Keep both
   tensor allocations alive through the asynchronous caller-stream launch.
   No launch dimensions or shared-memory changes are needed.

The copy mapping covers every staged element once: eight lanes split each row
into eight-element spans; sixteen lane groups cover sixteen rows, with four row
iterations completing the tile. The four calls cover all width tiles of both
buffers. Recompute validity as before, never dereference an invalid source row,
and write zero bits for every invalid staged element.

Check the complete problem using the existing evaluator. Inspect generated code
for staged KV writes and consumer reads, while retaining scalar copy operations.
Do not alter the workload, oracle, tolerances, seed, or timing policy.

# Precondition

- Data types: staging must preserve operand bits and the existing arithmetic
  order. There is no additional arithmetic dtype requirement: staging copies
  representations rather than calculating new values. Converting to a narrower
  staging representation would violate this dependency.
- Layout: source strides, sparse-selection offsets, and CTA consumer ownership
  must identify the same value at every staged read. A mismatched writer/reader
  layout supplies the wrong operand. Scalar transfer addresses require the
  natural alignment of their element type; no vector alignment is needed.
- Storage: original KV rows and selection indices must be readable in global
  memory and remain unchanged throughout reuse; otherwise the staged snapshot
  can differ from direct reads. Reused consumers must belong to one CTA so its
  publication can cover them; unrelated CTAs cannot rely on that CTA's barriers.
- Pipeline: source producers must finish before gathering. Every CTA participant
  must reach the publication barrier before consumption and the reuse barrier
  after its reads. These dependencies establish reader visibility and prevent
  uninitialized reads or overwrite races; they do not require a particular
  buffer count.
- Hardware: global loads/stores and CTA barriers are required for transfer and
  publication. Free global capacity must cover
  `N_CTA * B * R * D * sizeof(KV_element)` beyond existing allocations, where
  `B` is the number of simultaneously live staged buffers, `R` selected rows per
  buffer, and `D` source columns. Without that capacity the scratch allocation
  cannot hold each CTA's independent live rows.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

The direct gather helper in `attention.cuh` supplies the shown QK/PV operands.
The existing group-0 `load_valid` call writes only masks; the host allocates only
probability scratch.

```cuda
__device__ __forceinline__ float load_kv_value(
    const Params& params, int token, int block, int row, int col) {
    const int offset = token * kTopk + block * kTile + row;
    const int index = load_index(params.indices + offset);
    if (index < 0 || index >= kTokens) return 0.0f;

    return __bfloat162float(params.kv[int64_t(index) * params.kv_stride + col]);
}

// QK inner loop.
p[i] = fmaf(load_operand(q + row_index(row, lane) * kWidth + k),
            load_kv_value(params, token, block + Buffer, col, Tile * kTile + k), p[i]);
p[i + 1] = fmaf(load_operand(q + row_index(row, lane) * kWidth + k),
                load_kv_value(params, token, block + Buffer, col + 1, Tile * kTile + k), p[i + 1]);

// PV inner loop.
o[i] = fmaf(load_operand(s + prob_index(row_index(row, lane), k)),
            load_kv_value(params, token, block, k, value_base + col), o[i]);
o[i + 1] = fmaf(load_operand(s + prob_index(row_index(row, lane), k)),
                load_kv_value(params, token, block, k, value_base + col + 1), o[i + 1]);

// Existing PV calls, in their separate branches and synchronization positions.
// Group 0 local, then peer:
pv_smem(params, token, block, 0, local_prob, o);
pv_smem(params, token, block + 1, 0, prob, o);
// Group 1 local, then peer:
pv_smem(params, token, block + 1, kValue / 2, prob, o);
pv_smem(params, token, block, kValue / 2, prob0, o);
```

## After

Add these scalar-copy helpers in `hopper.cuh`:

```cuda
__device__ __forceinline__ int kv_index(int row, int col) {
    return row * kTile + col;
}

__device__ __forceinline__ const Bf16* row_pointer(const Bf16* base, int64_t index) {
    uint64_t address = reinterpret_cast<uint64_t>(base + index);
    asm volatile("" : "+l"(address));
    return reinterpret_cast<const Bf16*>(address);
}

__device__ __forceinline__ void copy_kv(
    const Bf16* src, Bf16* dst, int bytes, int64_t row_offset) {
    constexpr int kCopyBytes = 16;
    constexpr int kCopyElems = kCopyBytes / sizeof(Bf16);
    volatile int64_t offset = row_offset;

    // Preserve scalar transfers and zero-fill without reading invalid rows.
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

Replace the direct gather helper and add the cooperative loader in
`attention.cuh`:

```cuda
__device__ __forceinline__ Bf16* kv_buffer(const Params& params, int buffer) {
    return params.kv_tiles + (blockIdx.x * kKvBuffers + buffer) * kKvTileElems;
}

template<int Buffer, int Begin, int End>
__device__ __forceinline__ void copy_tiles(
    Shared& sm, const Params& params, int lane, int token, int block) {
    constexpr int kCopyBytes = 16;
    const int group = lane / kCopyGroup;
    const int column = (lane % kCopyGroup) * kVectorElems;
    Bf16* dst = kv_buffer(params, Buffer) + kv_index(group, column);
    const Bf16* src = params.kv + column;
    const int* rows = params.indices + token * kTopk;

#pragma unroll
    for (int row = 0; row < kCopyRows; ++row) {
        const int offset = (block + Buffer) * kTile + row * kCopyGroups + group;
#pragma unroll
        for (int tile = Begin; tile < End; ++tile) {
            const int index = load_index(rows + offset);
            const bool valid = index >= 0 && index < kTokens;
            copy_kv(src + tile * kTile,
                    dst + tile * kTileElems + row * kCopyGroups * kTile,
                    valid ? kCopyBytes : 0, int64_t(index) * params.kv_stride);
        }
    }
}

// Prepend to renamed load_kv, before its unchanged mask writes.
copy_tiles<0, 0, kHalfTiles>(sm, params, lane, token, block);
copy_tiles<1, kHalfTiles, kKeyTiles>(sm, params, lane, token, block);
copy_tiles<0, kHalfTiles, kKeyTiles>(sm, params, lane, token, block);
copy_tiles<1, 0, kHalfTiles>(sm, params, lane, token, block);
```

Replace arithmetic operands and calls at their existing locations:

```cuda
// In qk_tile, before its loops.
const Bf16* key = kv_buffer(params, Buffer) + Tile * kTileElems;

// QK inner loop; preserve both FMA accumulators and their order.
p[i] = fmaf(load_operand(q + row_index(row, lane) * kWidth + k),
            __bfloat162float(key[kv_index(col, k)]), p[i]);
p[i + 1] = fmaf(load_operand(q + row_index(row, lane) * kWidth + k),
                __bfloat162float(key[kv_index(col + 1, k)]), p[i + 1]);

// PV inner loop with restored const Bf16* v parameter.
const int base = (col / kTile) * kTileElems;
o[i] = fmaf(load_operand(s + prob_index(row_index(row, lane), k)),
            __bfloat162float(v[base + kv_index(k, col % kTile)]), o[i]);
o[i + 1] = fmaf(load_operand(s + prob_index(row_index(row, lane), k)),
                __bfloat162float(v[base + kv_index(k, col % kTile + 1)]), o[i + 1]);

// Group 0 local, then peer; retain their intervening operations/barriers.
pv_smem(local_prob, kv_buffer(params, 0), o);
pv_smem(prob, kv_buffer(params, 1), o);
// Group 1 local, then peer; retain their intervening operations/barriers.
pv_smem(prob, kv_buffer(params, 1) + kHalfTiles * kTileElems, o);
pv_smem(prob0, kv_buffer(params, 0) + kHalfTiles * kTileElems, o);
```

Add the host scratch allocation and append its pointer to Params:

```cuda
const auto kv_tiles = torch::empty(
    {kTokens * (kHeads / kTile), kKvBuffers, kKvTileElems}, q.options());

// Final two Params aggregate members, in order:
reinterpret_cast<Bf16*>(probabilities.data_ptr<at::BFloat16>()),
reinterpret_cast<Bf16*>(kv_tiles.data_ptr<at::BFloat16>())
```

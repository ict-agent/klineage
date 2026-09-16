---
skill_id: sparse-mla.shared-kv-tile-placement
intent: Reuse gathered KV tiles through CTA-shared memory.
preconditions:
- 'Data types: no arithmetic dtype restriction; moving the tiles must preserve operand
  bits and required arithmetic and rounding order because only storage placement changes.'
- 'Layout: known tile indexing, unambiguous writer ownership, and addresses aligned
  for the chosen element transfers must supply every QK/PV reader the same element;
  otherwise relocation changes operands or creates conflicting writes.'
- 'Storage: gathered KV tiles currently occupy global scratch private to one CTA,
  and all tile consumers are in that CTA; block-shared memory cannot directly supply
  other CTAs.'
- 'Pipeline: source values remain stable during gathering, and all CTA threads can
  rendezvous after tile stores and after the last tile read; these boundaries establish
  reader visibility and prevent premature buffer reuse.'
- 'Hardware: CUDA block-shared memory and CTA barriers, with capacity for all simultaneously
  live KV elements plus retained shared state and alignment padding; otherwise the
  relocated allocation or its ordering is unsupported.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Place cooperatively gathered KV tiles in CTA-shared memory so QK and PV reuse
on-chip operands. Restore this in `solution/attention.cuh`,
`solution/hopper.cuh`, and `solution/kernel.cu`. Change storage and address-space
handling only; retain gathering, indexing, arithmetic, and consumer scheduling.

## Replay

1. In `attention.cuh`, prepend `Bf16 kv[kKvBuffers][kKvTileElems];` to `Shared`.
   Add `sizeof(Shared::kv)` to `kSharedBytes`; retain its size assertion.
2. Replace every `kv_buffer(params, Buffer)` with `sm.kv[Buffer]` in `qk_tile`
   and `copy_tiles`. Replace every `kv_buffer(params, 0)` and
   `kv_buffer(params, 1)` in `consume` with `sm.kv[0]` and `sm.kv[1]`.
   Preserve all offsets, including the second consumer's
   `kHalfTiles * kTileElems` value-column offset. Remove `kv_buffer` and the
   `Params::kv_tiles` field after replacing their uses.
3. In `hopper.cuh`, add `shared_addr` as shown below. In `copy_kv`, replace
   only the destination store with the shown shared-memory store and address
   conversion. Keep scalar volatile global source loads, scalar volatile stores,
   loop unrolling, and invalid-row zero fill. Do not introduce vector transfers,
   async copies, descriptor prefetch, or cache-policy changes.
4. In the host `kernel` wrapper, remove the `kv_tiles` allocation and its final
   `Params` initializer argument. Keep the global probability allocation. The
   existing dynamic-shared-memory attribute and launch both use `sizeof(Shared)`;
   this automatically increases the allocation. Keep the launch grid, block,
   cluster, caller device/stream, tensor checks, and public ABI unchanged.
5. Preserve both `sync<Barrier::Stage, kThreads>()` calls in each `consume`
   iteration. Group 0 completes all scalar tile stores before the first barrier;
   both groups then read the published tiles. Both groups finish QK/PV before
   the second barrier permits overwriting either tile. Keep all probability,
   maximum, sum, local-group, and warp-exchange barriers unchanged.

The layout is unchanged: within width tile `t`, gathered row `r` and column `c`
occupy `t * kTileElems + kv_index(r, c)`. Both the writer and scalar QK/PV readers
already implement that map. Invalid indices still suppress global loads and
write zero bits; `Shared::valid` still masks their scores to negative infinity.
Preserve each reduction's existing operation order, BF16 probability conversion,
FP32 accumulators, output rounding, maximum/LSE formulas, and scale constants.

Check with the existing evaluator and inspect compiled QK/PV operand reads and
`copy_kv` stores for shared-memory accesses. Keep evidence outside this card.

## Example configuration

The problem is sparse MLA prefill with 8192 tokens, 128 heads, QK width 576,
value width 512, and 2048 selected indices. Inputs and output use BF16; indices
use int32; arithmetic accumulators, maximum, and LSE use FP32. Preserve
`kScale = 0.1352337788608801f`, base-2 softmax scaling, BF16 intermediate
probability rounding, and the supplied oracle and tolerances.

Each of 16384 CTAs owns one token and 64 heads. Keep 256 threads arranged as two
128-thread consumer groups and a one-CTA cluster. Group 0 also gathers both KV
tiles. Its 16 copy groups contain 8 threads each; each thread transfers 8 BF16
elements per tile row, over 4 rows and 9 width tiles. Keep the four `copy_tiles`
calls and their order. Each lane retains 32 score and 128 output registers;
output groups own the first and second 256 value columns. Group 0's QK visits
width tiles 0–8; group 1 visits 4–8 followed by 0–3. Preserve those orders.

Keep `kKvBuffers = 2`, `kKvTileElems = 64 * 576`, and the existing unswizzled
core layout: `kv_index(r,c) = (c/8)*64*8 + r*8 + c%8`. Each buffer occupies
73728 bytes. Relocation removes the 2415919104-byte global KV allocation and adds
147456 shared bytes per CTA: total shared storage changes from 1920 to 149376
bytes, including the retained masks, maxima, sums, and peer-exchange scratch.
The NVIDIA SM90 target supports this opt-in allocation. Keep all compiler flags,
scalar FMA loops, register tiling, global Q/probability placement, probability
layout, online softmax, split consumers, and scalar shared-statistics exchanges.
These are replay settings and retained mechanisms, not technique prerequisites.

# Precondition

- Data types: no arithmetic dtype restriction. The relocation copies operand
  representations without changing arithmetic or rounding. Changing operand bits
  or the required operation order would change results rather than just storage.
- Layout: tile indexing and writer ownership must be known and unambiguous, and
  addresses must satisfy the chosen element-transfer alignment. Every QK/PV
  reader must receive its original element; mismatched maps change operands,
  while conflicting writers create races. No particular core layout is required.
- Storage: gathered tiles reside in global scratch owned by one CTA, and every
  consumer belongs to that CTA. Shared-memory scope cannot directly support a
  consumer in another CTA.
- Pipeline: original source values remain stable while gathering. All CTA threads
  can reach boundaries after stores and after the final reads. The first boundary
  makes complete tiles visible; the second completes readers before buffer reuse.
  Without either boundary, readers can observe incomplete or overwritten data.
- Hardware: the target supplies CUDA block-shared storage and CTA barriers. For
  `B` simultaneously live tiles of `R * D` elements, capacity must cover
  `B * R * D * sizeof(element) + retained_shared_bytes + alignment_padding`.
  Otherwise the allocation cannot launch or the required ordering is unavailable.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These fragments replace the named storage declarations and accesses; surrounding
loops, scalar source loads, masks, and barriers stay unchanged. Apply every
replacement listed in Replay, including all four PV calls and host scratch removal.

## Before

```cuda
// attention.cuh: Params contains Bf16* kv_tiles; Shared has no KV field.
__device__ __forceinline__ Bf16* kv_buffer(const Params& params, int buffer) {
    return params.kv_tiles + (blockIdx.x * kKvBuffers + buffer) * kKvTileElems;
}

// qk_tile:
const Bf16* key = kv_buffer(params, Buffer) + Tile * kTileElems;

// copy_tiles:
Bf16* dst = kv_buffer(params, Buffer) + kv_index(group, column);

// consume: representative local PV; replace all four call sites.
pv_smem(local_prob, kv_buffer(params, 0), o);

// hopper.cuh: copy_kv, inside the unchanged scalar loop.
asm volatile("st.volatile.global.b16 [%0], %1;"
             :: "l"(dst + i), "h"(value) : "memory");
```

## After

```cuda
// attention.cuh: prepend this field to Shared; remove Params::kv_tiles
// and kv_buffer. Add sizeof(Shared::kv) to kSharedBytes.
Bf16 kv[kKvBuffers][kKvTileElems];

// qk_tile:
const Bf16* key = sm.kv[Buffer] + Tile * kTileElems;

// copy_tiles:
Bf16* dst = sm.kv[Buffer] + kv_index(group, column);

// consume: representative local PV; retain every column offset.
pv_smem(local_prob, sm.kv[0], o);

// hopper.cuh: add this address-space helper before copy_kv.
__device__ __forceinline__ uint32_t shared_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// copy_kv, inside the unchanged scalar loop.
asm volatile("st.volatile.shared.b16 [%0], %1;"
             :: "r"(shared_addr(dst + i)), "h"(value) : "memory");
```

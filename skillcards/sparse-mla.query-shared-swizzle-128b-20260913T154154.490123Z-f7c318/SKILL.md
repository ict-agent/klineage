---
skill_id: sparse-mla.query-shared-swizzle-128b
intent: Reduce shared-memory bank conflicts with a 128-byte query-tile swizzle.
preconditions:
- 'Data types: no additional arithmetic or dtype restriction; the permutation must
  preserve bits, and address calculations must use the actual element size.'
- 'Layout: query writers and all shared-memory readers have known, adjustable indexing;
  contiguous 16-byte atoms must form complete or padded 8-row by 128-byte swizzle
  blocks. Descriptor addresses and strides must be 16-byte representable, with a known
  1024-byte swizzle phase, or the reader selects different data.'
- 'Storage: query tiles already reside in CTA-shared memory with known ownership and
  lifetime; every reader of those tiles must accept the corresponding layout, since
  the permutation changes their physical addresses.'
- 'Pipeline: query writes must complete and become visible to the consumer proxy before
  reads; readers must finish before any overwrite. Preserve participating barriers
  and fences so the layout change introduces no incomplete or stale reads.'
- 'Hardware: the shared-memory consumer must support 128-byte swizzled descriptors;
  otherwise it cannot decode the new layout. The permutation requires no additional
  shared-memory capacity.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Restore the query tile's 128-byte shared-memory swizzle. Update its writer,
QK descriptor, and descriptor step together. The permutation redistributes
16-byte atoms across shared-memory banks without changing values or arithmetic.
KV retains its independent swizzle throughout.

In `solution/attention.cuh`, change `load_query` to use the existing `swizzle`
helper. In `qk_tile`, build the query descriptor with the existing `desc_k` and
advance it by `step * 2`, matching the key descriptor. Keep the key path unchanged.
Then remove the unused `kQDesc`, `kQMmaK`, `kQStep`, `q_index`, and `desc_q`
from `solution/hopper.cuh`. Keep `prob_index`, `kProbDesc`, and `kProbStep`.

For element size E and T = 16/E, the preceding interleaved tile address is
`(row/8)*8*kTile + (col/T)*8*T + (row%8)*T + col%T` elements.
The restored helper uses `offset = row*kTile + col` and
`offset ^ ((offset & (7 << 6)) >> 3)` for this example's element size and tile.
Equivalently, within an 8-row by 128-byte block, byte bits 7:9 XOR into
bits 4:6. Each 16-byte atom remains contiguous. These are alternative physical
layouts; changing only the descriptor's swizzle bits is insufficient.

The existing `kKDesc` selects swizzle mode 1 in bits 63:62, leading offset 1,
and stride offset 64, in 16-byte descriptor units. Reusing it for Q restores
128-byte rows and the 1024-byte stride between eight-row groups. The QK K-step
returns from 256 bytes in the interleaved layout to 32 bytes in the swizzled
layout. Both traverse the same 16 logical reduction elements per instruction.
See NVIDIA's [WGMMA layouts and descriptors](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout).

Retain the scalar global Q loads and their thread ownership. Each write still
owns exactly one query element. Retain `shared_fence()` after these writes and
the CTA barrier in `attention` before consumption. Q stays resident until the
CTA finishes; the optimization introduces no additional write or reuse point.
Keep every WGMMA fence, commit, wait, and register fence. No launch change is
needed. The fixed workload has complete query tiles; do not add early exits
that bypass the CTA barrier. For a padded adaptation, write every padding
atom before the consumer can read it.

Preserve QK instruction order, FP32 accumulation, softmax arithmetic and scale,
probability BF16 rounding, output BF16 rounding, masking, and output ownership.
Preserve the complete problem, ABI, caller stream, and compile flags. Validate
with the existing Kernel evaluator on that problem; inspect generated query
addresses and descriptor operands to confirm the restored layout.

## Example configuration

Preserve these replay settings:

- CUDA, `nvidia-sm90a-cuda13`, compiled for `sm_90a`.
- Tokens 8192, heads 128, QK width 576, value width 512, selected indices 2048.
  Q is contiguous `[8192,128,576]` BF16 with byte strides `(147456,1152,2)`;
  its shared tiles contain 64 heads by 64 QK elements.
- One CTA per token and 64-head slice: grid `(16384,1,1)`, block `(384,1,1)`,
  cluster `(1,1,1)`. Consumer group 0's 128 threads load Q, with
  `i = lane; i < kTile*kWidth; i += kWarpgroup`.
- Nine Q tiles occupy 73728 bytes at the start of dynamic shared memory;
  successive tile bases differ by 8192 bytes. Their swizzle base phase is zero.
  The dynamic shared symbol starts at address zero; retain this placement when
  using the existing zero-base-offset descriptor. Total shared allocation is
  231376 bytes; the transformation changes neither allocation nor alignment.
- Retain two KV buffers, one producer warpgroup, two consumer warpgroups,
  four transfer groups per iteration, 16-byte `cp.async.ca` copies, all
  producer/consumer barriers, and QK/PV overlap.
- Retain BF16 Q/K/V and probabilities, FP32 accumulators and statistics,
  QK `m64n64k16`, PV `m64n256k16`, 32 score registers and 128 output registers
  per consumer thread, and 32 selected-key tiles processed in pairs.
- Retain probability interleaving and RoPE-tile reuse, shared peer reductions,
  online softmax with scale `0.1352337788608801f`, and direct output stores.
  Do not change launch bounds or compiler register settings.

# Precondition

- Data types: no additional arithmetic or dtype restriction. This transformation
  only relocates bits. Index and descriptor strides must account for the actual
  element size so the permutation preserves each value's representation.
- Layout: every query writer and shared reader must have known, adjustable
  indexing. Contiguous 16-byte atoms must cover complete or padded 8-row by
  128-byte blocks; otherwise the XOR can select absent data. Descriptor bases
  and strides must be representable in 16-byte units, and the 1024-byte swizzle
  phase must be known. A mismatched phase or stride addresses the wrong atom.
- Storage: query tiles already occupy CTA-shared memory with known ownership
  and lifetime. All readers of a changed tile must accept the matching layout;
  a reader using the old addresses would consume different values.
- Pipeline: producer writes must finish and become visible to the consumer's
  memory proxy before reads. All readers must finish before any overwrite.
  Preserve the existing participating barriers and fences to enforce both
  edges; no particular stage count is required by the permutation.
- Hardware: the shared-memory consumer must decode 128-byte swizzled
  descriptors. A consumer lacking that mode cannot interpret the new addresses.
  The in-place permutation requires no additional shared-memory capacity.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The three excerpts replace the query store in `load_query`, the query
initialization in `qk_tile`, and that function's `qk_mma` call, respectively.
Keep their loops and surrounding code. The After helpers already exist for KV.
Remove the five unused Q-specific declarations listed above after replacement.

## Before

```cuda
// load_query: col selects a QK tile; row selects a head.
sm.q_o[(col / kTile) * kTileElems + q_index(row, col % kTile)] = src[i];

// qk_tile: unswizzled query descriptor.
const uint64_t q = desc_q(sm.q_o + Tile * kTileElems);

// qk_tile: step advances 16 logical QK elements in each operand.
qk_mma(desc_step(q, step * kQStep), desc_step(k, step * 2),
       p, step == 0 ? Mode : Accum::Add);
```

## After

```cuda
// load_query: match Q's writer to its swizzled reader.
sm.q_o[(col / kTile) * kTileElems + swizzle(row, col % kTile)] = src[i];

// qk_tile: Q and K now use the same descriptor layout.
const uint64_t q = desc_k(sm.q_o + Tile * kTileElems);

// qk_tile: preserve the logical reduction order with swizzled strides.
qk_mma(desc_step(q, step * 2), desc_step(k, step * 2),
       p, step == 0 ? Mode : Accum::Add);
```

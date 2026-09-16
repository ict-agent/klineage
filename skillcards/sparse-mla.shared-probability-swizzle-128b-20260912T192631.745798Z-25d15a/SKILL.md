---
skill_id: sparse-mla.shared-probability-swizzle-128b
intent: Store shared probability tiles with hardware-decoded 128-byte swizzling.
preconditions:
- 'Data types: no arithmetic dtype or value-range restriction; preserve value bits
  and required conversions. Element size must divide the 16-byte chunks permuted by
  this mapping, so elements do not cross chunk boundaries.'
- 'Layout: known complete writer ownership and compatible reader indexing; tiles must
  admit complete 8-row by 128-byte swizzle atoms, with padding if needed. Tile bases
  must be 16-byte aligned and their 1024-byte pattern phase known and representable;
  otherwise writer addresses and reader descriptors disagree.'
- 'Storage: probabilities are published into CTA-shared tiles accessible to every
  consumer. Each allocation must contain all permuted addresses, including any padding,
  because the reader cannot fetch outside that tile.'
- 'Pipeline: probability production must finish before publication; every participating
  reader must observe completed stores before consuming them. All asynchronous reads
  must finish before rewriting or repurposing the tile, including aliased storage;
  otherwise the permutation exposes stale or overwritten values.'
- 'Hardware: CTA shared memory and a matrix reader supporting 128-byte swizzle descriptors.
  Simultaneously live allocations, including padded probability tiles, must fit per-CTA
  shared capacity; otherwise this addressing or allocation is unavailable.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Replace no-swizzle core packing of shared probability tiles with 128-byte XOR
swizzling. Writers place each value at its permuted address; the PV matrix reader
decodes that address through its descriptor. This restores the hardware-supported
row layout without changing probability values or arithmetic.

In `solution/attention.cuh`, update both stores in `save_prob` from `prob_index`
to the existing `swizzle`. In `pv_smem`, replace `desc_prob(s)` with `desc_k(s)`
and change the A descriptor increment from `step * kProbStep` to `step * 2`.
Retain B's `desc_mn(v)`, its `step * 128` increment, and every MMA instruction.

In `solution/hopper.cuh`, delete the now-unused `prob_index`, `desc_prob`, and
constants `kDescUnitBytes`, `kCoreRows`, `kCoreElems`, `kCoreTileElems`,
`kProbMmaK`, `kProbStep`, and `kProbDesc`. Keep `swizzle`, `desc_k`, `desc_mn`,
and `desc_step`: Q/K/V still use them.

## Layout and ownership

For element size `e`, a no-swizzle K-major core contains eight rows of
`16/e` elements. The current BF16 address is
`(row/8)*8*kTile + (col/8)*64 + (row%8)*8 + col%8`.
The forward layout uses the existing element index
`offset = row*kTile + col; offset ^ ((offset & (7 << 6)) >> 3)`.
In byte coordinates within an aligned swizzle atom, this permutes 16-byte chunks
by XORing address bits 7–9 into bits 4–6.

Each consumer lane retains its score-fragment ownership. For loop variables
`row` and `i`, `row_index(row,lane)` gives the logical row and
`8*(i/4) + (lane%4)*2` gives the first column; the lane writes that column and
its successor. All logical coordinates have one writer. There is no new thread
mapping, staging pass, or launch change.

The same helper covers both probability buffers and every publication:
`sm.kv[0] + (kKeyTiles-1)*kTileElems` holds group 0's probabilities, while
`sm.prob` holds group 1's. Change every PV A descriptor through `pv_smem` so
local and peer readers agree with both writers. The aliased K0 RoPE tile keeps
its original K layout during QK and receives the probability layout only after
its QK readers finish.

## Ordering and bounds

Keep `shared_fence()` and the `Local0`/`Local1` barriers after local stores.
Keep `Prob0`/`Prob1` publication handshakes, WGMMA commits/waits, and all
`free0`/`free1`/`ready0`/`ready1` arrivals and waits. Group 0 completes its
local PV before rescaling and rewriting P0. Peer PV reads finish before the
corresponding buffer is released; the existing KV reuse handshakes also prevent
the next iteration from overwriting P1 prematurely. Changing the layout does
not replace these visibility and lifetime guarantees.

The fixed probability tiles are complete, so their stores need no new bounds
checks. Keep sparse-index validity checks, invalid-key zero-fill, score masking,
and output ownership. Preserve both probability BF16 round-to-nearest conversions,
QK/PV accumulation order, online-softmax arithmetic, and final BF16 rounding.

The layout and descriptor interpretation follow NVIDIA's
[PTX shared-matrix layout specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout).

## Example configuration

Retain the supplied workload: 8192 tokens, 128 heads, QK width 576, value width
512, TOPK 2048, and one KV head. Q/KV/output are BF16; logits, LSE, reductions,
and accumulators are FP32; indices are int32. Keep scale `0.1352337788608801f`,
its existing log2 conversion, and the complete problem/oracle.

Retain 64-by-64 probability tiles, 384 CTA threads, two 128-thread consumers,
one 128-thread producer, and 16384 CTAs in singleton clusters. Each consumer
lane retains two logical rows, 32 score registers, and 128 output accumulators.
Keep nine resident Q tiles loaded by TMA, two KV buffers, all four asynchronous
KV transfer groups, shared probability reuse, shared reduction exchanges,
Q/K/V swizzles, cache policies, and direct global output stores. Keep QK
`m64n64k16` and PV `m64n256k16` BF16 WGMMA with FP32 accumulation.

Each probability allocation is 8192 bytes. P0 starts at byte 139264 within
`Shared`, and P1 at byte 221184. Both retain a zero swizzle pattern phase.
Keep `sizeof(Shared) == 231376` and the existing dynamic shared launch;
no allocation or buffer lifetime changes. Global contiguous row strides remain
`576*sizeof(Bf16)` for KV and `512*sizeof(Bf16)` for output.

Before replay, P uses no-swizzle descriptor fields LBO=128 bytes, SBO=1024 bytes,
and advances 256 bytes per K16 instruction. After replay, `desc_k` selects
128-byte swizzling, LBO's unused field is encoded as 1, SBO remains 1024 bytes,
and the start advances 32 bytes per K16 instruction. Descriptor offsets are
encoded in 16-byte units. B's 2048-byte step is unchanged.

Keep the pybind11 destination-passing ABI, caller device/stream, launch bounds,
and all compile flags, including SM90a targeting, fast math, and register-usage
level 10. Rebuild from final source files. Use the existing Kernel evaluator on
the complete problem, then inspect PV descriptors to confirm their swizzle
mode is restored. Keep measurements outside this card.

# Precondition

- Data types: no arithmetic dtype or value-range restriction. This transformation
  permutes stored bits and preserves existing conversions. Element size must divide
  the 16-byte chunks used by the permutation; otherwise an element could straddle
  independently moved chunks.
- Layout: every logical element has known writer ownership, and all readers can
  adopt matching indexing. Tiles admit complete eight-row by 128-byte swizzle atoms,
  padding where needed. Bases are 16-byte aligned and the 1024-byte pattern phase is
  known and representable. These are address/descriptor constraints: violating them
  makes the consumer fetch different bytes from those published by the writer.
- Storage: probabilities already pass through CTA-shared tiles visible to all
  consumers. Allocations contain every permuted address, including padding; an
  otherwise valid permutation must not escape the tile's allocated storage.
- Pipeline: producers finish computing values before stores; the designated
  participants establish store completion and reader visibility before consumption.
  All asynchronous readers finish before the tile is rewritten or its allocation
  is reused, including aliases. These prevent stale reads and overwrite races;
  the technique imposes no particular stage count.
- Hardware: CTA shared memory and a matrix reader supporting 128-byte swizzle
  descriptors are needed to consume the writer's permutation directly. Per-CTA
  capacity must cover all simultaneously live allocations, including any padded
  probability tiles; the permutation itself adds no staging buffer.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The functions below are the only changed consumers/writers in
`solution/attention.cuh`. Remove the obsolete probability-only helpers/constants
from `solution/hopper.cuh` as listed above. All call sites stay unchanged.

## Before

```cuda
__device__ __forceinline__ void pv_smem(
    const Bf16* s, const Bf16* v, float (&o)[kOutputRegs]) {
    const uint64_t a = desc_prob(s);
    const uint64_t b = desc_mn(v);
    reg_fence(o);
    mma_fence();
#pragma unroll
    for (int step = 0; step < kTile / 16; ++step)
        pv_shared(desc_step(a, step * kProbStep), desc_step(b, step * 128), o);
    reg_fence(o);
}

__device__ __forceinline__ void save_prob(const Bf16* p, Bf16* dest, int lane) {
    constexpr int kLanesPerRow = 4;
    constexpr int kPairElems = 2;
    constexpr int kColStep = kLanesPerRow * kPairElems;
    constexpr int kRegsPerStep = kRowsPerThread * kPairElems;

    // Store probabilities in unswizzled WGMMA cores.
#pragma unroll
    for (int row = 0; row < kRowsPerThread; ++row) {
        const int real_row = row_index(row, lane);
#pragma unroll
        for (int i = row * kPairElems; i < kScoreRegs; i += kRegsPerStep) {
            const int col = kColStep * (i / kRegsPerStep) + (lane % kLanesPerRow) * kPairElems;
            dest[prob_index(real_row, col)] = p[i];
            dest[prob_index(real_row, col + 1)] = p[i + 1];
        }
    }
}
```

## After

```cuda
__device__ __forceinline__ void pv_smem(
    const Bf16* s, const Bf16* v, float (&o)[kOutputRegs]) {
    const uint64_t a = desc_k(s);
    const uint64_t b = desc_mn(v);
    reg_fence(o);
    mma_fence();
#pragma unroll
    for (int step = 0; step < kTile / 16; ++step)
        pv_shared(desc_step(a, step * 2), desc_step(b, step * 128), o);
    reg_fence(o);
}

__device__ __forceinline__ void save_prob(const Bf16* p, Bf16* dest, int lane) {
    constexpr int kLanesPerRow = 4;
    constexpr int kPairElems = 2;
    constexpr int kColStep = kLanesPerRow * kPairElems;
    constexpr int kRegsPerStep = kRowsPerThread * kPairElems;

    // Publish probabilities in the reader's 128-byte swizzle.
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

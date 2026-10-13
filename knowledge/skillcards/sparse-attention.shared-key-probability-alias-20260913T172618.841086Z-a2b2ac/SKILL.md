---
skill_id: sparse-attention.shared-key-probability-alias
intent: Reuse a dead shared key tile for probability storage.
preconditions:
- 'Data types: no additional dtype or arithmetic requirement; redirecting shared addresses
  must preserve stored bits and every conversion and reduction.'
- 'Layout: existing probability writers and readers have known, matching element offsets,
  and the donor base meets their access alignment; rebasing otherwise changes elements
  or misaligns accesses.'
- 'Storage: separate operand and probability arrays already reside in the same CTA
  shared memory, with a donor region no operand consumer needs after QK; the probability
  byte span must fit the donor region to avoid corrupting live storage.'
- 'Pipeline: all donor readers must finish before the first probability store; each
  probability version must be published before local or peer reads, all local readers
  must finish before rescaling overwrites it, and all probability readers must finish
  before the next operand load reuses it.'
- 'Hardware: no additional hardware or capacity beyond the existing CTA shared memory
  and participating-thread barriers; aliasing reduces allocation and relies on those
  barriers to delimit reuse.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Reuse a dead shared key tile for group 0 probabilities. In
`solution/attention.cuh`, remove `Shared::prob0` and redirect all three
`sm.prob0` references in `consume` to
`sm.kv[0] + (kKeyTiles - 1) * kTileElems`. Reduce `kSharedBytes` by
`kTileElems * sizeof(Bf16)`. Keep every other field and operation unchanged.

The last tile of `kv[0]` holds the key-only RoPE dimensions. After group 0
finishes QK, neither group's PV reads that tile: their value tiles precede it.
Store probabilities there using `save_prob` and its existing `prob_index` layout.
Both local and peer `pv_smem` calls must read that same probability base.
The key layout `kv_index` need not match `prob_index`: their lifetimes differ.
The donor is a byte region, not a simultaneous alternate view of live keys.

Keep the existing ordering:

- `qk_left`, `qk_right`, then `Local0` with all group 0 threads finish donor reads.
- `save_prob` followed by `Local0` publishes probabilities for local PV.
- `Max1`, reached after local PV, finishes its readers before rescaled probabilities
  overwrite the donor. Retain the existing maximum exchange and rescaling.
- `Prob0` publishes rescaled probabilities to group 1. Its `Prob1` arrival follows
  the peer PV; group 0 waits for that completion.
- The final `Stage` barrier includes both groups before the next `load_kv`
  overwrites the donor with keys. The initial `Stage` publishes those keys.

There are no new launches or barriers. `solution/kernel.cu` already uses
`sizeof(Shared)` for both the shared-memory opt-in and launch allocation; leave
it unchanged. Retain its caller device, caller stream, tensor checks, and ABI.
Keep invalid-index zero fills and masks. The donor offset is in bounds because
it is the last full key tile; probability accesses span exactly one tile.
Address changes introduce no arithmetic or rounding changes.

## Example configuration

The fixed workload has 8,192 tokens, 128 heads, QK width 576, value width 512,
and 2,048 selected indices per token. Query and KV elements and stored
probabilities are BF16; index elements are int32, reductions and accumulators
are FP32. Preserve all FMA traversal, online-softmax grouping, BF16 probability
rounding, output rounding, scale `0.1352337788608801f`, and max/log-sum-exp handling.

Each CTA owns one token and 64 heads. The launch has 16,384 CTAs, 256 threads
in two 128-thread consumer groups, and a one-CTA cluster. Each lane retains two
rows, 32 score registers, and 128 output accumulators with the existing
four-lane row ownership. Retain both KV buffers and the peer `Shared::prob` tile.
Process 32 key tiles in 16 pairs. Keep all transfer groups, scalar global copies,
shared KV/probability layouts, reductions, barriers, and compiler flags unchanged.

Each tile has 64 x 64 BF16 elements (8,192 bytes). The donor starts 65,536 bytes
into `kv[0]`, covering dimensions 512 through 575. The separate `prob0` starts
at byte 155,648. Removing it reduces the shared allocation from 165,760 to
157,568 bytes. Retain `Shared` alignment of 16 bytes and the size assertion.
The supplied target is NVIDIA SM90 with CUDA 13. No tensor-core operation is
introduced by this change.

# Precondition

- Data types: no additional dtype or arithmetic requirement. This is an address
  substitution, so stored bits, conversions, and reduction arithmetic must remain
  identical; BF16 is an example setting, not an aliasing prerequisite.
- Layout: existing probability writers and readers must have known, matching
  element offsets, and the donor base must meet their access alignment. Otherwise
  rebasing misaligns accesses or changes element identity. The expired operand
  and probability layouts may differ.
- Storage: distinct operand and probability arrays already occupy shared memory
  within one CTA. An operand region must become dead after QK, including for
  later PV readers; otherwise aliasing destroys live values. The probability
  byte span must fit that region or it overwrites neighboring live storage.
- Pipeline: finish every donor read before the first probability write. Publish
  each probability version before any local or peer consumer reads it. Complete
  local PV readers before rescaling overwrites their version, and finish all
  probability readers before the next operand load. These dependencies prevent
  uninitialized reads and reuse races; they do not require a fixed stage count.
- Hardware: no additional hardware or capacity is required beyond the existing
  CTA shared memory and barriers among participating threads. Aliasing reduces
  the allocation; those barriers delimit safe reuse of the shared storage.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These are focused excerpts from `solution/attention.cuh`. Apply all replacements;
retain intervening computation, other `Shared` members, and every barrier.

## Before

```cuda
constexpr int kSharedBytes = 156544 + kConsumerThreads * sizeof(float) +
                             kTileElems * sizeof(Bf16);

// Separate member in Shared, immediately after prob.
Bf16 prob0[kTileElems];

// consume<0>: first probability publication and local PV.
Bf16* local_prob = sm.prob0;
save_prob(s, local_prob, lane);
sync<Barrier::Local0, kWarpgroup>();
pv_smem(local_prob, sm.kv[0], o);

// consume<0>: publication after the existing Max1 and rescaling.
save_prob(s, sm.prob0, lane);
arrive<Barrier::Prob0>();
sync<Barrier::Prob1>();

// consume<1>: peer PV after its local PV.
sync<Barrier::Prob0>();
pv_smem(sm.prob0,
        sm.kv[0] + kHalfTiles * kTileElems, o);
arrive<Barrier::Prob1>();
```

## After

```cuda
constexpr int kSharedBytes = 156544 + kConsumerThreads * sizeof(float);

// Delete Shared::prob0; retain every other member.

// consume<0>: reuse the completed key-only tile.
Bf16* local_prob = sm.kv[0] + (kKeyTiles - 1) * kTileElems;
save_prob(s, local_prob, lane);
sync<Barrier::Local0, kWarpgroup>();
pv_smem(local_prob, sm.kv[0], o);

// consume<0>: publish the rescaled version in the same donor region.
save_prob(s, sm.kv[0] + (kKeyTiles - 1) * kTileElems, lane);
arrive<Barrier::Prob0>();
sync<Barrier::Prob1>();

// consume<1>: read the relocated probabilities after publication.
sync<Barrier::Prob0>();
pv_smem(sm.kv[0] + (kKeyTiles - 1) * kTileElems,
        sm.kv[0] + kHalfTiles * kTileElems, o);
arrive<Barrier::Prob1>();
```

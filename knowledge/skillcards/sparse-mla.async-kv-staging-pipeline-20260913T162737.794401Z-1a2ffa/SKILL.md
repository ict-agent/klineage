---
skill_id: sparse-mla.async-kv-staging-pipeline
intent: Overlap KV transfers and attention computation with asynchronous copies and
  per-region handoffs.
preconditions:
- 'Data types: no additional arithmetic dtype requirement; asynchronous copies move
  bits and must preserve zero-fill and the existing arithmetic order.'
- 'Layout: for instruction-supported copy width B bytes, global and shared addresses
  are B-byte aligned and valid sources expose B bytes; padding permits zero-fill,
  and known producer/reader ownership identifies each protected buffer region.'
- 'Storage: source operands already reside in global memory and reusable destinations
  in CTA shared memory; all readers protected by a handoff must belong to that CTA.'
- 'Pipeline: producer and consumer work can progress separately with known participating
  threads and copy/read completion boundaries; copies must become visible before reads,
  and buffers and masks must remain intact through their last reader, enabling safe
  counted handoffs.'
- 'Hardware: support for cp.async.ca.shared.global, cp.async.mbarrier.arrive.noinc,
  shared mbarrier initialization and parity waits, and reader-completion waits; per-CTA
  shared capacity must cover resident operands, reusable buffers, scratch, and barrier
  state simultaneously.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Overlap sparse KV staging with attention computation. Replace the synchronous
vector transfers and the two CTA-wide `Barrier::Stage` rendezvous with per-region
ready/free barriers. Producers refill a region as soon as its last consumer
finishes. Keep the two selected-key tiles, their shared layouts, and both consumer
warpgroups. Also restore Group 0's next-iteration QK overlap and Group 1's two
outstanding PV groups; their completion points release the corresponding regions.

Edit `solution/hopper.cuh` and `solution/attention.cuh`. The snippets below are
replacement fragments at the named locations, not complete translation units.
All unmentioned arithmetic, declarations, loops, stores, and synchronization stay.

## Example configuration

Preserve `TOKENS=8192`, `HEADS=128`, `QK_DIM=576`, `VALUE_DIM=512`, `TOPK=2048`.
The tensors are contiguous BF16 Q/KV/output, INT32 indices, and FP32 maxima/LSE.
QK/PV accumulate in FP32; probability and output conversions retain
`__float2bfloat16_rn`. Keep the softmax scale, `rescale_exp`, `output_scale`,
reduction grouping, and all existing compiler flags, including fast math.

The launch stays at 16,384 CTAs, 384 threads per CTA, one CTA per cluster, on the
caller stream. Each CTA owns one token and 64 heads. Consumer groups 0 and 1 each
have 128 threads; producer group 2 has 128 threads. Each consumer keeps 32 score
and 128 output registers per thread. Group 0 owns output columns 0–255; Group 1
owns 256–511. Preserve the existing `sm_90a` build and register-usage setting.

Each iteration handles two 64-row key tiles; `block` advances by two through 32
tiles. Q remains resident in `q_o`. Preserve `q_index`, `kv_index`, `prob_index`,
the unswizzled WGMMA descriptors, both KV buffers, and `sm.prob`. Group 0's local
probabilities still alias KV0's last QK tile after its QK readers finish. Keep all
Max0/Max1, Prob0/Prob1, Local0/Local1, Sum, register fences, MMA fences, commits,
and probability visibility fences except the waits explicitly changed below.

Producer lanes form 16 groups of eight. A group owns four sparse rows per buffer;
its lanes cover eight adjacent BF16 elements per 16-byte vector. Preserve this
copy width for the replay. The global KV row stride
is `params.kv_stride * sizeof(Bf16)` (1152 bytes here). Shared destinations retain
`kv_index` and the existing tile/row offsets. Preserve index snapshots and
`index >= 0 && index < kTokens`; invalid indices pass source size zero, which
zero-fills the full destination vector without reading the invalid source.

Add nine eight-byte barriers before `Shared::reduction`. Change
`kSharedBytes` from `230272 + kConsumerThreads * sizeof(float)` to
`230352 + kConsumerThreads * sizeof(float)`: `sizeof(Shared)` grows from 231296 to
231376 bytes including alignment. The unchanged host launch derives both the
shared-memory opt-in and allocation from `sizeof(Shared)`.

## Replay edits and ownership

1. In `hopper.cuh`, remove `Stage = 7` from `Barrier`, preserving `Max0 = 8` and
   subsequent values. Add `kWaitTicks = 0x989680`, `Mbar`, `elect`, and `init_fence`
   as shown below. Replace only `copy_kv` with the asynchronous version. The
   synchronous helper's volatile vector load/store prevents compilation from
   recreating async copies; remove that compiler control with the helper.
2. In `Shared`, insert
   `Mbar free0[2], ready0[2], free1[2], ready1[2], mask;`
   immediately before `reduction`. Adjust `kSharedBytes` as above. In `attention`,
   initialize these objects using the snippet before `load_query`; retain the
   existing query load and following `__syncthreads()` to publish initialization.
3. In `produce`, add `int phase = 1;` before its loop. Keep index/valid snapshots.
   Replace the four copy calls with the ready/free sequence below. In the existing
   `if (lane % kCopyGroup == 0)` block, append `sm.mask.arrive()` after all validity
   stores. Replace its trailing `shared_fence()` and both Stage barriers with
   `phase ^= 1;`. Each producer thread calls each ready `cp_arrive` exactly once;
   only the 16 row-group leaders arrive on `mask`.
4. Add `int phase` immediately after `Shared& sm` to `qk_left`, `qk_right`,
   `qk_peer`, `mask_scores`, and `softmax`; forward it at their consumer calls.
   Insert readiness waits from the table below. Retain their existing QK tile
   order and commits. Both `mask_scores` and `softmax` start with
   `sm.mask.wait(phase)`.
5. In `consume`, add `int phase = 0;` before the loop and remove both Stage
   barriers. Apply the three consumer fragments below. Keep all intermediate
   maximum, probability, rescaling, and output work. Group 0 performs its initial
   QK only at `block == 0`; subsequent QK runs at the prior iteration's tail.
   Both consumer groups toggle phase exactly once per iteration. Preserve the
   final `reduce_sum`, output stores, and output WGMMA completion wait.

| QK helper | Wait placement |
| --- | --- |
| `qk_left` | Start with `sm.ready0[0].wait(phase)` before tiles 0–3. |
| `qk_right` | Start with `sm.ready0[1].wait(phase)` before tiles 4–8. |
| `qk_peer` | Start with `sm.ready1[1].wait(phase)` before tiles 4–8; insert `sm.ready1[0].wait(phase)` immediately before tile 0, then retain tiles 0–3. |

Each ready/free pair protects one KV buffer region:

| Region | Producer order | Last reader and free arrival |
| --- | --- | --- |
| KV0 tiles 0–3 | First | Group 0 local PV: `free0[0]`. |
| KV1 tiles 4–8 | Second | Group 1 local PV: `free1[1]`. |
| KV0 tiles 4–8 | Third | Group 1 remote PV: `free0[1]`; this also protects the aliased probability tile. |
| KV1 tiles 0–3 | Fourth | Group 0 remote PV: `free1[0]`. |

## Ordering and bounds

Initialize every barrier with its exact arrival count, then publish initialization
before either role uses it. The producer starts at parity one so initially free
storage is available; consumers start at zero waiting for the first copies. Each
`cp_arrive` includes the issuing thread's preceding copies in that ready barrier's
completion without increasing its expected arrival count. Consumers wait for
copy completion before QK. The mask barrier publishes generic validity stores.

Free arrivals occur only after the indicated WGMMA readers complete. In Group 0,
`wait<1>()` after committing next QK-left completes the older remote PV while
allowing QK-left to remain pending. Release KV1-left then issue QK-right; the final
iteration drains all groups. In Group 1, `wait<1>()` completes local PV before
freeing KV1-right; `wait<0>()` completes remote PV before freeing KV0-right.
These are completion counts, not permissions to overwrite an in-use operand.

The two masks are reused only after the producer passes all four free waits.
Both consumers have already read their masks before releasing their last region.
Keep the probability fences and named barriers: generic probability writes must
be visible to local and peer WGMMA readers. In particular, KV0's final tile cannot
be refilled until Group 1's remote PV finishes. No cross-CTA communication is added.

All transfer bounds and causal masks stay unchanged. Reordering overlap must not
reorder a QK accumulation, PV update, softmax maximum update, or BF16 rounding.
Use the existing evaluator with the unchanged problem. Inspect generated device
code for restored async copies and barrier handoffs; keep measurements outside
this card.

# Precondition

- Data types: no additional arithmetic dtype requirement. The transfer moves raw
  bits, including zero-fill; changing the representation or arithmetic sequence
  would change the operator instead of merely overlapping its transfers.
- Layout: for the selected instruction-supported width `B` bytes, source and
  destination addresses are `B`-byte aligned and valid sources expose `B` bytes.
  Padding permits zero-fill. These conditions make the copy instruction legal.
  Known producer and reader ownership must associate each reusable region with its complete reader set;
  otherwise its free barrier can permit an overwrite too early.
- Storage: operands already come from global memory into reusable CTA shared
  destinations. This is the address-space pair supported by the transfer. Every
  protected reader belongs to that CTA because these handoffs have CTA scope.
- Pipeline: producer and consumer work must be separable, with known participants
  and completion boundaries. All designated threads must be able to reach their
  handoffs; otherwise a counted barrier can deadlock. Existing ordering must
  expose producer completion, read visibility, and final-reader completion for
  each reusable region. Masks also remain valid through their last read. These
  boundaries permit replacing the full-CTA rendezvous with correctly initialized,
  counted phases without stale reads or early overwrites. No fixed stage count
  is required.
- Hardware: the target supports `cp.async.ca.shared.global`, its
  `cp.async.mbarrier.arrive.noinc` completion association, shared mbarrier
  initialization and parity waits, and completion waits for asynchronous readers.
  Otherwise the handoffs
  cannot establish readiness or safe reuse. Available CTA shared memory must be
  at least `bytes(resident operands) + bytes(reusable buffers) + bytes(scratch) +
  bytes(barrier state)`, including alignment, since these objects coexist.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

`hopper.cuh`: synchronous vector transfer; `Mbar`, `elect`, and `init_fence` are absent.

```cuda
__device__ __forceinline__ void copy_kv(
    const Bf16* src, Bf16* dst, int bytes) {
    uint32_t x = 0, y = 0, z = 0, w = 0;
    if (bytes != 0) {
        asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(src) : "memory");
    }
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(shared_addr(dst)), "r"(x), "r"(y), "r"(z), "r"(w) : "memory");
}
```

`produce`: four synchronous copy groups, then mask publication and two full-CTA
rendezvous. `consume` has matching Stage barriers at its loop entry and exit.

```cuda
copy_tiles<0, 0, kHalfTiles>(sm, params, lane, indices, valid);
copy_tiles<1, kHalfTiles, kKeyTiles>(sm, params, lane, indices, valid);
copy_tiles<0, kHalfTiles, kKeyTiles>(sm, params, lane, indices, valid);
copy_tiles<1, 0, kHalfTiles>(sm, params, lane, indices, valid);
// Existing leader-only validity stores occur here.
shared_fence();
sync<Barrier::Stage, kThreads>();
sync<Barrier::Stage, kThreads>();
```

`consume<0>`: unconditional initial QK and no next-iteration overlap.

```cuda
// Group 0 entry, after the Stage barrier:
qk_left(sm, p);
qk_right(sm, p);
wait<0>();

// After local PV:
commit();
wait<0>();

// After remote PV, at the Group 0 tail:
commit();
wait<0>();
```

`consume<1>`: local and remote PV complete separately; the Stage barrier at the
loop exit protects both buffers.

```cuda
pv_smem(sm.prob, sm.kv[1] + kHalfTiles * kTileElems, o);
commit();
wait<0>();
sync<Barrier::Prob0>();
pv_smem(sm.kv[0] + (kKeyTiles - 1) * kTileElems,
        sm.kv[0] + kHalfTiles * kTileElems, o);
commit();
shared_fence();
arrive<Barrier::Prob1>();
wait<0>();
```

## After

`hopper.cuh`: add barrier helpers before `shared_fence`; replace `copy_kv`.

```cuda
constexpr uint32_t kWaitTicks = 0x989680;

__device__ __forceinline__ bool elect() {
    uint32_t elected;
    asm volatile("{ .reg .pred p; elect.sync _|p, 0xffffffff; selp.u32 %0, 1, 0, p; }"
                 : "=r"(elected));
    return elected;
}

struct alignas(8) Mbar {
    uint64_t state;

    __device__ __forceinline__ void init(int arrivals) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                     :: "r"(shared_addr(this)), "r"(arrivals) : "memory");
    }

    __device__ __forceinline__ void wait(int phase) const {
        asm volatile("{ .reg .pred p; wait_loop: "
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2; "
                     "@p bra done; bra wait_loop; done: }"
                     :: "r"(shared_addr(this)), "r"(phase), "r"(kWaitTicks) : "memory");
    }

    __device__ __forceinline__ void arrive() {
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                     :: "r"(shared_addr(this)) : "memory");
    }

    __device__ __forceinline__ void cp_arrive() {
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];"
                     :: "r"(shared_addr(this)) : "memory");
    }
};

__device__ __forceinline__ void init_fence() {
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void copy_kv(
    const Bf16* src, Bf16* dst, int bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(shared_addr(dst)), "l"(src), "r"(bytes));
}
```

`attention`: insert before the unchanged query load and CTA barrier.

```cuda
const int warp = warp_index();
if (warp == 0 && elect()) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        sm.free0[i].init(kWarpgroup);
        sm.ready0[i].init(kWarpgroup);
        sm.free1[i].init(kWarpgroup);
        sm.ready1[i].init(kWarpgroup);
    }
    sm.mask.init(kCopyGroups);
    init_fence();
}
```

`produce`: use `phase = 1` before the loop, then replace its copy/publication tail.
Keep the existing validity-store loops where indicated, inside their existing
leader-only branch; append the mask arrival inside that branch.

```cuda
sm.free0[0].wait(phase);
copy_tiles<0, 0, kHalfTiles>(sm, params, lane, indices, valid);
sm.ready0[0].cp_arrive();
sm.free1[1].wait(phase);
copy_tiles<1, kHalfTiles, kKeyTiles>(sm, params, lane, indices, valid);
sm.ready1[1].cp_arrive();
sm.free0[1].wait(phase);
copy_tiles<0, kHalfTiles, kKeyTiles>(sm, params, lane, indices, valid);
sm.ready0[1].cp_arrive();
sm.free1[0].wait(phase);
copy_tiles<1, 0, kHalfTiles>(sm, params, lane, indices, valid);
sm.ready1[0].cp_arrive();

if (lane % kCopyGroup == 0) {
    // Retain both validity-store loops here.
    sm.mask.arrive();
}
phase ^= 1;
```

`consume`: use `phase = 0` before the loop; remove both Stage barriers. Add the
phase arguments and readiness waits specified above. Apply these Group 0 edits:

```cuda
// Group 0 entry:
if (block == 0) {
    qk_left(sm, phase, p);
    qk_right(sm, phase, p);
    wait<0>();
}

// Immediately after local PV:
commit();
wait<0>();
sm.free0[0].arrive();

// Immediately after remote PV, replacing the Group 0 tail:
commit();
phase ^= 1;
if (block + 2 < kBlocks) {
    qk_left(sm, phase, p);
    wait<1>();
    sm.free1[0].arrive();
    qk_right(sm, phase, p);
    wait<0>();
} else {
    wait<0>();
    sm.free1[0].arrive();
}
```

Replace Group 1's PV tail with the following. Earlier QK, masks, softmax, and
probability stores retain their order with the added phase arguments/waits.

```cuda
pv_smem(sm.prob, sm.kv[1] + kHalfTiles * kTileElems, o);
commit();
sync<Barrier::Prob0>();
pv_smem(sm.kv[0] + (kKeyTiles - 1) * kTileElems,
        sm.kv[0] + kHalfTiles * kTileElems, o);
commit();
shared_fence();
arrive<Barrier::Prob1>();

wait<1>();
sm.free1[1].arrive();
wait<0>();
sm.free0[1].arrive();
phase ^= 1;
```

---
skill_id: cuda.wgmma-register-probability-forwarding
intent: Forward probability fragments directly from registers into local WGMMA operations.
preconditions:
- 'Data types: the existing MMA types must support equivalent shared-A and register-A
  forms; forwarding must preserve operand bits, accumulator type, and reduction order
  to preserve numerical behavior.'
- 'Layout: each consuming lane must already own the A elements required by the register
  fragment, with compatible packing; wrong ownership changes the matrix product. No
  new global contiguity or alignment requirement arises because global accesses stay
  unchanged.'
- 'Storage: A must be available in consumer-owned registers before its shared-memory
  staging; otherwise this bypass needs another transfer. Shared copies required by
  other consumers must remain available.'
- 'Pipeline: all 128 threads of each WGMMA warpgroup must participate uniformly; A
  production must finish before consumption and its registers must remain live and
  unchanged until asynchronous readers complete. Retained shared readers still require
  publication and completion before buffer reuse.'
- 'Hardware: a target and toolchain supporting the matching register-A WGMMA form
  and its legal register operand allocation are required; unsupported forms cannot
  execute. No additional shared-memory capacity is needed.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Forward each consumer warpgroup's probability fragment directly from registers
into its local PV WGMMA. Remove the local shared-memory round trip and its
publication barrier. Keep shared probability exchange for the other warpgroup.

The edits are in `solution/attention.cuh`, `solution/hopper.cuh`, and
`solution/mma.cuh`. Keep `config.toml`, `solution/kernel.cu`, the tensor ABI,
caller device/stream, and launch unchanged.

## Replay

1. In `solution/mma.cuh`, copy the existing `pv_shared` helper into a new
   `pv_local` helper immediately after it. Retain the complete accumulator
   operand list `%0` through `%127` and every `"+f"(d[i])` output constraint.
   Change only its signature, A operand, instruction tail, and input constraints:

   ```cuda
   __device__ __forceinline__ void pv_local(
       const uint32_t* a, uint64_t b, float (&d)[128]);
   ```

   The final asm string in `pv_shared` is:

   ```cuda
   "%128, %129, 1, 1, 1, 0, 1;"
   ```

   Replace that string in the copy with:

   ```cuda
   "{%128, %129, %130, %131}, %132, 1, 1, 1, 1;"
   ```

   Replace the copy's input constraints with:

   ```cuda
   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(b));
   ```

   This retains `wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16`,
   accumulation into D, positive operand scales, and transposed B. Register A
   has no transpose immediate. Keep the original `pv_shared` for peer PV.

2. Add the BF16 `reg_fence` overload below the FP32 overload in
   `solution/hopper.cuh`. Add `pv_regs` before `pv_smem` in
   `solution/attention.cuh`. Both helpers appear in After below. The compiler
   fences expose packed A and accumulator dependencies; `mma_fence()` orders
   their ordinary register accesses before asynchronous WGMMA.

3. Replace the two local staging sequences in `consume<Group>` as shown below.
   Group 0 no longer writes `local_prob` before local PV. Keep its later
   rescaled `save_prob` after `sync<Barrier::Max1>()`, the proxy fence, and
   `Prob0` arrival: group 1 still needs that shared copy. For group 1, move
   `save_prob(s, sm.prob, lane)` after the local PV commit. Keep its existing
   `Prob0` wait, peer PV, proxy fence, and `Prob1` arrival.

4. Remove only `Local0` and `Local1` from `Barrier`. Keep all other barriers,
   shared arrays, descriptors, producer transfers, phase changes, WGMMA
   commits/waits, and QK/PV scheduling. The BF16 fence and register-A helper
   are the adapters for this storage bypass, not separate optimizations.

## Ownership and ordering

For `lane = threadIdx.x % kWarpgroup`, `s[8*step+j]` owns probability row
`row_index((j % 4) / 2, lane)` and column
`16*step + 8*(j / 4) + 2*(lane % 4) + (j % 2)`, for `j` in `[0,8)`.
Each step passes four packed words; each word keeps its two BF16 values in
low-to-high order. This is the register-A fragment mapping. No shuffle or
shared-layout change is needed.

The preceding shared stores use `save_prob`: logical `(row,col)` maps through
`swizzle(row,col)`, and `pv_smem` reads it through `desc_k`. Forwarding replaces
that writer/reader pair only for local PV. The retained peer stores and their
matching descriptors keep exactly the same swizzle. V remains in shared memory
and keeps `desc_mn` and its descriptor increments.

Group 0 waits for its current QK before reusing KV0's RoPE tile for probabilities.
Its local PV finishes at the retained `wait<0>()` before `s` is rescaled. Group 1
keeps `s` unchanged through its local PV's completion at the existing
`wait<1>()`. Its shared probability tile remains published to group 0 through
the existing proxy fence and `Prob1` barrier. The next `qk_peer` waits for the
next KV1-left production, which follows group 0's `free1[0]` arrival after its
previous peer PV completes; this protects `sm.prob` from premature reuse.
Keep `free0[1]` after group 1's completed peer PV to protect KV0's probability
storage. Query/KV readiness waits still precede their consumers.

The deoptimized local barriers publish all participating lanes' shared stores.
They may be removed only with those local shared reads. Do not remove fences
or barriers that publish probabilities to the other warpgroup.

Keep invalid-index zero-fill and score masking, traversal and accumulation
order, softmax scaling, all BF16 round-to-nearest conversions, and FP32
max/log-sum-exp calculations. Forward existing `s` bits; do not recompute or
recast probabilities. Bounds handling and output ownership stay unchanged.

## Example configuration

The supplied workload has 8192 tokens, 128 heads, QK width 576, value width 512,
and TOPK 2048. Inputs are contiguous BF16 q/kv and int32 sparse indices;
output is BF16, with FP32 maximum and LSE. The YaRN scale is
`0.1352337788608801f`. Preserve the complete supplied numerical contract.

Keep 64-by-64 score tiles, 32 BF16 probability elements and 128 FP32 output
accumulators per consumer thread. Local PV traverses four K16 instructions
with `step * 4` packed-word offsets and `step * 128` B-descriptor offsets.
The latter encodes a 2048-byte advance in 16-byte descriptor units.

Each CTA handles one token and 64 heads; its two consumer warpgroups own the
first and second 256 output columns. Launch 16384 CTAs of 384 threads with a
one-CTA cluster. Keep 231376 shared bytes, resident Q, two KV buffers, the
shared peer probability tile, and the 256-float reduction scratch. KV0's last
8192-byte tile supplies group 0's probability storage; `sm.prob` supplies group
1's. No allocation shrinks when local staging disappears.

Retain the two-block traversal, four producer transfer groups, TMA Q loading,
`cp.async` KV loading, mbarriers, 128-byte shared swizzles, cache policy, direct
output stores, shared softmax reductions, and QK/peer-PV overlap. Preserve all
compile flags, including C++20, `-O3`, `--use_fast_math`, and register-usage level
10; build for `sm_90a`. These are replay settings, not general bypass prerequisites.

Register pressure and compiler spills can reduce the savings; inspect operand
loads and asynchronous waits in generated code. A spill-free kernel is not
required: register-A WGMMA can coexist with spills, including reloads before
issue. Preserve each issued fragment until its asynchronous reads complete.

Use the existing Kernel evaluator on the unchanged problem for correctness and
latency. Inspect generated HGMMA operands: local PV must use register A, while
peer PV remains descriptor sourced. Keep validation and measurements outside
this card. No compiler-control flag change is required.

# Precondition

- Data types: the existing MMA types need equivalent shared-A and register-A
  forms. Preserve operand bits, accumulator type, and reduction order; otherwise
  changing the operand path could change numerical results.
- Layout: each consuming lane already owns the elements expected by the
  register-A fragment and can pack them without changing their correspondence.
  A different lane mapping feeds incorrect matrix elements. Global addressing
  is untouched, so the bypass adds no global contiguity or alignment condition.
- Storage: A is available in the consuming lanes' registers before staging to
  shared memory. Without that availability, forwarding requires a new transfer.
  Any other consumers of shared A must retain their shared copies; removing
  those copies would leave their reads unsatisfied.
- Pipeline: all 128 warpgroup threads participate uniformly in WGMMA. Finish
  producing A before consumption and keep its registers live and unchanged
  until asynchronous completion; otherwise the reader can see incorrect data.
  Retained shared consumers still need producer publication, reader visibility,
  and completion before storage reuse. No particular stage count is required.
- Hardware: the target and compiler support the matching register-A WGMMA form
  and its legal register operand allocation. Unsupported forms cannot execute.
  No additional shared capacity is needed because forwarding removes local
  shared accesses and retains existing allocations.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

These are the two local PV sites in `consume<Group>`; intervening code stays.

```cuda
// Group 0, immediately after arrive<Barrier::Max0>().
Bf16* local_prob = sm.kv[0] + (kKeyTiles - 1) * kTileElems;
save_prob(s, local_prob, lane);
shared_fence();
sync<Barrier::Local0, kWarpgroup>();
pv_smem(local_prob, sm.kv[0], o);
commit();
wait<0>();
sm.free0[0].arrive();

// Group 1, immediately after arrive<Barrier::Max1>().
save_prob(s, sm.prob, lane);
shared_fence();
sync<Barrier::Local1, kWarpgroup>();
pv_smem(sm.prob, sm.kv[1] + kHalfTiles * kTileElems, o);
commit();
sync<Barrier::Prob0>();
```

## After

Add these helpers, using `pv_local` constructed from `pv_shared` as specified
in Replay. Keep the existing FP32 `reg_fence` overload.

```cuda
// solution/hopper.cuh, namespace hopper.
template<int N>
__device__ __forceinline__ void reg_fence(Bf16 (&r)[N]) {
#pragma unroll
    for (int i = 0; i < N / 2; ++i)
        asm volatile("" : "+r"(reinterpret_cast<uint32_t*>(r)[i]) :: "memory");
}

// solution/attention.cuh, namespace spa.
__device__ __forceinline__ void pv_regs(
    Bf16 (&s)[kScoreRegs], const Bf16* v, float (&o)[kOutputRegs]) {
    const uint64_t b = desc_mn(v);
    reg_fence(s);
    reg_fence(o);
    mma_fence();
#pragma unroll
    for (int step = 0; step < kTile / 16; ++step)
        pv_local(reinterpret_cast<const uint32_t*>(s) + step * 4,
                 desc_step(b, step * 128), o);
    reg_fence(o);
    reg_fence(s);
}
```

Replace the local sites with:

```cuda
// Group 0: retain its later rescaled save_prob for group 1.
pv_regs(s, sm.kv[0], o);
commit();
wait<0>();
sm.free0[0].arrive();

// Group 1: retain the shared copy for group 0 after the local commit.
pv_regs(s, sm.kv[1] + kHalfTiles * kTileElems, o);
commit();
save_prob(s, sm.prob, lane);
sync<Barrier::Prob0>();
```

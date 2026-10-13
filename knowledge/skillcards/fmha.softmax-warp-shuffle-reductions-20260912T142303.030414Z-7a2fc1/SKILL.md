---
skill_id: fmha.softmax-warp-shuffle-reductions
intent: Exchange softmax reduction values through warp shuffles instead of shared
  memory.
preconditions:
- 'Data types: exchanged values must have a lossless shuffle representation; preserve
  combining arithmetic and peer order because conversion or reassociation can change
  reduction results.'
- 'Layout: each XOR-selected source must belong to the same CUDA warp and participate
  in the exchange; shuffles cannot read another warp or an inactive source. No additional
  global layout or alignment requirement applies.'
- 'Storage: lane-produced values pass through shared scratch used only for these exchanges
  and remain available in their producing lanes; removing scratch must not discard
  another consumer''s data.'
- 'Pipeline: all lanes named by the mask must execute the same exchange with completed
  input values. Existing barriers must make scratch writes visible before reads and
  complete reads before reuse; remove them only when they order this eliminated scratch
  traffic, since shuffles do not fence other memory.'
- 'Hardware: the compiler and target must support synchronized XOR warp shuffles.
  No additional shared-memory capacity is required because the transformation eliminates
  exchange scratch.'
scope:
  cases:
  - fmha
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Replace softmax's shared-memory peer exchanges with register-to-register XOR
warp shuffles. Preserve the same maximum and denominator reduction trees.

Apply these edits to the supplied bundle:

1. In `solution/attention.cuh`, replace both `exchange` calls in `softmax`
   and both in `consumer`'s final denominator reduction with the After excerpts.
   Keep XOR distance 2 followed by 1, preserving each lane's operand order.
2. Remove the final `Shared& s` parameter from `softmax` and the final `s`
   argument from its `Step::First` and `Step::Next` calls in `consumer`.
3. In `solution/ops.cuh`, delete `exchange` and the final
   `Shared::reduction` member. Retain `Peer` and `kFullWarpMask`. Delete only the two
   warp barriers inside the removed helper.
4. Keep `solution/kernel.cu` unchanged: its attribute and launch already use
   `sizeof(Shared)`, so they automatically request the smaller allocation.

Each math lane supplies its current maximum or partial denominator. XOR peers
remain within its aligned group of four lanes; each lane receives the same peer
value as before. Distinct math warps need no mutual synchronization for this
exchange. Global tensor layouts, fragment ownership, CTA mapping, and launch
dimensions stay unchanged.

Retain score-producer WGMMA waits and operand fences before maximum reduction,
all TMA transaction barriers, named producer/consumer barriers, and K/V stage
release waits. The removed warp barriers solely publish and protect exchange
scratch; a shuffle supplies no ordering for other shared-memory accesses.
Removing this scratch also removes its reuse hazard.

Keep the last-K-tile invalid-score mask (`-INFINITY`) before softmax and the
epilogue's row bounds. Padded query lanes still participate in reductions;
invalid CTAs skip the entire consumer body uniformly. Do not predicate
individual shuffle calls on output validity.

Preserve scalar local reduction order, `fmaxf`, addition, online rescaling,
`exp2f`, reciprocal handling, and FP16 round-to-nearest packing. Keep the
complete problem and compilation flags. Validate through
`klineage.harness.evaluate` with the supplied oracle and policy; keep evidence
outside this card.

## Example configuration

This replay uses packed contiguous FP16 Q/K/V/output `[16384,64,128]`, int32
offsets `[9]`, eight sequences, FP32 accumulators and softmax values, noncausal
attention, no dropout, and scale `1/sqrt(128)`. The source's `kScale =
0.0883883461f` and `kLog2Scale = 0.127517432f` remain unchanged. Packed row
stride is `kHeads*kDim` elements; each head has contiguous feature elements.

Keep tiles `kM=128`, `kN=176`, `kDim=128`, two K/V stages, 384 CTA threads,
128-thread warpgroups, and 256 math threads. The first producer warp loads
tiles; math threads are `threadIdx.x=128..383`. Their local index is
`tid=threadIdx.x-kGroupSize`, warp is `tid/32`, and lane is `tid%32`.
Each lane owns two softmax rows; four adjacent lanes share each row's columns.
The shared implementation reserves `kMathThreads*sizeof(float)=1024` bytes
for exchanges. Removing the final member changes `sizeof(Shared)` from
214272 to 213248 bytes without moving earlier members.

For local row `r`, query ownership is
`work.y*kM + (tid/32)*16 + (tid%32)/4 + r*8`.
Score slot `i` owns key column `(tid%4)*2 + (i/4)*8 + i%2`.
`Peer::Alternate` is XOR 2 and `Peer::Adjacent` is XOR 1;
`kFullWarpMask=0xffffffffu` includes every CUDA lane. Both reductions use
Alternate then Adjacent; retaining this order preserves denominator rounding.

Retain WGMMA QK `m64n176k16` and PV `m64n128k16`, 88 score registers,
64 output registers, 44 packed probability registers, FP16 probability
conversion, online softmax, TMA SW128 operand layouts, cache hints, two-stage
K/V rotation, producer/math overlap, scheduler shuffles, and scalar output
stores. Keep direct CTA dispatch with `grid=kMaxQTiles*kHeads`,
`block=kThreads`, caller stream, host ABI, LSE workspace, and metadata setup.
Keep `-O3 -std=c++17 --use_fast_math --resource-usage -lineinfo -DNDEBUG`
and the SM90a build target. These retained mechanisms and dimensions are
replay settings, not prerequisites for exchanging lane values.

# Precondition

- Data types: every exchanged value needs a lossless shuffle representation.
  Preserve the combining operations and ordered peer sequence; changing
  representation or reassociating the reduction can alter its numerical result.
- Layout: the XOR-selected source must be a participating lane in the same
  CUDA warp. A shuffle cannot fetch across warps or from an inactive source.
  There is no additional global contiguity or alignment requirement because
  this transformation only replaces communication between lanes.
- Storage: producing lanes must retain the values currently communicated
  through shared scratch. That scratch must serve only these exchanges;
  deleting storage needed by another reader would discard live data.
- Pipeline: every lane named by the mask must execute the same exchange
  after producing its input. In the preceding implementation, barriers make
  scratch writes visible before peer reads and finish reads before slot reuse.
  Remove those barriers only if they exclusively protect eliminated scratch
  traffic. Preserve ordering for all other memory, since shuffles are not
  memory fences. No particular stage count is required.
- Hardware: the compiler and target must implement synchronized XOR warp
  shuffles. No additional shared-memory capacity is needed: the optimization
  removes the exchange allocation.

# Scope

- Cases: fmha
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

These excerpts occupy separate existing scopes. The signature and call-site
cleanup described above is also required. Keep `Peer` and `kFullWarpMask` in `ops.cuh`.

## Before

```cuda
// ops.cuh: final member of Shared.
float reduction[kMathThreads];

// ops.cuh: shared exchange helper; Peer selects XOR 2 or XOR 1.
__device__ __forceinline__ float exchange(float value, Peer peer, Shared& s) {
    const int tid = threadIdx.x - kGroupSize;
    volatile float* slots = s.reduction;

    // Publish values before peer reads.
    slots[tid] = value;
    __syncwarp(kFullWarpMask);
    const float other = slots[tid ^ int(peer)];

    // Finish reads before slot reuse.
    __syncwarp(kFullWarpMask);
    return other;
}

// attention.cuh: softmax, after each lane's local maximum.
m = fmaxf(m, exchange(m, Peer::Alternate, s));
m = fmaxf(m, exchange(m, Peer::Adjacent, s));

// attention.cuh: consumer, before reciprocal and LSE computation.
sum[r] += exchange(sum[r], Peer::Alternate, s);
sum[r] += exchange(sum[r], Peer::Adjacent, s);
```

## After

```cuda
// ops.cuh: delete Shared::reduction and exchange.
// Retain Peer and kFullWarpMask; no scratch or exchange barriers remain.

// attention.cuh: preserve each lane's maximum reduction tree.
m = fmaxf(m, __shfl_xor_sync(kFullWarpMask, m, int(Peer::Alternate)));
m = fmaxf(m, __shfl_xor_sync(kFullWarpMask, m, int(Peer::Adjacent)));

// attention.cuh: preserve denominator addition order.
sum[r] += __shfl_xor_sync(kFullWarpMask, sum[r], int(Peer::Alternate));
sum[r] += __shfl_xor_sync(kFullWarpMask, sum[r], int(Peer::Adjacent));
```

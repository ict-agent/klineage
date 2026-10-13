---
skill_id: kda.recurrence-value-register-tiling
intent: Reuse recurrence operands across multiple value tiles per warp through register
  tiling.
preconditions:
- 'Data types: no additional dtype requirement; register tiling reuses existing operand
  representations. Each output must retain its accumulator type, reduction order,
  and conversion points to preserve numerical behavior.'
- 'Layout: merged value tiles must use common left operands and have known, disjoint
  output/state ownership. Preserve the existing per-lane fragment mapping and valid
  tile addresses; otherwise reuse supplies the wrong values or stores overlap. No
  extra contiguity or alignment is introduced.'
- 'Storage: the reusable operands and each tile''s state must be accessible in CTA-visible
  shared storage before computation; the existing loads cannot retrieve another warp''s
  private registers.'
- 'Pipeline: complete warps must participate in fragment operations; input producers
  must finish before reads. Publish correction stores before their consumers, finish
  transpose-scratch reads before reuse, and retain CTA participation at chunk barriers
  before input, output, or state storage is reused. These dependencies prevent stale
  reads and overwrite races.'
- 'Hardware: no new instruction feature or shared-memory allocation is needed. The
  larger register live set must fit the allocation budget to remain register-resident:
  R_common + tiles_per_warp * R_tile; spilling can erase the reuse benefit. R_common
  and R_tile denote common and per-tile live register demand. The chosen block must
  fit the device thread limit.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Tile recurrence products across value columns in registers. One warp owns
multiple independent output fragments and reuses query, key, inverse, readout,
and decay operands across them. This removes repeated shared loads by neighboring
warps while keeping each output's arithmetic sequence.

In `solution/native.cuh`, replace the two thread constants shown below. In
`solution/recurrence.cuh`, replace only `recur_tile` with the After function.
The snippets combine these two locations; retain their existing file placement.
All other functions, shared structures, compiler flags, and the host ABI remain.

The enlarged warp owns two adjacent value tiles. Keep separate `u`, `o`,
`out_bf`, and `val` fragments for each. For every key step, load `ka` and `qa`
once, then accumulate each tile using its own state fragment. Reuse `inv` and
beta across residual solves, `mqk` across readouts, and `kr` and decay across
state updates. Offset tile-specific value addresses by `i*kChunk` or
`bi*kChunk`. Keep increasing key-step order, BF16 rounding points, scalar
BF16 addition, and the existing FP32 decay-update expression.

Ownership stays within the same head CTA. Correction data and value-key state
rows remain exclusive to their producing warp. Existing warp barriers publish
corrections and protect transpose scratch; CTA barriers publish cooperative
input copies, finish compute before output copies, and finish all readers before
chunk storage reuse. Every launched thread still participates in those CTA
barriers, including threads outside `kComputeWarps`.

## Example configuration

Replay uses batch 1, 4096 tokens, 96 heads, and 128 key/value dimensions with
16-token chunks. The retained fragment ABI is `Reg { uint32_t x[4]; }` and
`Acc { float x[8]; }`: one logical 16-by-16 product per warp, implemented by
two `mma.sync` m16n8k16 BF16 products with FP32 accumulation. The removed
software tiling is the array of these fragments; keep their internal MMA and
matrix-load/store adapters intact.

Before: eight compute warps, 256 block threads, value origin `warp*kChunk`.
After: four compute warps, 192 block threads, value origin `warp*32`, with
128 compute threads and 64 additional cooperative-copy threads. The launch in
`solution/native.cu` already uses `kRecurThreads` and `sizeof(RecurShared)`;
changing the constants updates it and the cooperative-copy strides. Keep
`kComputeWarps = kComputeThreads/kWarp` and the existing compute predicate.
The recurrence grid stays `(1,kHeads)`. Preserve the prepare grid
`(kTiles,kHeads)` and its 256 threads.

BF16 tiles use eight-column shared slabs:
`offset<Rows>(r,c) = r*8 + (c&7) + (c/8)*Rows*8`.
State is logically `[value,key]`; corrections and outputs are `[token,value]`.
Global activation/output arrays remain contiguous BTHD. Preserve the FP32
state transfer/conversion path and the beta allocation's existing zero fill.
All value tiles are complete in this workload, so the shown tile addresses
need no new bounds branch. Other extents need valid padding or whole-fragment
bounds handling compatible with the retained warp collectives.

Keep one serial input buffer, 256 ordered chunks per head, shared correction
materialization, q/k product interleaving, and current-step operand loading.
Dynamic shared sizes stay 42368 bytes for preparation and 124672 bytes for
recurrence; `InputShared` stays 18048 bytes. The existing transpose scratch
supports eight warps and remains unchanged. Preserve `__launch_bounds__`, all
barriers, fast-math flags, normalization grouping, and every conversion point.
Use the supplied problem's evaluator for correctness and latency.

# Precondition

- Data types: no additional dtype restriction follows from assigning more tiles
  to one warp. Reuse the existing representations and preserve each output's
  accumulator type, reduction order, and conversion points; reassociation or
  delayed rounding would change the numerical behavior.
- Layout: the tiles being merged share left operands and have known, disjoint
  output/state regions. The existing per-lane fragment mapping and valid tile
  addresses must survive reassignment, or operand reuse and stores become
  incorrect. No new contiguity or alignment constraint is imposed by tiling.
- Storage: reusable operands and tile state are available in CTA-visible shared
  storage before consumption. This permits the new owner to issue the existing
  loads; another warp's private registers would require a different transfer path.
- Pipeline: full warps participate in fragment operations, and producers finish
  before input reads. Correction stores become visible before consumers;
  transpose-scratch readers finish before scratch reuse. All CTA threads reach
  chunk barriers so computation, output copies, and subsequent input/state
  reuse cannot race. No particular number of buffers is required by tiling.
- Hardware: tiling adds no instruction feature or shared allocation. Keeping its
  live fragments in registers requires an allocation budget covering
  `R_common + tiles_per_warp * R_tile`, subject to per-thread and block/SM
  allocation limits. Here `R_common` and `R_tile` are common and per-tile live
  register demand; spilling can remove the benefit. The selected block's thread
  count must remain legal.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

```cuda
constexpr int kComputeThreads = kDim / kChunk * kWarp;
constexpr int kRecurThreads = kComputeThreads;

__device__ __forceinline__ void recur_tile(RecurShared& s, InputShared& in, BF16* out) {
    int warp = threadIdx.x/kWarp, lane = threadIdx.x%kWarp, group = lane/4;
    // Each warp owns one value tile and one accumulator per product.
    int value = warp*kChunk;
    Acc u, o;

    // Interleave k@state and q@state using only current-step operands.
    #pragma unroll
    for (int k = 0; k < kDim/kChunk; ++k) {
        Reg ka = load_a<kChunk>(in.kd,0,k*kChunk,lane);
        Reg qa = load_a<kChunk>(in.qd,0,k*kChunk,lane);
        Reg sb = load_b<kDim>(s.state,value,k*kChunk,lane);
        mma(u,ka,sb);
        mma(o,qa,sb);
    }

    Reg out_bf = quantize(o);
    Reg val = load_a<kChunk>(in.v,0,value,lane);
    Reg inv = load_a<kChunk>(in.inv,0,0,lane);
    BF16 beta0 = BF16(sigmoid(bf_float(in.beta[group])));
    BF16 beta1 = BF16(sigmoid(bf_float(in.beta[group+8])));

    // Publish this warp's corrected values for subsequent products.
    Reg residual = quantize(u);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        Pair v{val.x[j]}, r{residual.x[j]}, result;
        BF16 beta = j%2 == 0 ? beta0 : beta1;
        result.h[0] = (v.h[0]-r.h[0])*beta;
        result.h[1] = (v.h[1]-r.h[1])*beta;
        residual.x[j] = result.u;
    }
    u = Acc{};
    mma(u,inv,transpose(residual));
    store_c<kChunk>(s.correction,quantize(u),0,value,lane);
    __syncwarp(kAllLanes);
    Reg mqk = load_a<kChunk>(in.mqk,0,0,lane);
    Reg correction = load_correction(s,value,lane);
    o = Acc{};
    mma(o,mqk,correction);
    Reg prod = quantize(o);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        Pair a{out_bf.x[j]}, b{prod.x[j]}, c;
        c.u = add_pair(a.u,b.u);
        out_bf.x[j] = c.u;
    }
    store_c<kChunk>(out,out_bf,0,value,lane);

    // Load each key block and its state immediately before their update.
    #pragma unroll
    for (int m = 0; m < kDim/kChunk; ++m) {
        Reg kr = load_t<kChunk>(in.kr,m*kChunk,0,lane);
        float decay0 = in.gt[m*kChunk+group];
        float decay1 = in.gt[m*kChunk+group+8];
        u = Acc{};
        Reg correction = load_correction(s,value,lane);
        mma(u,kr,correction);
        Reg state = load_t<kDim>(s.state,m*kChunk,value,lane);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            Pair old{state.x[j]}, next;
            float decay = j%2 == 0 ? decay0 : decay1;
            next.h[0] = BF16(bf_float(old.h[0])*decay+u.x[2*j]);
            next.h[1] = BF16(bf_float(old.h[1])*decay+u.x[2*j+1]);
            state.x[j] = next.u;
        }
        store_t<kDim>(s.state,state,m*kChunk,value,lane);
    }
}
```

## After

```cuda
constexpr int kComputeThreads = 128;
constexpr int kRecurThreads = 192;

__device__ __forceinline__ void recur_tile(RecurShared& s, InputShared& in, BF16* out) {
    int warp = threadIdx.x/kWarp, lane = threadIdx.x%kWarp, group = lane/4;
    int value = warp*32;
    Acc u[2], o[2];

    // Interleave k@state and q@state using only current-step operands.
    #pragma unroll
    for (int k = 0; k < kDim/kChunk; ++k) {
        Reg ka = load_a<kChunk>(in.kd,0,k*kChunk,lane);
        Reg qa = load_a<kChunk>(in.qd,0,k*kChunk,lane);
        Reg sb = load_b<kDim>(s.state,value,k*kChunk,lane);
        mma(u[0],ka,sb);
        mma(o[0],qa,sb);
        sb = load_b<kDim>(s.state,value+kChunk,k*kChunk,lane);
        mma(u[1],ka,sb);
        mma(o[1],qa,sb);
    }

    Reg out_bf[2] = {quantize(o[0]),quantize(o[1])};
    Reg val[2] = {load_a<kChunk>(in.v,0,value,lane),load_a<kChunk>(in.v,0,value+kChunk,lane)};
    Reg inv = load_a<kChunk>(in.inv,0,0,lane);
    BF16 beta0 = BF16(sigmoid(bf_float(in.beta[group])));
    BF16 beta1 = BF16(sigmoid(bf_float(in.beta[group+8])));

    // Publish each warp's corrected values for subsequent products.
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        Reg residual = quantize(u[i]);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            Pair v{val[i].x[j]}, r{residual.x[j]}, result;
            BF16 beta = j%2 == 0 ? beta0 : beta1;
            result.h[0] = (v.h[0]-r.h[0])*beta;
            result.h[1] = (v.h[1]-r.h[1])*beta;
            residual.x[j] = result.u;
        }
        u[i] = Acc{};
        mma(u[i],inv,transpose(residual));
        store_c<kChunk>(s.correction,quantize(u[i]),0,value+i*kChunk,lane);
    }
    __syncwarp(kAllLanes);
    Reg mqk = load_a<kChunk>(in.mqk,0,0,lane);
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        Reg correction = load_correction(s,value+i*kChunk,lane);
        o[i] = Acc{};
        mma(o[i],mqk,correction);
        Reg prod = quantize(o[i]);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            Pair a{out_bf[i].x[j]}, b{prod.x[j]}, c;
            c.u = add_pair(a.u,b.u);
            out_bf[i].x[j] = c.u;
        }
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i) store_c<kChunk>(out,out_bf[i],0,value+i*kChunk,lane);

    // Load each key block and its state immediately before their update.
    #pragma unroll
    for (int m = 0; m < kDim/kChunk; ++m) {
        Reg kr = load_t<kChunk>(in.kr,m*kChunk,0,lane);
        float decay0 = in.gt[m*kChunk+group];
        float decay1 = in.gt[m*kChunk+group+8];
        #pragma unroll
        for (int bi = 0; bi < 2; ++bi) {
            u[bi] = Acc{};
            Reg correction = load_correction(s,value+bi*kChunk,lane);
            mma(u[bi],kr,correction);
        }
        #pragma unroll
        for (int bi = 0; bi < 2; ++bi) {
            Reg state = load_t<kDim>(s.state,m*kChunk,value+bi*kChunk,lane);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                Pair old{state.x[j]}, next;
                float decay = j%2 == 0 ? decay0 : decay1;
                next.h[0] = BF16(bf_float(old.h[0])*decay+u[bi].x[2*j]);
                next.h[1] = BF16(bf_float(old.h[1])*decay+u[bi].x[2*j+1]);
                state.x[j] = next.u;
            }
            store_t<kDim>(s.state,state,m*kChunk,value+bi*kChunk,lane);
        }
    }
}
```

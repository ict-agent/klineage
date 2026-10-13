---
skill_id: kda.shared-operand-register-lookahead
intent: Hide shared-memory load latency through register lookahead.
preconditions:
- 'Data types: no additional dtype requirement; moving loads earlier must preserve
  their bits and leave arithmetic, accumulation order, and conversions unchanged.'
- 'Layout: future operand addresses and consumer ownership must be known before use;
  the existing loads must remain valid for the same lanes. No additional contiguity
  or alignment is needed because their addresses and instructions do not change.'
- 'Storage: future operands must already reside in CTA-visible shared memory at the
  earlier load point; prefetching cannot supply values that have not been produced.'
- 'Pipeline: shared producers must finish and publish before hoisted loads; intervening
  writes must not change future operands. All lanes required by each collective load
  must participate, register operands must be consumed before replacement, and shared
  readers must finish before buffer reuse.'
- 'Hardware: CUDA shared loads and registers with capacity for current and lookahead
  operands alongside other live values within thread and CTA register limits; spills
  can defeat latency hiding. No asynchronous-copy hardware is required.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Pipeline shared-memory operand loads into registers ahead of their use to overlap
load latency with independent matrix products and state arithmetic. Restore this
one mechanism at all three sites below; replace each Before region with its After
region. Paths are relative to `submission/`.

- A: `solution/prepare.cuh`, `prepare()`, immediately after `Acc result` in
  `if (warp < 2)`: prime the first operands, load the next K operands before the
  current MMA, then rotate registers. Guard both next loads and register rotation
  at the final step.
- B: `solution/recurrence.cuh`, `recur_tile()`, between `Acc u[2], o[2]` and
  `Reg out_bf[2]`: prime key, query, and first state operands. Load the second
  value block before the first pair of products, and the next K operands before
  the second pair. Retain the four MMA calls in their existing order.
- C: the same function, after the `store_c` output loop through the final state
  update loop: prime restored keys, decay values, and both state fragments.
  Snapshot current decays before prefetching the next key/decay block; refill each
  state register only after storing its current block. Keep correction loads and
  their MMA calls in place.

Addresses, shared layouts, ownership, launch dimensions, and synchronization stay
unchanged. Every lookahead is bounded by the existing iteration count; the primed
first iteration exists in this workload. Empty or partial tiles would require
guarding the prime and retaining the original padding rules. Preserve all BF16
rounding points, FP32 accumulation order, state-update expressions, and compiler
flags. Early loads move bits without changing arithmetic.

Preparation's preceding CTA barrier publishes its decayed operands. Recurrence's
barrier after `load_input` publishes chunk inputs; the previous chunk's barriers
publish state. During state updates, each warp owns disjoint value rows and visits
distinct key blocks, so its current stores cannot change a future prefetched
block. Preserve the correction `__syncwarp(kAllLanes)` and all CTA barriers before
chunk buffers are overwritten. Register refill follows the last use of the
corresponding old operand. No new shared storage, barriers, or async operations
are introduced.

Check the rebuilt bundle with `klineage.harness.evaluate` on the preserved problem;
retain its numerical and timing policy and keep evaluation evidence outside this card.

## Example configuration

Retain batch 1, 4096 tokens, 96 heads, head dimension 128, chunk size 16, and
256 chunks per head. A traverses eight K steps with two preparation warps;
warp 0 builds the key system and warp 1 the query system. B and C use four compute
warps, each owning 32 value rows as two 16-row fragments; `value = warp*32`.
The replay uses one-step register lookahead and keeps loop unrolling.

Keep preparation's `(kTiles,kHeads)` grid and 256 threads, recurrence's
`(1,kHeads)` grid and 192 threads, and their existing launch bounds. Shared struct
sizes stay 42368 bytes (`PrepareShared`), 18048 bytes (`InputShared`), and 124672
bytes (`RecurShared`), plus the existing static transpose scratch. Keep the
single recurrence input buffer, scratch allocation/cache, and caller CUDA stream.

Global inputs/output remain contiguous BTHD; state remains contiguous BVK.
Prepared global matrices remain row-major. Shared matrix elements retain
`offset<Rows>(row,col) = row*8 + (col&7) + (col/8)*Rows*8`.
Keep `load_a`, `load_b`, `load_t`, `store_t`, `Reg`, and `Acc` unchanged, including
their lane-to-fragment mapping. `Reg` carries four packed words and `Acc` eight
FP32 accumulators. This instance retains BF16 `ldmatrix`/`stmatrix` adapters,
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, FP32 accumulation,
BF16 intermediate rounding, normalization grouping, gate approximations, triangular
inversion, and chunked recurrence. The target remains NVIDIA SM90 with CUDA 13.
These retained computation choices do not make their dimensions or dtypes
prerequisites for register lookahead.

# Precondition

- Data types: no additional dtype requirement. Prefetch the same representation
  with the existing loads; changing rounding or reduction order would be an
  arithmetic transformation, not this scheduling change.
- Layout: future addresses and their owning consumers must be known early, or the
  preload could fetch the wrong fragment. Keep the original lane participation
  and valid load mapping. No additional contiguity or alignment is imposed:
  the same instructions access the same addresses.
- Storage: future operands must already be available in CTA-visible shared
  memory at the proposed preload point. A load before production would capture
  stale values; this transformation does not move their producers.
- Pipeline: producer completion and visibility must precede hoisted loads, and
  intervening stores must not alter their values. Every lane required by a
  collective load must execute it. Finish consuming each register operand before
  refilling that register, and finish shared reads before shared-buffer reuse;
  otherwise operands can be stale, overwritten, or consumed by an invalid
  collective. Preserve the barriers establishing these dependencies.
- Hardware: CUDA shared loads and registers must accommodate current operands,
  prefetched operands, and other live state within the thread and CTA register
  limits. Spilling the lookahead values can erase the intended latency overlap.
  The technique needs no asynchronous-copy hardware.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

The three labeled regions are separate replacements inside existing functions.
All surrounding code and helpers remain unchanged.

## Before

```cuda
// A: prepare.cuh, prepare(), after Acc result;
        #pragma unroll
        for (int step = 0; step < kDim/kChunk; ++step) {
            // Load only this step's operands before consuming them.
            Reg ar = load_a<kChunk>(a,0,step*kChunk,lane);
            Reg br = load_b<kChunk>(s.ki,0,step*kChunk,lane);
            mma(result,ar,br);
        }

// B: recurrence.cuh, recur_tile(), after Acc u[2], o[2];
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

// C: recurrence.cuh, recur_tile(), after the output store loop
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
```

## After

```cuda
// A: prepare.cuh, prepare(), after Acc result;
        Reg ar = load_a<kChunk>(a,0,0,lane);
        Reg br = load_b<kChunk>(s.ki,0,0,lane);
        #pragma unroll
        for (int step = 0; step < kDim/kChunk; ++step) {
            Reg next_a, next_b;
            if (step+1 < kDim/kChunk) {
                next_a = load_a<kChunk>(a,0,(step+1)*kChunk,lane);
                next_b = load_b<kChunk>(s.ki,0,(step+1)*kChunk,lane);
            }
            mma(result,ar,br);
            if (step+1 < kDim/kChunk) { ar = next_a; br = next_b; }
        }

// B: recurrence.cuh, recur_tile(), after Acc u[2], o[2];
    Reg ak = load_a<kChunk>(in.kd,0,0,lane);
    Reg aq = load_a<kChunk>(in.qd,0,0,lane);
    Reg b = load_b<kDim>(s.state,value,0,lane);

    // Interleave k@state and q@state, prefetching one K step.
    #pragma unroll
    for (int k = 0; k < kDim/kChunk; ++k) {
        Reg ka = ak, qa = aq, sb = b;
        b = load_b<kDim>(s.state,value+kChunk,k*kChunk,lane);
        mma(u[0],ka,sb);
        mma(o[0],qa,sb);
        sb = b;
        if (k+1 < kDim/kChunk) {
            ak = load_a<kChunk>(in.kd,0,(k+1)*kChunk,lane);
            aq = load_a<kChunk>(in.qd,0,(k+1)*kChunk,lane);
            b = load_b<kDim>(s.state,value,(k+1)*kChunk,lane);
        }
        mma(u[1],ka,sb);
        mma(o[1],qa,sb);
    }

// C: recurrence.cuh, recur_tile(), after the output store loop
    // Update each warp's two value blocks; one restored-key block is prefetched.
    Reg kr = load_t<kChunk>(in.kr,0,0,lane);
    Reg state[2] = {load_t<kDim>(s.state,0,value,lane),load_t<kDim>(s.state,0,value+kChunk,lane)};
    float g0 = in.gt[group], g1 = in.gt[group+8];
    #pragma unroll
    for (int m = 0; m < kDim/kChunk; ++m) {
        float decay0 = g0, decay1 = g1;
        #pragma unroll
        for (int bi = 0; bi < 2; ++bi) {
            u[bi] = Acc{};
            Reg correction = load_correction(s,value+bi*kChunk,lane);
            mma(u[bi],kr,correction);
        }
        if (m+1 < kDim/kChunk) {
            kr = load_t<kChunk>(in.kr,(m+1)*kChunk,0,lane);
            g0 = in.gt[(m+1)*kChunk+group];
            g1 = in.gt[(m+1)*kChunk+group+8];
        }
        #pragma unroll
        for (int bi = 0; bi < 2; ++bi) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                Pair old{state[bi].x[j]}, next;
                float decay = j%2 == 0 ? decay0 : decay1;
                next.h[0] = BF16(bf_float(old.h[0])*decay+u[bi].x[2*j]);
                next.h[1] = BF16(bf_float(old.h[1])*decay+u[bi].x[2*j+1]);
                state[bi].x[j] = next.u;
            }
            store_t<kDim>(s.state,state[bi],m*kChunk,value+bi*kChunk,lane);
            if (m+1 < kDim/kChunk) state[bi] = load_t<kDim>(s.state,(m+1)*kChunk,value+bi*kChunk,lane);
        }
    }
```

---
skill_id: kda.state-decay-register-reuse
intent: Reuse per-key state-decay coefficients across a thread's state elements.
preconditions:
- 'Data types: repeated coefficient evaluations must produce identical values for
  identical inputs under the same floating-point mode; reuse must preserve the coefficient
  representation and consumer rounding.'
- 'Layout: a thread must own multiple state elements using the same known, in-bounds
  key coefficient; otherwise reuse supplies the wrong key or saves no evaluations.
  No additional contiguity or alignment is required.'
- 'Storage: log-decays must be readable ordinary memory without required per-read
  side effects; replacing repeated reads with one read must not discard observable
  behavior. No particular memory level or additional shared storage is required.'
- 'Pipeline: producers must complete and make log-decays visible before the hoisted
  reads; values must remain unchanged through their consumers, and buffer reuse must
  wait for readers to finish, so cached coefficients cannot become stale.'
- 'Hardware: the existing scalar operations suffice; the cached coefficients and other
  live values must fit the thread register allocation to retain register reuse without
  spills. No new instruction feature is required.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Cache each distinct state-decay coefficient once per thread and key block,
then reuse it for the thread's state elements sharing that key.

In `solution/recurrence.cuh`, replace the final `m` loop in `recur_tile` with
the After snippet. Immediately after loading `kr`, compute `decay0` and
`decay1`. In the `j` loop, select by parity and reuse that FP32 value for both
BF16 elements. Delete the per-element loop and its local constants. This reduces
eight log-decay reads and exponentials to two per thread per key block.

The Before snippet uses `ld_scalar`, whose volatile shared loads prevent the
compiler from merging repeated reads and exponentials. The After snippet uses
ordinary `in.gt` reads. Retain `ld_scalar` and all its other call sites.

## Ownership and ordering

A warp owns value columns beginning at `value = warp*kChunk`; each lane uses
`group = lane/4`. For fragment `j` and element `e`, the key is
`m*kChunk + group + (j%2)*(kChunk/2)`. Its value column is
`value + (lane%4)*2 + (j/2)*8 + e`. Both elements in a pair and both fragments
of matching parity therefore share a key coefficient. Retain `load_t`,
`store_t`, and the row-major `state[value,key]` and `gt[key]` layouts.
All these indices are in bounds for the example; no guard or launch changes
are needed. Keep existing ABI checks and beta boundary zero fill.

`load_input` writes `s.input.gt`, then the CTA barrier in `recurrence` makes it
visible before `recur_tile`. No operation in `recur_tile` overwrites `gt`.
The following CTA barrier completes readers before exit or storage reuse;
caller-stream kernel ordering preserves the chunk handoff. Keep these barriers
and all warp barriers unchanged. The cached values are private to each thread;
no new synchronization or shared allocation is needed.

Retain `exp2_fast` and its FP32 result, the BF16-to-FP32 state conversion, the
existing multiply-add expression, and final BF16 conversion. Keep MMA product
order, intermediate quantization, and all compiler flags. Reuse changes no
reduction or rounding step.

## Example configuration

The workload is batch 1, 4096 tokens, 96 heads, and dimension 128. Chunks have
16 tokens; 256 sequential recurrence launches each use grid `(1,96)` and
256 threads. Each warp owns a 16-column value tile. Each lane has four packed
state registers with two BF16 elements each and eight FP32 accumulator entries.
Each of eight key blocks caches two FP32 coefficients per lane. Replay uses
key offsets `group` and `group+8` and fragment parity `j%2`.

Retain the CUDA SM90 target, BF16 tensor-core MMA with FP32 accumulation,
scalar fragment transfers, shared correction materialization, row-major shared
staging, and global BF16 state handoff. Keep structure sizes:
`PrepareShared=42368`, `InputShared=18048`, and `RecurShared=124672` bytes;
keep the separate 1024-byte transpose scratch. Preserve the existing build flags,
including `--use_fast_math`, launch bounds, and unrolling. These are replay
settings, not additional dependencies of coefficient reuse.

Inspect compiled `recurrence` code to confirm two coefficient loads and
exponentials per key block replace the eight repeated evaluations. Use the
existing evaluator with the unchanged problem for correctness and latency.

# Precondition

- Data types: identical log-decays must yield identical coefficients under the
  same floating-point mode. Store the existing coefficient representation and
  preserve every consumer's conversions and rounding; changing them would turn
  reuse into a numerical transformation. No particular dtype is intrinsic to reuse.
- Layout: each consuming thread must own multiple state elements with the same
  known, in-bounds key coefficient. The ownership-to-key mapping identifies the
  reusable value; a wrong mapping changes the state update. No additional
  contiguity or alignment is needed because loads remain scalar.
- Storage: source log-decays must reside in readable ordinary memory with no
  required per-read effects. One read can then replace repeated reads without
  changing behavior. The technique requires no particular memory level and adds
  no shared storage.
- Pipeline: producer completion and visibility must precede the hoisted reads.
  The source values must remain unchanged until all corresponding consumers
  finish; readers must complete before buffer reuse. Otherwise an early or stale
  coefficient can replace the required value. No fixed stage count is required.
- Hardware: reuse needs no feature beyond the existing scalar computation.
  For register reuse, live cached coefficients plus other live values must fit
  the thread's register allocation; spills would defeat the intended storage
  choice. The coefficient storage is `unique_coefficients_per_thread *
  sizeof(coefficient)`, independent of this example's tile size.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Both snippets replace only the final key-block loop in `recur_tile`; its
existing `s`, `in`, `lane`, `group`, `value`, and `u` provide the shared context.

## Before

```cuda
    #pragma unroll
    for (int m = 0; m < kDim/kChunk; ++m) {
        Reg kr = load_t<kChunk>(in.kr,m*kChunk,0,lane);
        u = Acc{};
        Reg correction = load_correction(s,value,lane);
        mma(u,kr,correction);
        Reg state = load_t<kDim>(s.state,m*kChunk,value,lane);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            Pair old{state.x[j]}, next;
            constexpr int kPairElems = sizeof(uint32_t) / sizeof(BF16);
            constexpr int kHalfChunk = kChunk / 2;
            #pragma unroll
            for (int e = 0; e < kPairElems; ++e) {
                // Reload per element so the compiler cannot reuse decay factors.
                float decay = exp2_fast(ld_scalar(in.gt+m*kChunk+group+(j%2)*kHalfChunk));
                next.h[e] = BF16(bf_float(old.h[e])*decay+u.x[kPairElems*j+e]);
            }
            state.x[j] = next.u;
        }
        store_t<kDim>(s.state,state,m*kChunk,value,lane);
    }
```

## After

```cuda
    #pragma unroll
    for (int m = 0; m < kDim/kChunk; ++m) {
        Reg kr = load_t<kChunk>(in.kr,m*kChunk,0,lane);
        float decay0 = exp2_fast(in.gt[m*kChunk+group]);
        float decay1 = exp2_fast(in.gt[m*kChunk+group+8]);
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
```

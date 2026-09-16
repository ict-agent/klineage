---
skill_id: kda.decay-operand-register-reuse
intent: Cache shared decay operands in registers for reuse across products.
preconditions:
- 'Data types: cached loads must preserve operand bits and leave arithmetic and rounding
  unchanged; reuse needs no particular floating-point format.'
- 'Layout: each thread repeatedly consumes known shared addresses for the same logical
  elements; changing the element between consumers would make the cached value incorrect.
  No additional contiguity or alignment is needed beyond the existing scalar loads.'
- 'Storage: the reused operands already reside in CTA shared memory; thread-private
  copies can replace their repeated shared loads without communication between threads.'
- 'Pipeline: producers must complete and publish operands before caching, operands
  must remain unchanged through their last consumer, and all participating threads
  must finish reads before storage reuse; otherwise the cached value can be premature
  or stale.'
- 'Hardware: thread-local register capacity under the launch''s allocation must retain
  cached operands through their reuse; spilling those operands defeats register residency.
  Unrelated values may spill. No additional accelerator instruction is required.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Cache shared key and log-gate operands once per thread, then reuse them across
decay products in `solution/prepare.cuh::prepare`. This removes repeated scalar
shared loads without sharing computed exponential factors.

Add `float rg[4][2]` and `BF16 rk[4][2]` beside `rgt` and `rq`. In the existing
first `m/n/j` gather, load `rg` before `rgt` and `rk` after `rq`. In the second
`m/n/j` loop, replace each load of `s.g` with `rg[t][j]` and each load of `s.k`
with `rk[t][j]`. The snippets show these three edit sites; keep both loop nests,
their unroll directives, and the surrounding stores.

Each thread retains its own operands. Keep row-major shared layouts, output
ownership, launch parameters, and all barriers. Normalization and cumulative
gate producers finish before the gather's existing CTA barriers. The intervening
barrier and the barrier after output stores remain. `s.k` and `s.g` stay unchanged
while disjoint `s.qd`, `s.kd`, `s.ki`, and `s.kr` receive results; no new buffer or
cross-thread exchange is introduced.

Keep scalar `ld_scalar`/`st_scalar` helpers and the noinline `decay_each` helper.
Each of the four products still calls `decay_each` separately. Preserve signs,
BF16 operand bits, intermediate BF16 multiply rounding, and multiplication order.
Do not cache decay factors or replace arithmetic instructions. Keep the ABI,
caller stream, complete problem, and compile flags.

Inspect compiled `prepare`: shared loads for a reused key and log gate should
occur once per owned element, with subsequent consumers using those values.
Check register residency and spills. Use the supplied Kernel evaluator for
correctness and timing; keep its evidence outside this card.

## Example configuration

The workload is B=1, T=4096, H=96, D=128. Preparation handles one 16-token chunk
and one head per CTA: grid `(256,96)`, 256 threads, launch bounds `(256,8)`,
41,856 dynamic shared bytes. Preserve the existing helper scratch allocations.
The recurrence retains 256 ordered launches, 256 threads per head, and 124,672
dynamic shared bytes per CTA.

The gather and consumer loops each use `m,n,j` in `[0,2)`, with
`t=m*2+n`, `r=m*8+warp`, `c=n*64+group*8+pair*2`, `warp=tid/32`,
`group=(tid%32)/4`, and `pair=(tid%32)%4`. Each warp owns two complete rows;
each thread owns eight elements. These coordinates exactly cover the full
16-by-128 tile, so no new bounds checks are needed. Preserve existing global
copy bounds and beta zero fill.

Shared `s.k` contains normalized BF16 keys and `s.g` FP32 cumulative log2 gates.
Their row strides are `kDim*sizeof(element)`. Cache eight elements from each:
eight FP32 values and eight BF16 values per thread. Keep single-use query and
terminal-gate gathers in `rq` and `rgt`. Retain `kScale`, `kGateScale`,
`kNormEps`, the BF16 preparation products, FP32 accumulators, tensor-core MMA,
scalar fragment adapters, shared reductions/transposes, triangular inversion,
and global chunk handoff. Use the existing SM90a compilation, fast-math flag,
register-usage setting, and noinline factor calls unchanged.

# Precondition

- Data types: retain each loaded operand's representation and every arithmetic
  operation's rounding. Caching only changes where an unchanged value is read;
  no particular floating-point format is necessary.
- Layout: repeated consumers in one thread must refer to the same logical
  element at known shared addresses. Reusing a value for another element is
  incorrect. Existing scalar-load alignment suffices; caching adds no
  contiguity or alignment constraint.
- Storage: the preceding implementation repeatedly reads operands already in
  CTA shared memory. Private cached values replace those reads; this recipe
  does not require another thread to access the cache.
- Pipeline: producers must publish completed values before caching. Inputs
  must remain unchanged through their last consumer, and storage reuse must
  wait for every participating reader. These requirements prevent premature
  reads and stale cached values. Existing barriers supply ordering; no stage
  count or new synchronization primitive is required.
- Hardware: thread-local register capacity under the launch's allocation
  must retain cached operands until their last reuse. Spilling those operands
  replaces intended register residency with memory traffic; unrelated values
  may spill. The transformation adds no accelerator instruction requirement.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

These are the declaration, first gather's `j` body, and second loop's `j` body,
respectively. Their existing loop variables, address mapping, and stores remain.

```cuda
float rgt[4][2];
BF16 rq[4][2];

// First gather's j body.
rgt[t][j] = ld_scalar(s.gt+c+j);
rq[t][j] = ld_scalar(s.q+r*kDim+c+j);

// Second loop's j body, after the existing CTA barrier.
qd.h[j] = rq[t][j] * decay_each(ld_scalar(s.g+r*kDim+c+j)) * BF16(kScale);
kd.h[j] = ld_scalar(s.k+r*kDim+c+j) * decay_each(ld_scalar(s.g+r*kDim+c+j));
ki.h[j] = ld_scalar(s.k+r*kDim+c+j) * decay_each(-ld_scalar(s.g+r*kDim+c+j));
kr.h[j] = ld_scalar(s.k+r*kDim+c+j) * decay_each(-ld_scalar(s.g+r*kDim+c+j)) * BF16(exp2_fast(rgt[t][j]));
```

## After

```cuda
float rg[4][2], rgt[4][2];
BF16 rq[4][2], rk[4][2];

// First gather's j body: retain the shared operands for all products.
rg[t][j] = ld_scalar(s.g+r*kDim+c+j);
rgt[t][j] = ld_scalar(s.gt+c+j);
rq[t][j] = ld_scalar(s.q+r*kDim+c+j);
rk[t][j] = ld_scalar(s.k+r*kDim+c+j);

// Second loop's j body: retain each independent factor evaluation.
qd.h[j] = rq[t][j] * decay_each(rg[t][j]) * BF16(kScale);
kd.h[j] = rk[t][j] * decay_each(rg[t][j]);
ki.h[j] = rk[t][j] * decay_each(-rg[t][j]);
kr.h[j] = rk[t][j] * decay_each(-rg[t][j]) * BF16(exp2_fast(rgt[t][j]));
```

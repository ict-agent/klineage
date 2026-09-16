---
skill_id: kda.prepare-query-gate-prefetch
intent: Prefetch shared query and chunk-gate operands ahead of decay preparation.
preconditions:
- 'Data types: no additional dtype restriction; scalar loads and private temporaries
  must preserve operand bits because this change only reschedules reads.'
- 'Layout: consumer operand addresses are known and in bounds before the intervening
  work, with stable thread ownership; incorrect ownership would supply another element.
  No alignment beyond the original scalar loads is needed.'
- 'Storage: operands already reside in CTA-shared memory and remain unchanged between
  the proposed early load and the original read; otherwise a prefetched snapshot could
  be stale.'
- 'Pipeline: operand producers complete and publish before early loads; all CTA threads
  still reach the intervening barrier, and shared storage is not reused until readers
  finish. Moving reads must not cross a producer or overwrite dependency.'
- 'Hardware: no additional feature beyond the existing CUDA scalar shared loads and
  compiler-managed private temporaries; early reads require no new collective instruction
  or shared-memory capacity.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

In `solution/prepare.cuh`, `prepare()`, load the query and chunk-total
log-gate operands ahead of the loop forming `qd`, `kd`, `ki`, and `kr`.
Keep each lane's operands in private arrays until its consumers run. This
separates shared-load latency from dependent decay arithmetic.

Insert the After gather loop immediately after the existing `lane`, `warp`,
`group`, and `pair` declarations, before their following `__syncthreads()`.
In the consumer loop, add `t = m*2+n`, replace the query load with `rq[t][j]`,
and replace the chunk-gate load with `rgt[t][j]`. The snippets show that region;
the four scalar stores and the barrier following the consumer loop stay intact.

The two barriers before these declarations already publish normalized `s.q`
and completed `s.gt`. Retain them and the barrier between gather and consume.
All CTA threads execute both loops. Separate shared arrays hold source operands
and destinations, so stores cannot invalidate the prefetched values. No launch,
shared-layout, buffer-reuse, or caller-stream change is needed.

Preserve `ld_scalar`'s volatile scalar accesses, all key and per-row log-gate
reloads, `decay_each` calls, `exp2_fast`, BF16 rounding points, and multiplication
order. Only the timing of query and chunk-gate reads changes. Keep the full
problem, oracle, workload, numerical tolerances, and compiler flags unchanged.
Inspect compiled `prepare` code: the query/gate loads should precede the
intervening CTA barrier and feed later arithmetic. Check correctness and latency
with `klineage.harness.evaluate`; keep evidence outside this card.

## Example configuration

Retain CUDA SM90, batch 1, 4096 tokens, 96 heads, head dimension 128, and
16-token chunks. `prepare` launches `grid=(256,96)`, 256 threads, and
41856 dynamic shared bytes, with `__launch_bounds__(kPrepareThreads,8)`.
Recurrence retains 256 ordered launches, 256 threads per CTA, and 124672
dynamic shared bytes. Static scratch, MMA fragments, chunk algebra, beta
conversion, and global carry remain unchanged.

`s.q` is a row-major BF16 `[kChunk,kDim]` matrix; `s.gt` is an FP32
`[kDim]` vector. Their element strides are `(kDim,1)` and `(1)`; byte
strides multiply by their element sizes. Each lane prefetches eight BF16
queries and eight FP32 gate values into `rq[4][2]` and `rgt[4][2]`.
`warp=tid/32`, `lane=tid%32`, `group=lane/4`, and `pair=lane%4` map
`r=m*8+warp`, `c=n*64+group*8+pair*2`, with `m,n,j` in `[0,2)`.
Thus `r<16` and `c+j<128`; this fixed workload needs no new bounds checks.
Each query has one consumer, and gate loads retain their existing multiplicity.

Keep the existing `-O3`, `--use_fast_math`, feature macros, ptxas register
settings, and line information. Preserve scalar BF16 operations and FP32
accumulation, `kScale`, and every existing conversion. No new compiler-control
flag or assembly helper is required.
Do not impose a no-spill requirement or change register settings; the compiler
may spill private temporaries without undoing the early-load schedule.

# Precondition

- Data types: no additional dtype restriction. Prefetching moves existing
  scalar-load results into private temporaries without arithmetic or conversion;
  changing their bits would change the consumers' numerical inputs.
- Layout: consumer addresses must be known and in bounds at the early load,
  with stable thread ownership through the later use. Incorrect ownership reads
  another element. The original scalar-load alignment suffices; vector alignment,
  a particular row extent, and a particular lane count are unnecessary.
- Storage: the operands must already be in CTA-shared memory and remain
  unchanged through their original read points. The optimization snapshots
  these values; intervening writes would make the snapshots stale.
- Pipeline: producers must complete and make operands visible before the
  gather. Every CTA thread must still reach the retained intervening barrier.
  Consumers must finish before shared storage is reused; the move cannot cross
  producer or overwrite dependencies. No additional stage count is required.
- Hardware: no additional feature is needed beyond the existing CUDA scalar
  shared loads and compiler-managed private temporaries. The schedule introduces
  no collective instruction or shared-memory allocation. Register capacity affects
  spilling and cost, but is not a prerequisite for moving these reads earlier.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

```cuda
int lane = tid % kWarp, warp = tid / kWarp;
int group = lane / 4, pair = lane % 4;
__syncthreads();
#pragma unroll
for (int m = 0; m < 2; ++m) {
    #pragma unroll
    for (int n = 0; n < 2; ++n) {
        int r = m*8+warp;
        int c = n*64+group*8+pair*2;
        Pair qd, kd, kr, ki;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            qd.h[j] = ld_scalar(s.q+r*kDim+c+j) * decay_each(ld_scalar(s.g+r*kDim+c+j)) * BF16(kScale);
            kd.h[j] = ld_scalar(s.k+r*kDim+c+j) * decay_each(ld_scalar(s.g+r*kDim+c+j));
            ki.h[j] = ld_scalar(s.k+r*kDim+c+j) * decay_each(-ld_scalar(s.g+r*kDim+c+j));
            kr.h[j] = ld_scalar(s.k+r*kDim+c+j) * decay_each(-ld_scalar(s.g+r*kDim+c+j)) * BF16(exp2_fast(ld_scalar(s.gt+c+j)));
        }
        // Retain the existing dst calculation and four scalar-store calls.
    }
}
```

## After

```cuda
int lane = tid % kWarp, warp = tid / kWarp;
int group = lane / 4, pair = lane % 4;
float rgt[4][2];
BF16 rq[4][2];
#pragma unroll
for (int m = 0; m < 2; ++m) {
    #pragma unroll
    for (int n = 0; n < 2; ++n) {
        int t = m*2+n;
        int r = m*8+warp;
        int c = n*64+group*8+pair*2;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            rgt[t][j] = ld_scalar(s.gt+c+j);
            rq[t][j] = ld_scalar(s.q+r*kDim+c+j);
        }
    }
}
__syncthreads();
#pragma unroll
for (int m = 0; m < 2; ++m) {
    #pragma unroll
    for (int n = 0; n < 2; ++n) {
        int t = m*2+n;
        int r = m*8+warp;
        int c = n*64+group*8+pair*2;
        Pair qd, kd, kr, ki;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            qd.h[j] = rq[t][j] * decay_each(ld_scalar(s.g+r*kDim+c+j)) * BF16(kScale);
            kd.h[j] = ld_scalar(s.k+r*kDim+c+j) * decay_each(ld_scalar(s.g+r*kDim+c+j));
            ki.h[j] = ld_scalar(s.k+r*kDim+c+j) * decay_each(-ld_scalar(s.g+r*kDim+c+j));
            kr.h[j] = ld_scalar(s.k+r*kDim+c+j) * decay_each(-ld_scalar(s.g+r*kDim+c+j)) * BF16(exp2_fast(rgt[t][j]));
        }
        // Retain the existing dst calculation and four scalar-store calls.
    }
}
```

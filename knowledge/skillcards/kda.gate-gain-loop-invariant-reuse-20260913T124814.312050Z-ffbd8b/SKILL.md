---
skill_id: kda.gate-gain-loop-invariant-reuse
intent: Reuse a loop-invariant gate gain across rows.
preconditions:
- 'Data types: evaluating an unchanged scalar must produce the same represented result
  under the required arithmetic and rounding; retaining that result must not change
  numerical behavior.'
- 'Layout: each consuming thread repeatedly addresses the same valid scalar across
  the loop; otherwise one cached result cannot replace its evaluations. No additional
  contiguity, alignment, or inter-thread mapping is required.'
- 'Storage: the repeated reads access ordinary device memory without required read
  side effects; caching cannot replace externally observable volatile reads.'
- 'Pipeline: producers must finish and make the scalar visible before the hoisted
  read, and its storage must remain unchanged until readers finish; otherwise hoisting
  can observe incomplete or stale data.'
- 'Hardware: no additional feature beyond scalar device execution is required; the
  reused scalar has thread-local lifetime and needs no shared buffer or collective
  instruction.'
scope:
  cases:
  - kda_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Hoist the head's gate-gain load and exponentiation out of the row loop in
`solution/prepare.cuh::prepare`. Each thread reuses its own result across rows.
This removes repeated global reads and exponentiations without changing gate
arithmetic or exchanging values between threads.

Insert `float gain = expf(a_log[head]);` immediately after
`load_prepare(s,args,tile,head);`, before the first `__syncthreads()`.
Remove the per-row `gain` declaration and its comment from the `tid < kDim`
branch. Keep `ld_global_scalar` in `solution/native.cuh`: the independent
per-row bias loads still use it. Its volatile PTX load currently prevents
compiler hoisting of the gain input; removing only that call permits reuse.
No compile-flag change is needed.

Keep all normalization code, barriers, gate accumulation, and consumers intact.
Each active gate thread owns one column and writes `s.g[r*kDim+tid]` in
increasing row order, then `s.gt[tid]`. Preserve the original multiplication,
addition, `sigmoid`, and rounding order; do not combine `gain` with other factors
or substitute another exponential. The earlier load reads only `a_log`, which
is ready on the caller stream before launch. Existing barriers still publish
staged gates and completed gate sums before their consumers; no buffer lifetime
or reuse changes.

Keep the ABI, launches, shared layouts, bounds handling, and workspace unchanged.
`head = blockIdx.y` remains valid for every launched thread, including threads
outside the gate branch. The forward snippets omit unchanged normalization code
between the initial barriers and gate branch; retain it verbatim.

## Example configuration

This replay uses batch 1, 4096 tokens, 96 heads, dimension 128, and 16-row chunks.
`prepare` launches `dim3(kTiles,kHeads)` with 256 threads and
`sizeof(PrepareShared) = 41856` dynamic shared bytes. Threads `tid < 128`
process 16 gate rows each. `a_log` is a contiguous FP32 array of 96 elements;
its byte offset is `head*sizeof(float)`. Gates are staged BF16 row-major
`[kChunk,kDim]`; gain, bias, gate accumulation, and gate sums use FP32.
Keep `kGateScale`, `--use_fast_math`, other compile flags, and the FP32
`expf` implementation unchanged.

Retain BF16 normalization rounding, XOR-grouped norm reductions, chunked
triangular inversion, MMA fragments, shared-memory staging, scalar fragment
transfers, per-row bias reloads, and the stream-ordered recurrence launches.
These settings and mechanisms are context, not prerequisites for scalar reuse.
Inspect generated code at the gain input and exponential: the hoisted value
should feed every gate row. Do not infer hoisting from latency alone.

# Precondition

- Data types: the scalar evaluation must return the same represented result for
  unchanged input under the required arithmetic and rounding. Reuse is valid
  only if retaining that result preserves numerical behavior; no particular
  scalar dtype is inherent to this technique.
- Layout: each consuming thread must address the same valid scalar throughout
  its loop. A row-dependent address would require different results. No added
  contiguity, alignment, or mapping between threads is needed for thread-local
  reuse.
- Storage: repeated reads must access ordinary device memory without required
  read side effects. The deoptimized volatile load is compiler control, not a
  requirement to observe an external device or changing producer on each read.
- Pipeline: the producer must finish and publish the scalar before the earlier
  read. Its storage must stay unchanged until all readers finish, including
  before buffer reuse; otherwise hoisting can use incomplete or stale data.
- Hardware: no additional feature beyond scalar device execution is needed.
  A thread-local scalar lifetime needs no shared-memory buffer, capacity
  relationship, or collective instruction.

# Scope

- Cases: kda_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

```cuda
load_prepare(s,args,tile,head);
__syncthreads();
__syncthreads();

// Keep the intervening normalization code and its barriers unchanged.

if (tid < kDim) {
    float sum = 0.0f;
    #pragma unroll
    for (int r = 0; r < kChunk; ++r) {
        // Reload the gain input per row to prevent compiler hoisting.
        float gain = expf(ld_global_scalar(a_log+head));
        // Reload each row's bias; volatile loads prevent register reuse.
        float bias = ld_global_scalar(args.bias+head*kDim+tid);
        float g = gain * (bf_float(s.gate[r*kDim+tid])+bias);
        sum += kGateScale * sigmoid(g);
        s.g[r*kDim+tid] = sum;
    }
    s.gt[tid] = sum;
}
```

## After

```cuda
load_prepare(s,args,tile,head);
float gain = expf(a_log[head]);
__syncthreads();
__syncthreads();

// Keep the intervening normalization code and its barriers unchanged.

if (tid < kDim) {
    float sum = 0.0f;
    #pragma unroll
    for (int r = 0; r < kChunk; ++r) {
        // Reload each row's bias; volatile loads prevent register reuse.
        float bias = ld_global_scalar(args.bias+head*kDim+tid);
        float g = gain * (bf_float(s.gate[r*kDim+tid])+bias);
        sum += kGateScale * sigmoid(g);
        s.g[r*kDim+tid] = sum;
    }
    s.gt[tid] = sum;
}
```

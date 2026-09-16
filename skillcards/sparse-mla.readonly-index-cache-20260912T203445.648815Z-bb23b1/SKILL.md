---
skill_id: sparse-mla.readonly-index-cache
intent: Cache immutable sparse indices through read-only global loads.
preconditions:
- 'Data types: the loaded type must have a representation-preserving __ldg overload;
  cache selection adds no arithmetic or rounding requirement.'
- 'Layout: each address must cover the selected load width and satisfy its natural
  alignment; __ldg retains each thread''s address and needs no additional contiguity
  or collective ownership.'
- 'Storage: the source must reside in device-accessible global memory, because __ldg
  cannot load shared or local storage.'
- 'Pipeline: source production must finish and become visible before kernel execution;
  the source must remain unchanged for the kernel''s lifetime, with reuse ordered
  after completion, because read-only caching does not provide coherence with concurrent
  writes. No new intermediate buffer is used.'
- 'Hardware: the CUDA target and compiler must support __ldg read-only global loads;
  no additional shared-memory capacity or fixed cache residency is required.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Use `__ldg` to request read-only caching for immutable sparse indices. In
`solution/attention.cuh`, inside `spa::produce`'s buffer/row loop, replace
`rows[offset]` with `__ldg(rows + offset)`. This is the only source change.
Retain `const int* rows = params.indices + token * kTopk` and the existing
offset, signed validity test, and widened stride multiplication.

Each producer lane still loads its own index into a register. Lanes sharing
`group = lane / kCopyGroup` read the same index; no lane election, broadcast,
shared staging, or launch change is introduced. An invalid index remains a
signed value: the existing predicate selects zero-filled KV copies and masks
its score. Every index-array access remains in bounds under the fixed launch.

Keep the caller stream and tensor ABI. Input producers must complete before
kernel execution, and index storage must remain immutable until completion.
Keep all existing KV free/ready arrivals and waits, mask publication, WGMMA
fences and waits, and probability-buffer lifetime rules. They continue to
protect the downstream copies and consumers; the index cache request replaces
none of them. Arithmetic, BF16 rounding, reduction grouping, softmax scale,
and all three outputs are unchanged.

The intrinsic requests a load policy; it does not promise a cache hit or a
speedup. Preserve the bundle's compile flags. Inspect generated instructions
to distinguish the ordinary load from the read-only load; CUDA can infer
read-only loads from other pointer facts. No compiler-control change is part
of this replay. See NVIDIA's [read-only load documentation](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#read-only-data-cache-load-function).

## Example configuration

Preserve `TOKENS=8192`, `HEADS=128`, `QK_DIM=576`, `VALUE_DIM=512`, and
`TOPK=2048`. Indices are contiguous int32 `[8192,1,2048]`, with byte strides
`(8192,8192,4)`. Q/KV and output use BF16; maximum and LSE use FP32.
The index policy has no dependency on those floating-point representations.

The launch has 16384 CTAs, 384 threads per CTA, and one CTA per cluster.
Each CTA owns one token and a 64-head slice. Two 128-thread consumer groups
and one 128-thread producer group remain. For producer-local `lane`, eight
lanes share each of 16 index groups. Each group visits four rows in each of
two buffers. The loop advances by two 64-key blocks across 32 blocks:
`offset = (block + buffer) * kTile + row * kCopyGroups + group`.
These bounds cover offsets 0 through 2047 without tail handling.

Keep both KV buffers and all four ordered transfer groups, the 16-byte
`cp.async.ca` copies, Q/KV swizzles, resident shared query, probability tiles,
and probability/KV storage alias. Dynamic shared storage remains 231376 bytes
per CTA.
Keep WGMMA, 32 score and 128 output accumulator elements per consumer thread,
shared reduction exchange, online softmax, and direct output stores.
Retain launch bounds `(kThreads,1,1)` and every compiler flag, including
`--use_fast_math` and ptxas register-usage level 10. The retained computation
targets SM90a; its hardware and capacity needs are separate from index caching.

# Precondition

- Data types: use a type supported by a representation-preserving `__ldg`
  overload. Unsupported types need another load implementation. This operation
  copies bits, so it adds no arithmetic precision or rounding requirement.
- Layout: each address must cover the chosen load width with its natural
  alignment; otherwise the load can be invalid or misaligned. Each thread
  keeps its original address. No extra contiguity, lane grouping, or collective
  participation is needed for the cache request.
- Storage: the source must be in device-accessible global memory, the address
  space served by `__ldg`. Shared and local sources cannot use this instruction.
- Pipeline: complete and publish source production before kernel execution.
  Keep the source unchanged for the entire kernel lifetime and order storage
  reuse after completion. Read-only caching cannot make concurrent writes
  coherent. This substitution creates no intermediate buffer, so it adds no
  intermediate-reader visibility or buffer-reuse barrier.
- Hardware: the target and compiler must implement `__ldg` read-only global
  loads; otherwise this intrinsic cannot supply the requested policy. It adds
  no shared-memory allocation. Cache misses remain valid, so the source need
  not fit in cache and no fixed cache capacity is a prerequisite.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

## Before

```cuda
const int offset = (block + buffer) * kTile + row * kCopyGroups + group;
const int index = rows[offset];
indices[buffer][row] = int64_t(index) * params.kv_stride;
valid[buffer][row] = index >= 0 && index < kTokens;
```

## After

```cuda
const int offset = (block + buffer) * kTile + row * kCopyGroups + group;
const int index = __ldg(rows + offset);
indices[buffer][row] = int64_t(index) * params.kv_stride;
valid[buffer][row] = index >= 0 && index < kTokens;
```

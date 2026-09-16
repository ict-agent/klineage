---
skill_id: cuda.async-kv-l1-bypass
intent: Bypass L1 caching for asynchronous global-to-shared KV copies.
preconditions:
- 'Data types: no additional dtype or arithmetic requirement; the cache qualifier
  changes where copied bits may be cached, without conversion or arithmetic.'
- 'Layout: each existing copy transfers 16 bytes between 16-byte-aligned addresses;
  cp.async.cg only supports this transaction size, and cp.async requires natural alignment.
  No additional contiguity or thread mapping is required.'
- 'Storage: existing copies read global memory into CTA shared memory; these are the
  address spaces accepted by cp.async.cg.shared.global. No additional storage or visibility
  requirement is introduced.'
- 'Pipeline: no additional ordering or participation requirement; preserve source
  publication, copy completion before consumers read, and reader completion before
  shared-buffer reuse. The cache hint supplies no synchronization.'
- 'Hardware: a CUDA target and toolchain supporting cp.async.cg (SM80 or newer); otherwise
  the instruction is unavailable. No additional shared-memory capacity is required
  because the buffers are unchanged.'
scope:
  cases:
  - sparse_mla_prefill
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Bypass L1 for asynchronous KV gathers by changing `cp.async.ca.shared.global`
to `cp.async.cg.shared.global` in `solution/hopper.cuh`, inside `copy_kv`.
This lets streaming KV traffic avoid L1 allocation while retaining L2 caching.
Its benefit depends on reuse and cache pressure. The qualifier is a performance
hint; it does not change memory consistency or copied values.
See NVIDIA's [cp.async specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async).

Replace only the instruction string shown below. Keep the helper signature,
operand constraints, `shared_addr`, byte count, and all callers unchanged.
`copy_tiles` in `solution/attention.cuh` supplies each producer lane's gathered
global address and swizzled shared destination. The helper changes cache policy
for every instantiated KV copy. Do not change query TMA loads or index loads.

Keep invalid-index handling: the source-byte argument is zero for invalid rows
and 16 for valid rows. The instruction still writes a full destination vector,
zero-filling invalid rows without reading their invalid source addresses.
Retain score masking and every arithmetic instruction, including BF16 rounding,
FP32 reductions, and softmax scaling. No launch, ownership, layout, or allocation
changes are needed.

Keep the four producer `free*.wait` / `copy_tiles` / `ready*.cp_arrive` sequences.
Consumers must still wait for matching ready phases; WGMMA completion must still
precede free-barrier arrivals. Retain probability publication fences, named
barriers, and mask synchronization. Changing caching neither completes copies
nor releases buffers.

After applying, build the frozen bundle with its existing flags. Inspect SASS:
the KV `LDGSTS` instructions should carry `BYPASS`; copy widths and zero-fill
predication remain unchanged. Validate with `klineage.harness.evaluate` using the
complete problem and existing tolerances and timing policy. Keep evidence outside
this card.

## Example configuration

Retain `TOKENS=8192`, `HEADS=128`, `QK_DIM=576`, `VALUE_DIM=512`, and `TOPK=2048`.
Inputs Q/KV and output use BF16; indices use int32; maxima, LSE, and accumulators
use FP32. Preserve `kScale=0.1352337788608801f`, all accumulation order, and
round-to-nearest BF16 conversions.

Q is contiguous `[token,head,576]`, KV `[token,1,576]`, indices
`[token,1,2048]`, and output `[token,head,512]`. KV row stride is 576 elements
(1152 bytes); `Params::kv_stride` stays a runtime field. Preserve the row-base
assembly fence and immediate tile offsets.

Each CTA owns one token and 64 heads. Launch 16384 CTAs with 384 threads and a
one-CTA cluster on the caller stream. The first two 128-thread warpgroups consume;
the third produces. Eight producer lanes cover one row with eight BF16 elements
(16 bytes) per lane. Sixteen lane groups cover four rows each per 64-row tile.
Keep `swizzle(row,col)` and the 64-by-64 tile layout. There are nine width tiles;
the four copy groups retain their existing buffer and half-width order.

Retain two KV buffers, resident Q, probability storage/reuse, the shared reduction
scratch, and `sizeof(Shared)=231376` bytes. Keep WGMMA QK/PV, online softmax,
overlapped producer/consumer scheduling, and direct output stores. Preserve
`config.toml`, the six-tensor destination-passing ABI, the SM90a build target,
`__launch_bounds__`, and all serialized compiler flags, including `-O3`,
`--use_fast_math`, and `--register-usage-level=10`.
These retained settings are not prerequisites for changing the cache qualifier.

# Precondition

- Data types: no additional dtype or arithmetic requirement. This change controls
  caching of raw copied bits; it neither converts representations nor reorders
  arithmetic. BF16 is an example setting, not a cache-policy dependency.
- Layout: existing transactions must copy 16 bytes between 16-byte-aligned source
  and destination addresses. The `.cg` form permits only that transaction size;
  natural alignment is required by the copy instruction. No additional row
  contiguity, swizzle, or lane ownership is required because addresses are unchanged.
- Storage: the existing transfer reads global memory and writes CTA shared memory.
  Other address spaces cannot be used with `cp.async.cg.shared.global`. There is
  no additional allocation or visibility requirement from the cache qualifier.
- Pipeline: no additional ordering or thread participation is required. Preserve
  publication of source data, asynchronous-copy completion before consumer reads,
  and consumer completion before destination reuse. Otherwise readers could see
  incomplete or overwritten data; caching supplies none of these guarantees.
- Hardware: the target and toolchain must support `cp.async.cg`, available on
  SM80 and newer. Unsupported targets cannot execute this instruction. No extra
  shared-memory capacity is needed: source/destination buffers and their lifetimes
  remain unchanged.

# Scope

- Cases: sparse_mla_prefill
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Replace only `copy_kv`'s cache qualifier in `solution/hopper.cuh`.
`Bf16` and `shared_addr` already exist; all call sites remain unchanged.

## Before

```cuda
__device__ __forceinline__ void copy_kv(
    const Bf16* src, Bf16* dst, int bytes) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(shared_addr(dst)), "l"(src), "r"(bytes));
}
```

## After

```cuda
__device__ __forceinline__ void copy_kv(
    const Bf16* src, Bf16* dst, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(shared_addr(dst)), "l"(src), "r"(bytes));
}
```

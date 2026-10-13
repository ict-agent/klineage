---
skill_id: topk.cluster-distributed-selection
intent: Distribute radix Top-K selection across a CTA cluster using distributed shared
  memory.
preconditions:
- 'Data types: histogram and reservation counts must support exact unsigned 32-bit
  additions without overflow; the supplied remote atomics operate on u32. Payload
  bits need no additional floating-point constraint, but output order and tied indices
  must be unspecified because reservations can reorder them.'
- 'Layout: input ownership must admit a disjoint, exhaustive partition, and shared-object
  offsets must be reproducible across CTAs so remote addresses name matching bins
  and counters. Paired state loads require adjacent 32-bit fields aligned to 8 bytes;
  otherwise ld.shared::cluster.u64 reads the wrong fields or is misaligned.'
- 'Storage: histogram, threshold state, and reservation counters must occupy shared
  memory at stable addresses; cluster address mapping cannot expose register-only
  or thread-local objects.'
- 'Pipeline: every thread must be able to reach uniform cluster barriers. Initialization
  must finish before cluster access, local histogram updates before merge reads, all
  histogram updates before threshold reads, state writes before peer reads, readers
  before reset, resets before reuse, and remote users before owner exit; otherwise
  shared data is uninitialized, raced, stale, or expired.'
- 'Hardware: CUDA cluster launch, distributed shared memory, cluster-scoped u32 atomics,
  u64 loads, and release/acquire cluster barriers are required. The chosen cluster
  must support concurrent residency, with its CTA thread, register, and shared-memory
  demands within device launch limits; otherwise the cluster cannot launch or access
  live peer storage.'
scope:
  cases:
  - topk
  languages:
  - cuda
  platforms:
  - nvidia-sm90a-cuda13
---

# Overview

Parallelize one Top-K selection across a CTA cluster. Each CTA builds a local
histogram; peer CTAs add their counts into rank zero's histogram through
distributed shared memory. Rank zero chooses the threshold and publishes its
state. All CTAs emit through rank zero's two atomic reservation counters.

Apply the focused replacements below in `solution/topk.cuh`. Add the native
primitives to `solution/native.cuh` inside `namespace native`, retaining its
includes, `kWarpMask`, and license. Insert `merge` before `choose`.

In `solution/kernel.cu`, enable nonportable cluster size in `prepare`, before
the retained carveout setting, and attach the cluster dimension to the existing
launch. Keep tensor checks, destination-passing ABI, device guard, caller stream,
error handling, and `config.toml` unchanged.

## Ownership and ordering

Set `kBlocks` to 16. The existing `kChunks` formula then assigns two chunks to
each CTA. Rank `r` owns chunk `r + chunk*kBlocks`; thread `t` visits
`t + n*kThreads` within that chunk. Only rank zero handles the alignment head;
only the last rank handles the tail. Keep the existing count rounding, all
histogram/emit call sites, and each original global index.

Initialize each CTA's shared state and counters using one elected thread. Insert
`native::sync()` after the initial `reset(sm)`, before the existing CTA barrier.
Keep the CTA barrier after local histogram updates; merge into rank zero, then
use a cluster barrier before rank zero calls `choose`. Keep the conditional CTA
barrier and histogram reset after `choose`. Replace the following CTA barrier
with a cluster barrier before reading rank zero's packed state. That barrier
publishes the decision and finishes all resets before histogram reuse. All CTAs
use the same published stop flag, so early termination remains uniform.

Seed only rank zero's tie counter after threshold selection, then insert a
cluster barrier before the existing CTA barrier and emission. Replace each local
output reservation with a remote atomic on rank zero's matching counter. Append
`native::sync()` after all emit calls, before returning, to keep remote storage
alive until every user finishes.

Preserve `ordered`, radix masks, splitter refinement, early stopping, scalar
loads, direct prefix summation, and the `out >= kSelected` guard. Values retain
their original bits and token indices their original offsets; reservation order
and which equal-valued tokens are selected may change. Do not alter the problem,
oracle, workload, or numerical requirements.

After replay, require an accepted `klineage.harness.evaluate` result on the
unchanged problem. Check complete input coverage and every barrier placement.

## Example configuration

Replay uses CUDA on `nvidia-sm90a-cuda13`, with one contiguous FP32 input row of
131072 scores, K=2048, FP32 selected values, and int64 indices. Row strides are
`S*sizeof(float)` for input and `K*sizeof(output_element)` for each output.
The workload requests largest values with unspecified order and tied indices.

Keep 512 threads per CTA, `__launch_bounds__(kThreads, 1)`, 11-bit radix digits,
2048 histogram bins, four bins per thread, and at most three passes (11/11/10
bits). Retain scalar global rereads and shared atomic histogram updates.
`Shared` stays 8216 bytes per CTA: 8192 histogram bytes, 16 state bytes, and
two 4-byte counters. `State` remains `alignas(8)` with consecutive fields
`candidates`, `remaining`, `bucket`, `stop`. No dynamic shared memory is added.

Keep 16 KiB chunks (4096 scores) and the 128-byte alignment peel. The single CTA
visits 32 chunks; the cluster visits two per CTA. Grid and cluster dimensions
become `(1,16,1)`; block dimensions stay `(512,1,1)`. Retain the 25% shared-memory
carveout and existing SM90 build target. The replay requires this device to admit
a nonportable 16-CTA cluster. Election uses one lane of the first 32-lane warp;
all threads still participate in barriers. These settings specify this replay,
not general limits of cluster selection.

# Precondition

- Data types: histogram counts and output reservations use exact unsigned
  32-bit additions without overflow, as required by the supplied u32 remote
  atomic instructions. Moving payload bits adds no floating-point requirement.
  Output order and tied indices must be unspecified: cross-CTA atomic arrival
  order changes both without changing the selected value multiset.
- Layout: input work must split into disjoint spans covering every element once;
  overlap double-counts keys and omissions lose candidates. Shared-object offsets
  must be reproducible across CTAs because mapped addresses use those offsets.
  The paired state reads require consecutive 32-bit fields at 8-byte alignment;
  otherwise a u64 load is misaligned or combines the wrong fields.
- Storage: histograms, threshold state, and allocation counters reside in shared
  memory at stable addresses. Cluster mapping addresses peer shared memory;
  register-only or thread-local state cannot serve these operations.
- Pipeline: all threads can reach uniform cluster barriers. Initialization
  finishes before cluster access; local histogram updates before merge reads;
  all histogram updates before threshold reads; state writes before peer reads;
  histogram readers before reset; resets before reuse; and remote users before
  owner exit. These dependencies prevent uninitialized or raced counts, stale
  decisions, and expired storage.
- Hardware: the device and CUDA toolchain provide cluster launch, distributed
  shared memory, cluster-scoped u32 atomics, u64 loads, and release/acquire
  cluster barriers. The chosen cluster must be concurrently resident, and each
  CTA's thread, register, and shared-memory requirements must fit launch limits.
  Otherwise launching the cluster or accessing live peer storage fails.

# Scope

- Cases: topk
- Languages: cuda
- Platforms: nvidia-sm90a-cuda13

# Code Change Snippet

Snippets show changed sites, not concatenated function bodies. Retain intervening
code. `counter`, `bit`, `splitter`, `pass`, `head`, and `tail` already exist.

## Before

```cuda
// solution/kernel.cu: prepare
const auto status = cudaFuncSetAttribute(topk::select,
    cudaFuncAttributePreferredSharedMemoryCarveout, kSharedCarveoutPercent);
TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));

// kernel launch: no cluster attribute
cudaLaunchConfig_t config{};
config.gridDim = dim3(1, topk::kBlocks, 1);
config.blockDim = dim3(topk::kThreads, 1, 1);
config.dynamicSmemBytes = 0;
config.stream = c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
// Existing cudaLaunchKernelEx and error checks follow.

// solution/topk.cuh: constant, then emit reservation
constexpr int kBlocks = 1;
const unsigned out = atomicAdd(counter, 1u);

// select: leader and ownership sites
const bool leader_thread = threadIdx.x == 0;
const int head_count = head;
const int tail_count = tail;
offsets[chunk] = head + chunk * kChunkItems;

// select: initial reset, after unchanged state/counter initialization
reset(sm);
__syncthreads();

// select: after each local histogram
__syncthreads();
choose(sm);
if (pass + 1 < kPasses) {
  __syncthreads();
  reset(sm);
}
__syncthreads();
const unsigned bucket = sm.state.bucket;
splitter |= bucket << bit;
final_bit = bit;
if (sm.state.stop) break;

// select: after the pass loop and its existing CTA barrier
const unsigned selected = kSelected - sm.state.remaining;
if (leader_thread) atomicAdd(&sm.tied, selected);
__syncthreads();
// Existing emit calls follow; no cluster exit barrier.
```

## After

```cuda
// solution/native.cuh: add inside namespace native
constexpr int kWarp = 32;

__device__ __forceinline__ unsigned address(const void* ptr) {
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ unsigned remote(const void* ptr, unsigned rank) {
  unsigned result;
  asm("mapa.shared::cluster.u32 %0, %1, %2;"
      : "=r"(result) : "r"(address(ptr)), "r"(rank));
  return result;
}

__device__ __forceinline__ void add_remote(unsigned ptr, unsigned value) {
  asm volatile("red.relaxed.cluster.shared::cluster.add.u32 [%0], %1;"
      : : "r"(ptr), "r"(value) : "memory");
}

__device__ __forceinline__ unsigned fetch_add_remote(unsigned ptr, unsigned value) {
  unsigned result;
  asm volatile("atom.relaxed.cluster.shared::cluster.add.u32 %0, [%1], %2;"
      : "=r"(result) : "r"(ptr), "r"(value) : "memory");
  return result;
}

__device__ __forceinline__ uint64_t load_remote(unsigned ptr) {
  uint64_t result;
  asm volatile("ld.shared::cluster.u64 %0, [%1];" : "=l"(result) : "r"(ptr));
  return result;
}

__device__ __forceinline__ void arrive() {
  asm volatile("barrier.cluster.arrive.release.aligned;" : : : "memory");
}

__device__ __forceinline__ void wait() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" : : : "memory");
}

__device__ __forceinline__ void sync() {
  arrive();
  wait();
}

__device__ __forceinline__ unsigned rank() {
  unsigned value;
  asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(value));
  return value;
}

__device__ __forceinline__ bool elect() {
  const unsigned warp = __shfl_sync(kWarpMask, threadIdx.x / kWarp, 0);
  if (warp != 0) return false;

  unsigned elected;
  asm volatile("{ .reg .pred p; elect.sync _|p, %1; selp.u32 %0, 1, 0, p; }"
      : "=r"(elected) : "r"(kWarpMask));
  return elected != 0;
}

// solution/kernel.cu: prepare
// Configure the nonportable cluster before the existing carveout setting.
auto status = cudaFuncSetAttribute(topk::select,
    cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
status = cudaFuncSetAttribute(topk::select,
    cudaFuncAttributePreferredSharedMemoryCarveout, kSharedCarveoutPercent);
TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));

// kernel launch: attribute storage stays live through cudaLaunchKernelEx.
cudaLaunchAttribute attr{};
attr.id = cudaLaunchAttributeClusterDimension;
attr.val.clusterDim = {1, topk::kBlocks, 1};
cudaLaunchConfig_t config{};
config.gridDim = dim3(1, topk::kBlocks, 1);
config.blockDim = dim3(topk::kThreads, 1, 1);
config.dynamicSmemBytes = 0;
config.stream = c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
config.attrs = &attr;
config.numAttrs = 1;
// Existing cudaLaunchKernelEx and error checks follow.

// solution/topk.cuh: constant, then new helper before choose
constexpr int kBlocks = 16;
__device__ __forceinline__ void merge(Shared& sm, unsigned rank, unsigned leader) {
  if (rank == 0) return;

  #pragma unroll
  for (int i = 0; i < kBucketsPerThread; ++i) {
    const int bucket = threadIdx.x + i * kThreads;
    const unsigned count = sm.hist[bucket];
    if (count != 0) native::add_remote(leader + bucket * sizeof(unsigned), count);
  }
}

// emit: replace only the reservation; retain counter choice and output guard.
const unsigned out = native::fetch_add_remote(native::remote(counter, 0), 1u);

// select: replace the leader declaration after __shared__ Shared sm;
const unsigned rank = native::rank();
const bool leader_thread = native::elect();
const unsigned leader_hist = native::remote(sm.hist, 0);
const unsigned leader_state = native::remote(&sm.state, 0);

// Replace ownership sites; retain head/tail and counts calculations.
const int head_count = rank == 0 ? head : 0;
const int tail_count = rank == kBlocks - 1 ? tail : 0;
offsets[chunk] = head + (rank + chunk * kBlocks) * kChunkItems;

// Initial reset, after unchanged per-CTA state/counter initialization
reset(sm);
native::sync();
__syncthreads();

// After each local histogram
__syncthreads();
merge(sm, rank, leader_hist);
native::sync();
if (rank == 0) choose(sm);
if (pass + 1 < kPasses) {
  __syncthreads();
  reset(sm);
}
native::sync();
const uint64_t result = native::load_remote(leader_state + 2 * sizeof(unsigned));
const unsigned bucket = static_cast<unsigned>(result);
splitter |= bucket << bit;
final_bit = bit;
if (result >> kKeyBits) break;

// After the pass loop and its existing CTA barrier
const uint64_t size = native::load_remote(leader_state);
const unsigned selected = kSelected - static_cast<unsigned>(size >> kKeyBits);
if (rank == 0 && leader_thread) atomicAdd(&sm.tied, selected);
native::sync();
__syncthreads();
// Keep all existing emit calls here.

// Before select returns, after all emit calls
native::sync();
```

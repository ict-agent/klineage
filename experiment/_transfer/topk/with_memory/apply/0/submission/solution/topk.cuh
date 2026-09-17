// Radix Top-K selection for one [1, S] FP32 score row, distributed over an
// SM90 CTA cluster with a resident shared-memory copy of the input.
//
// Layout of the algorithm
// -----------------------
//   sweep A : global -> shared cache (float4), then 11-bit radix histogram
//   merge   : remote `red` of each CTA histogram into rank 0's accumulator
//   choose  : rank 0 block-scan finds the bucket holding the Kth key
//   sweep B : prefix-filtered histogram of the next radix digit
//   sweep C : prefix-filtered histogram of the last radix digit
//   sweep D : local count of strict winners (A) and threshold ties (T)
//   exchange: suffix sums give every CTA a disjoint output interval
//   sweep E : emit values and token indices into the reserved slots
//
// `splitter` is the accumulated key prefix; lower encodings are better.
#pragma once

#include <cstdint>

#include "native.cuh"

namespace topk {

constexpr int kLength = 131072;
constexpr int kSelected = 2048;

constexpr int kThreads = 512;
constexpr int kBlocks = 16;
constexpr int kWarps = kThreads / 32;

constexpr int kChunkItems = 4096;
constexpr int kChunksPerCta = kLength / (kBlocks * kChunkItems);
constexpr int kCacheItems = kChunksPerCta * kChunkItems;

constexpr int kKeyBits = 32;
constexpr int kRadixBits = 11;
constexpr int kBuckets = 1 << kRadixBits;
constexpr int kBucketsPerThread = kBuckets / kThreads;
constexpr int kPasses = 3;

static_assert(kBlocks % 2 == 0);
static_assert(kCacheItems * kBlocks == kLength);
static_assert(kBuckets % kThreads == 0);

// Radix digit window per pass: bits [bit, bit + width).
constexpr int kPassBit[kPasses] = {21, 10, 0};
constexpr int kPassWidth[kPasses] = {11, 11, 10};
constexpr unsigned kPassMask[kPasses] = {0x7ffu, 0x7ffu, 0x3ffu};

static_assert(kPassBit[0] + kPassWidth[0] == kKeyBits);
static_assert(kPassBit[1] + kPassWidth[1] == kPassBit[0]);
static_assert(kPassBit[2] + kPassWidth[2] == kPassBit[1]);
static_assert(kPassBit[2] == 0);

// Order-preserving FP32 -> u32 map, inverted so that larger scores get smaller
// keys: the cumulative bucket scan then walks best-to-worst from bucket zero.
__device__ __forceinline__ unsigned ordered(float value) {
  const unsigned bits = __float_as_uint(value);
  const unsigned monotone =
      bits ^ (static_cast<unsigned>(static_cast<int>(bits) >> 31) | 0x80000000u);
  return ~monotone;
}

struct State {
  unsigned bucket;
  unsigned above;
  unsigned need;
  unsigned stop;
};

struct alignas(16) Shared {
  unsigned hist[kPasses * kBuckets];
  unsigned merge[kPasses * kBuckets];
  State state;
  unsigned selected;
  unsigned tied;
  unsigned warp_totals[2 * kWarps];
  unsigned above;                       // global strict-winner count
  unsigned exchange[2 * kBlocks];       // rank 0: per-CTA (A, T) tallies
  unsigned pairs[2 * kBlocks];          // every CTA: copy of `exchange`
  unsigned bases[2 * kBlocks];          // reserved output intervals
  alignas(16) float cache[kCacheItems];
};

static_assert(kCacheItems % 4 == 0);

// ===== shared histogram =====================================================

template <int Pass>
__device__ __forceinline__ void histogram(Shared& sm, unsigned splitter) {
  unsigned* hist = sm.hist + Pass * kBuckets;
  constexpr int kBit = kPassBit[Pass];
  constexpr unsigned kMask = kPassMask[Pass];
  constexpr int kPriorBit = Pass == 0 ? 0 : kPassBit[Pass - 1];

  #pragma unroll 1
  for (int i = threadIdx.x; i < kCacheItems; i += kThreads) {
    const unsigned key = ordered(sm.cache[i]);
    if (Pass != 0 && ((key >> kPriorBit) << kPriorBit) != splitter) continue;
    native::red_shared(hist + ((key >> kBit) & kMask));
  }
}

// rank 0: exclusive prefix over the merged histogram, pick the Kth bucket.
template <int Pass>
__device__ __forceinline__ void choose(Shared& sm) {
  const int tid = threadIdx.x;
  const unsigned need = sm.state.need;
  const unsigned above = sm.state.above;
  const unsigned* hist = sm.merge + Pass * kBuckets;

  // Contiguous ownership: thread `tid` holds buckets [tid*4, tid*4+4), so the
  // thread-order prefix scan below reproduces the bucket-order prefix.
  const int first = tid * kBucketsPerThread;
  unsigned counts[kBucketsPerThread];
  unsigned total = 0;
  #pragma unroll
  for (int i = 0; i < kBucketsPerThread; ++i) {
    counts[i] = hist[first + i];
    total += counts[i];
  }

  // Warp-level scan of the per-thread totals, then serially add earlier warps.
  unsigned inclusive = total;
  #pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1)
    inclusive = native::scan_step(inclusive, offset);
  if ((tid % 32) == 31) sm.warp_totals[tid / 32] = inclusive;
  __syncthreads();

  unsigned running = inclusive - total;
  for (int warp = 0; warp < tid / 32; ++warp) running += sm.warp_totals[warp];

  // The bucket that holds the `need`-th best key; every earlier bucket is
  // strictly better and is fully selected. `running >= need` means the crossing
  // bucket lies in an earlier thread's range, which owns the update.
  #pragma unroll
  for (int i = 0; i < kBucketsPerThread; ++i) {
    if (running >= need) break;
    if (running + counts[i] < need) {
      running += counts[i];
      continue;
    }
    sm.state.bucket = static_cast<unsigned>(first + i);
    sm.state.above = above + running;
    sm.state.need = need - running;
    sm.state.stop = (counts[i] == need - running) ? 1u : 0u;
    break;
  }
  __syncthreads();
}

// ===== one radix pass =======================================================

template <int Pass>
__device__ __forceinline__ void run_pass(Shared& sm, unsigned rank, unsigned& splitter,
    int& final_bit, bool& stop) {
  constexpr int kBit = kPassBit[Pass];

  histogram<Pass>(sm, splitter);
  __syncthreads();                      // local histogram complete (CTA-only)

  // Push this CTA's counts into rank 0's accumulator; later passes are sparse.
  const unsigned dst = native::remote(sm.merge + Pass * kBuckets, 0);
  const unsigned* hist = sm.hist + Pass * kBuckets;
  #pragma unroll
  for (int i = 0; i < kBucketsPerThread; ++i) {
    const int bucket = threadIdx.x + i * kThreads;
    const unsigned count = hist[bucket];
    if (count == 0) continue;
    native::red_remote(dst + bucket * sizeof(unsigned), count);
  }
  native::sync();                       // merged histogram complete

  if (rank == 0) choose<Pass>(sm);
  native::sync();                       // threshold published

  const unsigned base = native::remote(&sm.state, 0);
  const unsigned bucket = native::load_remote(base);
  stop = native::load_remote(base + 3 * sizeof(unsigned)) != 0;
  splitter |= bucket << kBit;
  final_bit = kBit;
}

// ===== emission =============================================================

__device__ __forceinline__ void block_sum2(Shared& sm, unsigned& first, unsigned& second) {
  const int tid = threadIdx.x;
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    first += __shfl_down_sync(native::kWarpMask, first, offset);
    second += __shfl_down_sync(native::kWarpMask, second, offset);
  }
  if ((tid % 32) == 0) {
    sm.warp_totals[tid / 32] = first;
    sm.warp_totals[kWarps + tid / 32] = second;
  }
  __syncthreads();
  if (tid == 0) {
    unsigned a = 0;
    unsigned b = 0;
    #pragma unroll
    for (int warp = 0; warp < kWarps; ++warp) {
      a += sm.warp_totals[warp];
      b += sm.warp_totals[kWarps + warp];
    }
    sm.warp_totals[0] = a;
    sm.warp_totals[kWarps] = b;
  }
  __syncthreads();
  first = sm.warp_totals[0];
  second = sm.warp_totals[kWarps];
}

__device__ __forceinline__ void emit(Shared& sm, unsigned rank, unsigned splitter,
    int final_bit, float* values, int64_t* indices) {
  const int tid = threadIdx.x;

  unsigned strict = 0;
  unsigned ties = 0;
  #pragma unroll 1
  for (int i = tid; i < kCacheItems; i += kThreads) {
    const unsigned key = (ordered(sm.cache[i]) >> final_bit) << final_bit;
    strict += key < splitter ? 1u : 0u;
    ties += key == splitter ? 1u : 0u;
  }
  block_sum2(sm, strict, ties);

  // Publish the tallies, then take descending-rank suffix sums so every CTA
  // owns a disjoint interval of both output regions.
  if (native::elect()) {
    const unsigned base = native::remote(&sm.exchange[2 * rank], 0);
    native::store_remote(base, strict);
    native::store_remote(base + sizeof(unsigned), ties);
  }
  native::sync();

  if (tid < kBlocks) {
    const unsigned base = native::remote(&sm.exchange[2 * tid], 0);
    sm.pairs[2 * tid] = native::load_remote(base);
    sm.pairs[2 * tid + 1] = native::load_remote(base + sizeof(unsigned));
  }
  __syncthreads();
  native::sync();                       // all rank-0 reads finished

  if (tid == 0) {
    unsigned running_strict = 0;
    unsigned running_ties = 0;
    #pragma unroll
    for (int r = kBlocks - 1; r >= 0; --r) {
      sm.bases[r] = running_strict;
      sm.bases[kBlocks + r] = sm.above + running_ties;
      running_strict += sm.pairs[2 * r];
      running_ties += sm.pairs[2 * r + 1];
    }
  }
  __syncthreads();
  sm.selected = sm.bases[rank];
  sm.tied = sm.bases[kBlocks + rank];
  __syncthreads();

  #pragma unroll 1
  for (int i = tid; i < kCacheItems; i += kThreads) {
    const float value = sm.cache[i];
    const unsigned key = (ordered(value) >> final_bit) << final_bit;
    if (key > splitter) continue;

    // Unique slot per candidate; excess threshold ties fall past kSelected.
    const unsigned out = key == splitter ? atomicAdd(&sm.tied, 1u) : atomicAdd(&sm.selected, 1u);
    if (out >= kSelected) continue;

    values[out] = value;
    indices[out] = static_cast<int64_t>(rank + (i / kChunkItems) * kBlocks) * kChunkItems
        + (i % kChunkItems);
  }
}

// ===== kernel ===============================================================

__device__ __forceinline__ void load_cache(Shared& sm, const float* input, unsigned rank) {
  constexpr int kVecPerChunk = kChunkItems / 4;
  #pragma unroll
  for (int chunk = 0; chunk < kChunksPerCta; ++chunk) {
    const float4* src = reinterpret_cast<const float4*>(
        input + (rank + chunk * kBlocks) * kChunkItems);
    float4* dst = reinterpret_cast<float4*>(sm.cache + chunk * kChunkItems);
    #pragma unroll 1
    for (int i = threadIdx.x; i < kVecPerChunk; i += kThreads) dst[i] = src[i];
  }
}

__global__ void __launch_bounds__(kThreads, 1) select(const float* __restrict__ input,
    float* __restrict__ values, int64_t* __restrict__ indices) {
  __shared__ Shared sm;
  const unsigned rank = native::rank();

  if (native::elect()) sm.state = {0u, 0u, static_cast<unsigned>(kSelected), 0u};
  #pragma unroll 1
  for (int i = threadIdx.x; i < kPasses * kBuckets; i += kThreads) {
    sm.hist[i] = 0u;
    sm.merge[i] = 0u;
  }

  // One global sweep: the whole row lands in shared and stays resident.
  load_cache(sm, input, rank);
  __syncthreads();

  unsigned splitter = 0u;
  int final_bit = 0;
  bool stop = false;

  run_pass<0>(sm, rank, splitter, final_bit, stop);
  if (!stop) run_pass<1>(sm, rank, splitter, final_bit, stop);
  if (!stop) run_pass<2>(sm, rank, splitter, final_bit, stop);

  if (native::elect())
    sm.above = native::load_remote(native::remote(&sm.state.above, 0));
  __syncthreads();

  emit(sm, rank, splitter, final_bit, values, indices);
}

}  // namespace topk

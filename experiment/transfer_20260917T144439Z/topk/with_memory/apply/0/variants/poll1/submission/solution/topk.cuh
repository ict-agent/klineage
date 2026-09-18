// Radix Top-K selection of 2048 FP32 indexer scores per decode request.
//
//   input row (131072 FP32)
//        |
//        v  one 16-CTA cluster per row; CTA r owns chunks r, r+16, ...
//   [16 KiB chunks] --bulk copy--> [shared resident cache]
//        |
//        v  radix pass p: 11/11/10 bits, smaller ordered key == larger score
//   local histogram (2048 bins) --remote add--> rank 0 histogram
//        |
//        v  rank 0: prefix scan, pick the bin holding the Kth key, publish it
//   splitter --cluster barrier--> every CTA refines the next digit
//        |
//        v  after the last pass: every CTA reserves disjoint output slots
//   [top_values | indices]
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "native.cuh"

namespace topk {

constexpr int kThreads = 512;
constexpr int kBlocks = 16;
constexpr int kSelected = 2048;
constexpr int kKeyBits = 32;
constexpr int kRadixBits = 11;
constexpr int kPasses = (kKeyBits + kRadixBits - 1) / kRadixBits;
constexpr int kBuckets = 1 << kRadixBits;
constexpr int kBucketsPerThread = kBuckets / kThreads;
constexpr int kChunkItems = 16 * 1024 / int(sizeof(float));  // 4096 scores
constexpr int kMaxChunks = 4;                                // 64 KiB cache per CTA
constexpr int kStages = 8;                                   // mbarrier slots
constexpr int kItems = 8;                                    // keys in flight per thread
constexpr int kAlignBytes = 128;
constexpr int kAlignItems = kAlignBytes / int(sizeof(float));
constexpr int kWarps = kThreads / native::kWarp;

constexpr int kDynamicBytes = kMaxChunks * kChunkItems * int(sizeof(float)) + kAlignBytes;
constexpr int kMaxStaticBytes = 48 * 1024;  // static shared needs no opt-in below this

struct alignas(8) State {
  unsigned candidates;
  unsigned remaining;
  unsigned bucket;
  unsigned stop;
};

// Warp totals used by the hierarchical bucket-prefix scan.
struct alignas(32) Scan {
  unsigned warps[kWarps];
};

struct Shared {
  unsigned hist[kBuckets];
  State state;
  unsigned selected;        // first free slot among strictly better keys
  unsigned tied;            // first free slot among cutoff ties
  unsigned local_selected;  // this CTA's strictly better key count
  unsigned local_tied;      // this CTA's cutoff tie count
  uint64_t decision;        // rank 0 ballot: (generation, stop, remaining, bucket)
  Scan scan;
  uint64_t barriers[kStages];
  float edges[2 * kAlignItems];  // staged misaligned head and tail
};
static_assert(sizeof(Shared) <= kMaxStaticBytes, "static shared must not need opt-in");

// Rank 0 ballot: generation, stop flag, residual rank, chosen bucket.
constexpr int kBallotBucket = 0;
constexpr int kBallotRemaining = 24;
constexpr int kBallotStop = 44;
constexpr int kBallotGeneration = 48;
constexpr unsigned kBallotMask = (1u << 20) - 1u;

__device__ __forceinline__ uint64_t ballot(int pass, const State& state) {
  return (uint64_t(pass + 1) << kBallotGeneration) | (uint64_t(state.stop) << kBallotStop)
      | (uint64_t(state.remaining) << kBallotRemaining) | uint64_t(state.bucket);
}

// Order-reversing key: a larger score yields a smaller unsigned encoding, so the
// radix passes refine towards the smallest encodings. Negatives keep their bits
// (more negative == larger), positives flip the magnitude bits.
__device__ __forceinline__ unsigned ordered(float value) {
  const unsigned bits = __float_as_uint(value);
  return bits ^ (((bits >> 31) - 1u) & 0x7fffffffu);
}

__device__ constexpr int start_bit(int pass) {
  return kKeyBits - (pass + 1) * kRadixBits > 0 ? kKeyBits - (pass + 1) * kRadixBits : 0;
}

__device__ __forceinline__ void reset(Shared& sm) {
  #pragma unroll
  for (int i = 0; i < kBucketsPerThread; ++i) sm.hist[threadIdx.x + i * kThreads] = 0;
}

// Exclusive prefix of this thread's four buckets, over the whole CTA histogram.
__device__ __forceinline__ void scan(Shared& sm,
    unsigned (&counts)[kBucketsPerThread], unsigned (&prefixes)[kBucketsPerThread]) {
  const int tid = threadIdx.x;
  const int lane = tid % native::kWarp;
  const int warp = tid / native::kWarp;

  unsigned total = 0;
  #pragma unroll
  for (int i = 0; i < kBucketsPerThread; ++i) {
    counts[i] = sm.hist[tid * kBucketsPerThread + i];
    total += counts[i];
  }

  unsigned inclusive = total;
  #pragma unroll
  for (int step = 1; step < native::kWarp; step *= 2) {
    const unsigned other = native::scan_step(inclusive, step);
    if (lane >= step) inclusive += other;
  }
  if (lane == native::kWarp - 1) sm.scan.warps[warp] = inclusive;
  __syncthreads();

  unsigned warp_prefix = 0;
  unsigned running = sm.scan.warps[0];
  #pragma unroll
  for (int w = 1; w < kWarps; ++w) {
    if (warp == w) warp_prefix = running;
    running += sm.scan.warps[w];
  }

  unsigned prefix = inclusive - total + warp_prefix;
  #pragma unroll
  for (int i = 0; i < kBucketsPerThread; ++i) {
    prefixes[i] = prefix;
    prefix += counts[i];
  }
}

// Rank 0: locate the bucket holding the Kth key and publish it.
__device__ __forceinline__ void choose(Shared& sm) {
  const unsigned target = sm.state.remaining;
  unsigned counts[kBucketsPerThread];
  unsigned prefixes[kBucketsPerThread];
  scan(sm, counts, prefixes);

  #pragma unroll
  for (int i = 0; i < kBucketsPerThread; ++i) {
    if (prefixes[i] >= target || prefixes[i] + counts[i] < target) continue;

    const unsigned left = target - prefixes[i];
    sm.state.candidates = counts[i];
    sm.state.remaining = left;
    sm.state.bucket = threadIdx.x * kBucketsPerThread + i;
    sm.state.stop = counts[i] == left;  // the bucket exactly fills the selection
  }
}

// Push this CTA's local histogram into rank 0's bins.
__device__ __forceinline__ void merge(Shared& sm, unsigned rank, unsigned leader_hist) {
  if (rank == 0) return;

  #pragma unroll
  for (int i = 0; i < kBucketsPerThread; ++i) {
    const unsigned bucket = threadIdx.x + i * kThreads;
    const unsigned count = sm.hist[bucket];
    if (count != 0) native::add_remote(leader_hist + bucket * sizeof(unsigned), count);
  }
}

enum class Pass { first, later };

// Count keys per radix digit; later passes keep only keys matching the splitter.
template <Pass mode>
__device__ __forceinline__ void histogram(Shared& sm, const float* keys, int count,
    unsigned splitter, int prior_bit, int bit, unsigned mask) {
  #pragma unroll 1
  for (int base = 0; base < count; base += kItems * kThreads) {
    float regs[kItems];
    #pragma unroll
    for (int i = 0; i < kItems; ++i) {
      const int index = base + i * kThreads + threadIdx.x;
      if (index < count) regs[i] = keys[index];
    }

    #pragma unroll
    for (int i = 0; i < kItems; ++i) {
      const int index = base + i * kThreads + threadIdx.x;
      if (index >= count) continue;
      const unsigned bits = ordered(regs[i]);
      if constexpr (mode == Pass::later) {
        if (((bits >> prior_bit) << prior_bit) != splitter) continue;
      }
      atomicAdd(sm.hist + ((bits >> bit) & mask), 1u);
    }
  }
}

// Write every selected key into the interval this CTA reserved.
__device__ __forceinline__ void emit(Shared& sm, const float* keys, int count,
    int global_base, unsigned splitter, int bit, float* values, int64_t* indices) {
  #pragma unroll 1
  for (int base = 0; base < count; base += kItems * kThreads) {
    float regs[kItems];
    unsigned bits[kItems];
    #pragma unroll
    for (int i = 0; i < kItems; ++i) {
      const int index = base + i * kThreads + threadIdx.x;
      if (index >= count) continue;
      regs[i] = keys[index];
      bits[i] = (ordered(regs[i]) >> bit) << bit;
    }

    #pragma unroll
    for (int i = 0; i < kItems; ++i) {
      const int index = base + i * kThreads + threadIdx.x;
      if (index >= count || bits[i] > splitter) continue;
      unsigned* counter = bits[i] == splitter ? &sm.tied : &sm.selected;
      const unsigned out = atomicAdd(counter, 1u);
      if (out >= kSelected) continue;
      values[out] = regs[i];
      indices[out] = static_cast<int64_t>(global_base + index);
    }
  }
}

__global__ void __launch_bounds__(kThreads, 1) select(const float* __restrict__ input,
    float* __restrict__ values, int64_t* __restrict__ indices, int length) {
  __shared__ Shared sm;

  // Resident copy of this CTA's chunks; the dynamic region is 128-byte aligned.
  extern __shared__ char dynamic[];
  float* keys = reinterpret_cast<float*>(
      (reinterpret_cast<uintptr_t>(dynamic) + (kAlignBytes - 1)) & ~uintptr_t(kAlignBytes - 1));

  const unsigned rank = native::rank();
  const bool leader = native::elect();
  const int tid = threadIdx.x;

  // Peel the misaligned head so every chunk start is 128-byte aligned.
  const int head = int((kAlignBytes - (reinterpret_cast<uintptr_t>(input) & (kAlignBytes - 1)))
      & (kAlignBytes - 1)) / int(sizeof(float));
  const int body = length - head;
  const int chunks = body / kChunkItems;
  const int tail = body - chunks * kChunkItems;

  // Rank r owns global chunks r, r + kBlocks, ...; head and tail go to the ends.
  int offsets[kMaxChunks];
  int owned = 0;
  for (int c = 0; c < kMaxChunks; ++c) {
    const int chunk = int(rank) + c * kBlocks;
    if (chunk >= chunks) break;
    offsets[c] = head + chunk * kChunkItems;
    ++owned;
  }

  const int head_count = rank == 0 ? head : 0;
  const int tail_count = rank == kBlocks - 1 ? tail : 0;
  const bool tail_cached = tail_count <= kAlignItems;
  const float* head_src = sm.edges;
  const float* tail_src = tail_cached ? sm.edges + kAlignItems : input + length - tail_count;

  const unsigned leader_hist = native::remote(sm.hist, 0);
  const unsigned leader_decision = native::remote(&sm.decision, 0);

  if (leader) {
    sm.state = {static_cast<unsigned>(length), kSelected, 0u, 0u};
    sm.selected = 0;
    sm.tied = 0;
    sm.local_selected = 0;
    sm.local_tied = 0;
    sm.decision = 0;
  }
  reset(sm);
  if (tid < kStages) native::init_barrier(sm.barriers + tid);
  native::arrive();  // publish the histogram reset to the cluster
  __syncthreads();

  // Stage the misaligned head and tail; full chunks arrive through bulk copies.
  if (head_count != 0 && tid < head_count) sm.edges[tid] = input[tid];
  if (tail_count != 0 && tail_cached && tid < tail_count)
    sm.edges[kAlignItems + tid] = input[length - tail_count + tid];
  if (leader) {
    for (int c = 0; c < owned; ++c)
      native::copy_async(keys + c * kChunkItems, input + offsets[c],
          unsigned(kChunkItems * sizeof(float)), sm.barriers + c);
  }
  __syncthreads();

  for (int c = 0; c < owned; ++c) {
    native::wait_copy(sm.barriers + c);
    histogram<Pass::first>(sm, keys + c * kChunkItems, kChunkItems, 0u, 0, start_bit(0),
        kBuckets - 1);
  }
  histogram<Pass::first>(sm, head_src, head_count, 0u, 0, start_bit(0), kBuckets - 1);
  histogram<Pass::first>(sm, tail_src, tail_count, 0u, 0, start_bit(0), kBuckets - 1);

  unsigned splitter = 0;
  unsigned remaining = kSelected;
  int final_bit = 0;
  for (int pass = 0; pass < kPasses; ++pass) {
    const int bit = start_bit(pass);
    const int prior_bit = start_bit(pass - 1);
    const unsigned mask = (1u << (prior_bit - bit)) - 1u;

    if (pass != 0) {
      // The cached chunks are contiguous in shared memory, so one call covers them.
      histogram<Pass::later>(sm, keys, owned * kChunkItems, splitter, prior_bit, bit, mask);
      histogram<Pass::later>(sm, head_src, head_count, splitter, prior_bit, bit, mask);
      histogram<Pass::later>(sm, tail_src, tail_count, splitter, prior_bit, bit, mask);
    }
    __syncthreads();
    if (pass == 0) native::wait();  // the reset arrived at initialization
    merge(sm, rank, leader_hist);
    native::arrive();

    // Nonleaders scan their own bins while the cluster reduction completes.
    unsigned counts_local[kBucketsPerThread] = {};
    unsigned prefixes[kBucketsPerThread] = {};
    if (rank != 0) scan(sm, counts_local, prefixes);
    native::wait();
    if (rank == 0) choose(sm);

    if (pass + 1 < kPasses) {
      __syncthreads();
      reset(sm);  // rank 0 clears the bins it merges into
    }
    if (rank == 0) {
      __syncthreads();
      if (leader) native::store_release(&sm.decision, ballot(pass, sm.state));
    }
    const uint64_t result = native::poll(leader_decision, pass + 1);
    remaining = unsigned((result >> kBallotRemaining) & kBallotMask);
    const unsigned bucket = static_cast<unsigned>(result & kBallotMask);
    if (rank != 0 && tid == int(bucket) / kBucketsPerThread) {
      const int slot = int(bucket) % kBucketsPerThread;
      atomicAdd(&sm.local_selected, prefixes[slot]);
      sm.local_tied = counts_local[slot];
    }
    splitter |= bucket << bit;
    final_bit = bit;
    if ((result >> kBallotStop) & 1u) break;  // the bucket fills the selection
  }
  __syncthreads();

  // Seed every CTA's disjoint output interval: descending CTA rank, then order.
  const unsigned strict = kSelected - remaining;
  if (leader) atomicAdd(&sm.tied, strict);
  if (sm.local_selected != 0 || sm.local_tied != 0) {
    if (tid < int(rank)) {
      const unsigned lower = native::remote(&sm.selected, tid);
      native::add_remote(lower, sm.local_selected);
      native::add_remote(lower + sizeof(unsigned), sm.local_tied);
    }
  }
  native::sync();
  __syncthreads();

  for (int c = 0; c < owned; ++c)
    emit(sm, keys + c * kChunkItems, kChunkItems, offsets[c], splitter, final_bit, values,
        indices);
  emit(sm, head_src, head_count, 0, splitter, final_bit, values, indices);
  emit(sm, tail_src, tail_count, length - tail_count, splitter, final_bit, values, indices);
}
}  // namespace topk

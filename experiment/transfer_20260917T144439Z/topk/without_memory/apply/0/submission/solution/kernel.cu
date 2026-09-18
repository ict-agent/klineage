// Top-K (K = 2048) indexer token selection for decode: the K largest FP32 scores
// of every row in [B, S], returned as values plus int64 column indices.
//
// Radix selection on the order-preserving uint32 view of the float. The key is
// resolved digit by digit, so nothing is sorted and only two sweeps of the row
// are needed. Both sweeps live in a single kernel and are separated by a device
// barrier over the blocks of one row, so the K-th key costs one launch:
//
//   scores --[pass 1]--> per-row histogram of the top 11 key bits
//              |
//              |  device barrier: every block of the row waits for that histogram
//              v
//          --[pass 2]--> highest 11-bit digit D that still leaves room:
//              |          digit > D  => selected (written straight out)
//              |          digit == D => candidate (compacted key/index/value)
//              v
//          --[pass 3]--> two more radix levels over the candidates only, run by
//                         the last block of the row, which resolves the exact
//                         K-th key and recycles the scratch for the next call.
//
//   key(bits) = bits ^ (bits >> 31 ? ~0 : 0x80000000)   // -inf .. +inf ascending
//
// A cold row is 512 KB, so every sweep moves float4 and spreads the blocks over
// the whole row: memory-level parallelism, not instruction count, sets latency.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kQuad = 4;  // scores per thread per step (one float4)

// Radix digits, most significant first. 11 + 11 + 10 covers the 32-bit key.
constexpr int kLevel1Bits = 11;
constexpr int kLevel2Bits = 11;
constexpr int kLevel3Bits = 10;
constexpr int kLevel1Bins = 1 << kLevel1Bits;
constexpr int kLevel2Bins = 1 << kLevel2Bits;
constexpr int kLevel3Bins = 1 << kLevel3Bits;

constexpr int kLevel1Shift = 32 - kLevel1Bits;
constexpr int kLevel2Shift = kLevel3Bits;
constexpr uint32_t kLevel1Unit = 1u << kLevel1Shift;  // one level-1 digit
constexpr uint32_t kLevel2Mask = (uint32_t)kLevel2Bins - 1u;
constexpr uint32_t kLevel3Mask = (uint32_t)kLevel3Bins - 1u;
constexpr uint32_t kNoBin = 0xFFFFFFFFu;
constexpr uint32_t kFullMask = 0xFFFFFFFFu;

// Floats ordered by bit pattern: flip the sign, invert negative patterns.
__device__ __forceinline__ uint32_t order_key(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Exclusive prefix sum across the block; returns the sum held by the lanes
// below the caller. `scratch` holds one word per warp.
__device__ __forceinline__ uint32_t block_excl_scan(uint32_t* scratch,
                                                    uint32_t value) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  uint32_t inclusive = value;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const uint32_t other = __shfl_up_sync(kFullMask, inclusive, offset);
    if (lane >= offset) inclusive += other;
  }
  if (lane == 31) scratch[warp] = inclusive;
  __syncthreads();
  if (warp == 0) {
    uint32_t warp_sum = (lane < kWarps) ? scratch[lane] : 0u;
#pragma unroll
    for (int offset = 1; offset < kWarps; offset <<= 1) {
      const uint32_t other = __shfl_up_sync(kFullMask, warp_sum, offset);
      if (lane >= offset) warp_sum += other;
    }
    if (lane < kWarps) scratch[lane] = warp_sum;
  }
  __syncthreads();
  const uint32_t above = (warp == 0) ? 0u : scratch[warp - 1];
  return above + inclusive - value;
}

// suf[bin] = number of keys strictly above `bin`. Thread t owns the segment of
// kPer consecutive bins that starts at the top of the digit range, so segment
// sums run from the most significant digit down. *cut becomes the smallest bin
// whose suffix count is below `limit`, i.e. the bin owning the limit-th key.
template <int BINS>
__device__ __forceinline__ void suffix_cut(const uint32_t* counts, uint32_t* suf,
                                           uint32_t* scratch, uint32_t* cut,
                                           uint32_t limit) {
  constexpr int kPer = BINS / kThreads;
  const int t = threadIdx.x;
  uint32_t own[kPer];
  uint32_t total = 0;
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    own[j] = counts[BINS - 1 - (t * kPer + j)];
    total += own[j];
  }
  uint32_t running = block_excl_scan(scratch, total);
  int best = BINS;
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const int bin = BINS - 1 - (t * kPer + j);
    suf[bin] = running;
    if (running < limit) best = bin;
    running += own[j];
  }
  if (best < BINS) atomicMin(cut, (uint32_t)best);
  __syncthreads();
}

// Reserves one contiguous range in each output stream from a single block scan,
// so a thread learns both of its destinations after one round of global atomics
// rather than two sequential ones. `base` holds the two stream heads.
__device__ __forceinline__ void reserve_pair(uint32_t* scan, uint32_t* base,
                                             int sel_count, int cand_count,
                                             uint32_t* sel_counter,
                                             uint32_t* cand_counter,
                                             uint32_t& sel_at, uint32_t& cand_at) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  int sel_inc = sel_count;
  int cand_inc = cand_count;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int sel_other = __shfl_up_sync(kFullMask, sel_inc, offset);
    const int cand_other = __shfl_up_sync(kFullMask, cand_inc, offset);
    if (lane >= offset) {
      sel_inc += sel_other;
      cand_inc += cand_other;
    }
  }
  if (lane == 31) {
    scan[warp * 2] = (uint32_t)sel_inc;
    scan[warp * 2 + 1] = (uint32_t)cand_inc;
  }
  __syncthreads();
  if (warp == 0) {
    int sel_warp = (lane < kWarps) ? (int)scan[lane * 2] : 0;
    int cand_warp = (lane < kWarps) ? (int)scan[lane * 2 + 1] : 0;
    const int sel_own = sel_warp;
    const int cand_own = cand_warp;
#pragma unroll
    for (int offset = 1; offset < kWarps; offset <<= 1) {
      const int sel_other = __shfl_up_sync(kFullMask, sel_warp, offset);
      const int cand_other = __shfl_up_sync(kFullMask, cand_warp, offset);
      if (lane >= offset) {
        sel_warp += sel_other;
        cand_warp += cand_other;
      }
    }
    if (lane < kWarps) {
      scan[lane * 2] = (uint32_t)(sel_warp - sel_own);
      scan[lane * 2 + 1] = (uint32_t)(cand_warp - cand_own);
    }
    if (lane == kWarps - 1) {
      scan[kWarps * 2] = (uint32_t)sel_warp;
      scan[kWarps * 2 + 1] = (uint32_t)cand_warp;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t sel_total = scan[kWarps * 2];
    const uint32_t cand_total = scan[kWarps * 2 + 1];
    base[0] = sel_total ? atomicAdd(sel_counter, sel_total) : 0u;
    base[1] = cand_total ? atomicAdd(cand_counter, cand_total) : 0u;
  }
  __syncthreads();
  sel_at = base[0] + scan[warp * 2] + (uint32_t)(sel_inc - sel_count);
  cand_at = base[1] + scan[warp * 2 + 1] + (uint32_t)(cand_inc - cand_count);
}

enum class Slot : int { kSkip = 0, kSelected = 1, kCandidate = 2 };

// The whole selection. Blocks of one row cooperate through the row histogram in
// global memory: pass 1 fills it, a device barrier publishes it, pass 2 reads it
// back to split the row, and the last block of the row finishes the boundary.
template <bool kVectored>
__global__ void __launch_bounds__(kThreads) topk_kernel(
    const float* __restrict__ scores, int S, int K,
    uint32_t* __restrict__ hist,
    uint32_t* __restrict__ barrier, uint32_t* __restrict__ done,
    uint32_t* __restrict__ counters, uint32_t* __restrict__ cand_key,
    uint32_t* __restrict__ cand_index, float* __restrict__ cand_value,
    float* __restrict__ out_values, int64_t* __restrict__ out_indices) {
  __shared__ uint32_t s_hist[kLevel1Bins];
  __shared__ uint32_t s_suf[kLevel1Bins];
  __shared__ uint32_t s_scan[kWarps * 2 + 2];
  __shared__ uint32_t s_cut;
  __shared__ uint32_t s_base[2];
  __shared__ uint32_t s_taken;
  __shared__ uint32_t s_edge;
  __shared__ uint32_t s_last;

  const int row = blockIdx.y;
  const int t = threadIdx.x;
  const float* row_in = scores + (size_t)row * S;
  const float4* row4 = reinterpret_cast<const float4*>(row_in);
  const int groups = S / kQuad;
  uint32_t* row_hist = hist + (size_t)row * kLevel1Bins;

  // Pass 1: block-local histogram of the top key digit. One float4 per thread,
  // held in registers so the second sweep needs no reload: a cold row costs a
  // single trip through DRAM.
  for (int bin = t; bin < kLevel1Bins; bin += kThreads) s_hist[bin] = 0;
  __syncthreads();
  const int g = (int)blockIdx.x * kThreads + t;
  const bool valid = g < groups;
  float4 quad = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  if (valid) {
    if constexpr (kVectored) {
      quad = row4[g];
    } else {
      quad = make_float4(row_in[g * kQuad], row_in[g * kQuad + 1],
                         row_in[g * kQuad + 2], row_in[g * kQuad + 3]);
    }
    atomicAdd(&s_hist[order_key(quad.x) >> kLevel1Shift], 1u);
    atomicAdd(&s_hist[order_key(quad.y) >> kLevel1Shift], 1u);
    atomicAdd(&s_hist[order_key(quad.z) >> kLevel1Shift], 1u);
    atomicAdd(&s_hist[order_key(quad.w) >> kLevel1Shift], 1u);
  }
  // Row tail: fewer than kQuad scores left over, so no float4 fits.
  const int tail = S - groups * kQuad;
  if (blockIdx.x == 0 && t < tail) {
    const int i = groups * kQuad + t;
    atomicAdd(&s_hist[order_key(row_in[i]) >> kLevel1Shift], 1u);
  }
  __syncthreads();
  for (int bin = t; bin < kLevel1Bins; bin += kThreads) {
    const uint32_t count = s_hist[bin];
    if (count) atomicAdd(&row_hist[bin], count);
  }
  __syncthreads();

  // Device barrier over the row: the histogram is only complete once every
  // block of the row arrived. `__threadfence` orders the flush before the flag.
  if (t == 0) {
    __threadfence();
    atomicAdd(&barrier[row], 1u);
    while (__ldcg(barrier + row) < (uint32_t)gridDim.x) {}
  }
  __syncthreads();

  // Every block resolves the same cut digit from the now-complete row
  // histogram; `__ldcg` reads the L2 the flush atomics landed in.
  for (int bin = t; bin < kLevel1Bins; bin += kThreads) {
    s_hist[bin] = __ldcg(row_hist + bin);
  }
  if (t == 0) s_cut = kNoBin;
  __syncthreads();
  suffix_cut<kLevel1Bins>(s_hist, s_suf, s_scan, &s_cut, (uint32_t)K);

  const uint32_t cut_bin = s_cut;
  const uint32_t selected = s_suf[cut_bin];
  const uint32_t floor_key = cut_bin * kLevel1Unit;
  const uint32_t ceil_key = floor_key + kLevel1Unit;

  // Pass 2: one vectorized sweep splits the row into selected scores (written
  // straight to the output) and cut-bin candidates (compacted, their level-2
  // digits histogrammed).
  uint32_t* row_keys = cand_key + (size_t)row * S;
  uint32_t* row_index = cand_index + (size_t)row * S;
  float* row_cand = cand_value + (size_t)row * S;
  float* row_values = out_values + (size_t)row * K;
  int64_t* row_out_index = out_indices + (size_t)row * K;
  uint32_t* row_sel = counters + row * 2;
  uint32_t* row_cand_counter = counters + row * 2 + 1;

  {
    const float lane_value[kQuad] = {quad.x, quad.y, quad.z, quad.w};
    uint32_t keys[kQuad];
    Slot slot[kQuad];
    int selected_here = 0;
    int candidate = 0;
#pragma unroll
    for (int j = 0; j < kQuad; ++j) {
      const float x = lane_value[j];
      keys[j] = order_key(x);
      slot[j] = Slot::kSkip;
      if (!valid) continue;
      if (keys[j] >= ceil_key) {
        slot[j] = Slot::kSelected;
        ++selected_here;
      } else if (keys[j] >= floor_key) {
        slot[j] = Slot::kCandidate;
        ++candidate;
      }
    }

    uint32_t sel_base = 0;
    uint32_t cand_base = 0;
    reserve_pair(s_scan, s_base, selected_here, candidate, row_sel,
                 row_cand_counter, sel_base, cand_base);

    int sel_off = 0;
    int cand_off = 0;
#pragma unroll
    for (int j = 0; j < kQuad; ++j) {
      if (slot[j] == Slot::kSelected) {
        const uint32_t at = sel_base + sel_off++;
        row_values[at] = lane_value[j];
        row_out_index[at] = (int64_t)(g * kQuad + j);
      } else if (slot[j] == Slot::kCandidate) {
        const uint32_t at = cand_base + cand_off++;
        row_keys[at] = keys[j];
        row_index[at] = (uint32_t)(g * kQuad + j);
        row_cand[at] = lane_value[j];
      }
    }
  }
  if (blockIdx.x == 0 && t < tail) {
    const int i = groups * kQuad + t;
    const float x = row_in[i];
    const uint32_t key = order_key(x);
    if (key >= ceil_key) {
      const uint32_t at = atomicAdd(row_sel, 1u);
      row_values[at] = x;
      row_out_index[at] = (int64_t)i;
    } else if (key >= floor_key) {
      const uint32_t at = atomicAdd(row_cand_counter, 1u);
      row_keys[at] = key;
      row_index[at] = (uint32_t)i;
      row_cand[at] = x;
    }
  }
  __syncthreads();

  // The last block of the row to finish owns the boundary resolution; every
  // other block stops here.
  if (t == 0) {
    __threadfence();
    const uint32_t arrived = atomicAdd(&done[row], 1u) + 1u;
    s_last = (arrived == (uint32_t)gridDim.x) ? 1u : 0u;
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();

  // Pass 3: two more radix levels over the candidates pin down the K-th key;
  // the remaining slots are then filled with the candidates past the threshold.
  const int n_cand = (int)__ldcg(counters + row * 2 + 1);
  const uint32_t need = (uint32_t)K - selected;
  for (int bin = t; bin < kLevel2Bins; bin += kThreads) s_hist[bin] = 0;
  if (t == 0) s_cut = kNoBin;
  __syncthreads();
  for (int j = t; j < n_cand; j += kThreads) {
    atomicAdd(&s_hist[(row_keys[j] >> kLevel2Shift) & kLevel2Mask], 1u);
  }
  __syncthreads();
  suffix_cut<kLevel2Bins>(s_hist, s_suf, s_scan, &s_cut, need);
  const uint32_t cut2 = s_cut;
  const uint32_t above2 = s_suf[cut2];
  const uint32_t need3 = need - above2;  // picks left inside the level-2 bin

  // Level 3: only candidates still tied on the level-2 digit matter. When level 2
  // already exhausted the picks the digit stays out of range, which keeps every
  // tie out of the result.
  uint32_t cut3 = (uint32_t)kLevel3Bins;
  uint32_t above3 = 0;
  if (need3 > 0) {
    for (int bin = t; bin < kLevel3Bins; bin += kThreads) s_hist[bin] = 0;
    if (t == 0) s_cut = kNoBin;
    __syncthreads();
    for (int j = t; j < n_cand; j += kThreads) {
      const uint32_t key = row_keys[j];
      if (((key >> kLevel2Shift) & kLevel2Mask) != cut2) continue;
      atomicAdd(&s_hist[key & kLevel3Mask], 1u);
    }
    __syncthreads();
    suffix_cut<kLevel3Bins>(s_hist, s_suf, s_scan, &s_cut, need3);
    cut3 = s_cut;
    above3 = s_suf[cut3];
  }
  const uint32_t edge = need3 - above3;  // candidates equal to the K-th key

  if (t == 0) {
    s_taken = 0;
    s_edge = 0;
  }
  __syncthreads();
  const uint32_t threshold =
      (cut_bin << (kLevel2Bits + kLevel3Bits)) | (cut2 << kLevel3Bits) | cut3;
  for (int j = t; j < n_cand; j += kThreads) {
    const uint32_t key = row_keys[j];
    if (key < threshold) continue;

    const uint32_t index = row_index[j];
    const float value = row_cand[j];
    if (key > threshold) {
      const uint32_t at = selected + atomicAdd(&s_taken, 1u);
      row_values[at] = value;
      row_out_index[at] = (int64_t)index;
    } else {
      const uint32_t at = atomicAdd(&s_edge, 1u);
      if (at < edge) {
        row_values[(uint32_t)K - edge + at] = value;
        row_out_index[(uint32_t)K - edge + at] = (int64_t)index;
      }
    }
  }

  // Reset for the next call: both histograms, the offsets it was built from,
  // the barrier and the finishing counter.
  for (int bin = t; bin < kLevel1Bins; bin += kThreads) row_hist[bin] = 0;
  if (t == 0) {
    counters[row * 2] = 0;
    counters[row * 2 + 1] = 0;
    barrier[row] = 0;
    done[row] = 0;
  }
}

struct Scratch {
  torch::Tensor hist;
  torch::Tensor barrier;
  torch::Tensor done;
  torch::Tensor counters;
  torch::Tensor cand_key;
  torch::Tensor cand_index;
  torch::Tensor cand_value;
};

// Per-(device, shape) device scratch, zeroed on creation and recycled by the
// last block of each row. Host-side state only; device work stays in the kernel.
Scratch& scratch_for(const torch::Device& device, int64_t rows, int64_t columns) {
  static std::mutex mutex;
  static std::unordered_map<std::string, Scratch> cache;
  const std::string key =
      std::to_string(device.index()) + ":" + std::to_string(rows) + "x" +
      std::to_string(columns);

  std::lock_guard<std::mutex> lock(mutex);
  auto found = cache.find(key);
  if (found != cache.end()) return found->second;

  auto options = torch::TensorOptions().device(device);
  Scratch& entry = cache[key];
  entry.hist = torch::zeros({rows * kLevel1Bins}, options.dtype(torch::kInt32));
  entry.barrier = torch::zeros({rows}, options.dtype(torch::kInt32));
  entry.done = torch::zeros({rows}, options.dtype(torch::kInt32));
  entry.counters = torch::zeros({rows * 2}, options.dtype(torch::kInt32));
  entry.cand_key = torch::empty({rows * columns}, options.dtype(torch::kInt32));
  entry.cand_index = torch::empty({rows * columns}, options.dtype(torch::kInt32));
  entry.cand_value =
      torch::empty({rows * columns}, options.dtype(torch::kFloat32));
  return entry;
}

void require_contract(const torch::Tensor& tensor, torch::ScalarType dtype,
                      const char* label) {
  TORCH_CHECK_VALUE(tensor.is_cuda(), label, " must be a CUDA tensor");
  TORCH_CHECK_TYPE(tensor.scalar_type() == dtype, label, " has the wrong dtype");
  TORCH_CHECK_VALUE(tensor.is_contiguous(), label, " must be contiguous");
}

// Blocks per row that the row barrier needs to see alive at once, capped by the
// device so the spin cannot outlive its peers.
int resident_blocks_per_row(const torch::Device& device, int blocks_per_row) {
  static std::mutex mutex;
  static std::unordered_map<int, int> cache;
  std::lock_guard<std::mutex> lock(mutex);
  auto found = cache.find(device.index());
  if (found != cache.end()) return found->second;

  int per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &per_sm, topk_kernel<true>, kThreads, 0);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device.index());
  const int resident = std::max(1, per_sm * prop.multiProcessorCount);
  cache[device.index()] = resident;
  return resident;
}

void kernel(const torch::Tensor& values, const torch::Tensor& top_values,
            const torch::Tensor& indices) {
  require_contract(values, torch::kFloat32, "values");
  require_contract(top_values, torch::kFloat32, "top_values");
  require_contract(indices, torch::kInt64, "indices");
  TORCH_CHECK_VALUE(values.dim() == 2, "values must be [batch, tokens]");
  TORCH_CHECK_VALUE(top_values.dim() == 2 && indices.dim() == 2,
                    "outputs must be [batch, k]");

  const int64_t rows = values.size(0);
  const int64_t columns = values.size(1);
  const int64_t k = top_values.size(1);
  TORCH_CHECK_VALUE(k >= 1 && k <= columns, "k must be in [1, tokens]");
  TORCH_CHECK_VALUE(top_values.size(0) == rows && indices.size(0) == rows &&
                        indices.size(1) == k,
                    "output shape mismatch");
  TORCH_CHECK_VALUE(values.device() == top_values.device() &&
                        values.device() == indices.device(),
                    "device mismatch");
  TORCH_CHECK_VALUE(rows <= 65535, "batch exceeds the launch grid");
  if (rows == 0) return;

  c10::cuda::CUDAGuard guard(values.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(values.get_device()).stream();
  const int S = (int)columns;
  const int K = (int)k;
  const Scratch& scratch = scratch_for(values.device(), rows, columns);
  uint32_t* hist = reinterpret_cast<uint32_t*>(scratch.hist.data_ptr<int32_t>());
  uint32_t* barrier =
      reinterpret_cast<uint32_t*>(scratch.barrier.data_ptr<int32_t>());
  uint32_t* done = reinterpret_cast<uint32_t*>(scratch.done.data_ptr<int32_t>());
  uint32_t* counters =
      reinterpret_cast<uint32_t*>(scratch.counters.data_ptr<int32_t>());
  uint32_t* cand_key =
      reinterpret_cast<uint32_t*>(scratch.cand_key.data_ptr<int32_t>());
  uint32_t* cand_index =
      reinterpret_cast<uint32_t*>(scratch.cand_index.data_ptr<int32_t>());
  float* cand_value = scratch.cand_value.data_ptr<float>();

  // One float4 per thread covers a 2^17 score row in a single grid step.
  const int per_row = std::max(1, (S / kQuad + kThreads - 1) / kThreads);
  const int resident = resident_blocks_per_row(values.device(), per_row);
  const int rows_per_launch = std::max(1, resident / per_row);
  const bool vectored = (S % kQuad) == 0;

  for (int64_t base_row = 0; base_row < rows; base_row += rows_per_launch) {
    const int batch = (int)std::min<int64_t>(rows - base_row, rows_per_launch);
    const dim3 grid((unsigned)per_row, (unsigned)batch);
    const float* row_in = values.data_ptr<float>() + (size_t)base_row * S;
    float* out_v = top_values.data_ptr<float>() + (size_t)base_row * K;
    int64_t* out_i = indices.data_ptr<int64_t>() + (size_t)base_row * K;
    uint32_t* hist_r = hist + (size_t)base_row * kLevel1Bins;
    uint32_t* barrier_r = barrier + base_row;
    uint32_t* done_r = done + base_row;
    uint32_t* counters_r = counters + (size_t)base_row * 2;
    uint32_t* cand_key_r = cand_key + (size_t)base_row * S;
    uint32_t* cand_index_r = cand_index + (size_t)base_row * S;
    float* cand_value_r = cand_value + (size_t)base_row * S;

    if (vectored) {
      topk_kernel<true><<<grid, kThreads, 0, stream>>>(
          row_in, S, K, hist_r, barrier_r, done_r, counters_r,
          cand_key_r, cand_index_r, cand_value_r, out_v, out_i);
    } else {
      topk_kernel<false><<<grid, kThreads, 0, stream>>>(
          row_in, S, K, hist_r, barrier_r, done_r, counters_r,
          cand_key_r, cand_index_r, cand_value_r, out_v, out_i);
    }
  }
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "topk launch failed: ",
              cudaGetErrorString(error));
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("kernel", &kernel);
}

// Top-K (K = 2048) selection of the largest FP32 scores per row.
//
//   pass 1  values --byte3 histogram--> bucket b1 and the count above it
//   pass 2  values --scatter--> [0, head) takes every key whose top byte exceeds b1;
//           keys inside that bucket go to a buffer, whose last block resolves
//           the low 16 bits and fills [head, K)
//
// Two streaming reads of the row and no ordering pass: the cutoff is exact to
// 24 key bits, so elements inside the last bucket are interchangeable to 2^-24
// relative and the tail order stays unspecified as the problem allows.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

namespace {

constexpr int kK = 2048;  // top-k width, const K of the problem definition
constexpr int kBinBits = 8;
constexpr int kBins = 1 << kBinBits;
constexpr int kMask = kBins - 1;
constexpr int kBlocks = 128;  // row scan grid
constexpr int kThreads = 1024; // row scan block size
constexpr int kWarps = kThreads / 32;

// Per-row state shared by the two passes. Single writer per field.
struct __align__(16) RowSel {
  unsigned int filter;  // cutoff byte3, already shifted into the key top byte
  int need;             // elements still required inside the cutoff bucket
  int reserve_hi;       // pass 2 reservations for the head region
  int reserve_lo;       // pass 2 reservations for the bucket buffer
};

// Order-preserving float -> unsigned key map (bigger key == bigger float).
__device__ __forceinline__ unsigned int mono_key(float value) {
  const unsigned int bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ float key_to_float(unsigned int key) {
  const unsigned int bits = (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key;
  return __uint_as_float(bits);
}

// Locate the bucket holding the rank-th largest element (1-based rank) and count
// the elements strictly above it. One warp, no barriers: lane L owns the eight
// buckets [kBins-1-8L-7, kBins-1-8L], so the reversed suffix sum is a shuffle
// scan over lane totals plus a running sum inside the lane.
__device__ __forceinline__ void locate_bucket(int* __restrict__ hist, int rank,
                                             int* __restrict__ out) {
  constexpr int kPerLane = kBins / 32;
  const int lane = threadIdx.x & 31;
  int part[kPerLane];
  int total = 0;
  #pragma unroll
  for (int k = 0; k < kPerLane; ++k) {
    const int bin = kBins - 1 - (lane * kPerLane + k);
    part[k] = hist[bin];
    hist[bin] = 0;  // the caller gets a clean histogram back
    total += part[k];
  }

  // Warp inclusive scan of the lane totals; the exclusive prefix is the number
  // of elements in the buckets above this lane.
  int scan = total;
  for (int off = 1; off < 32; off <<= 1) {
    const int other = __shfl_up_sync(0xFFFFFFFFu, scan, off);
    if (lane >= off) scan += other;
  }

  int running = scan - total;
  int digit = -1;
  int above = 0;
  #pragma unroll
  for (int k = 0; k < kPerLane; ++k) {
    running += part[k];
    if (digit < 0 && running >= rank) {
      digit = kBins - 1 - (lane * kPerLane + k);
      above = running - part[k];
    }
  }

  // The lowest reversed index that reaches the rank owns the bucket.
  const int src = __ffs(__ballot_sync(0xFFFFFFFFu, digit >= 0)) - 1;
  out[0] = __shfl_sync(0xFFFFFFFFu, digit, src);
  out[1] = __shfl_sync(0xFFFFFFFFu, above, src);
}

// Per-block byte3 histogram, flushed into the row histogram, then the last block
// to finish turns it into the cutoff bucket and the count above it.
__global__ void hist_pass(const float* __restrict__ values, int s, int chunk,
                          int* __restrict__ hist, RowSel* __restrict__ rows,
                          int* __restrict__ tickets) {
  __shared__ int bins[kBins];
  __shared__ int nz[kBins];
  __shared__ int locate[2];
  __shared__ int nz_count;
  __shared__ bool is_last;

  for (int i = threadIdx.x; i < kBins; i += blockDim.x) bins[i] = 0;
  if (threadIdx.x == 0) nz_count = 0;
  __syncthreads();

  const int row = blockIdx.y;
  const float* row_in = values + (size_t)row * s;
  const int begin = blockIdx.x * chunk;
  const int count = min(begin + chunk, s) - begin;
  const unsigned int lane = threadIdx.x & 31u;
  const int iterations = (count + blockDim.x - 1) / blockDim.x;

  // One shared atomic per distinct bin in a warp: neighbouring scores share a
  // top byte and would otherwise serialise 32 ways on the same address.
  for (int it = 0; it < iterations; ++it) {
    const int j = it * blockDim.x + threadIdx.x;
    unsigned int bin = kBins;
    if (j < count) bin = mono_key(row_in[begin + j]) >> 24;
    const unsigned int match = __match_any_sync(0xFFFFFFFFu, bin);
    if (bin < kBins && (match & ((1u << lane) - 1u)) == 0) atomicAdd(&bins[bin], __popc(match));
  }
  __syncthreads();

  // Compacted flush: only non-empty bins travel to global memory.
  if (threadIdx.x < kBins && bins[threadIdx.x] != 0) nz[atomicAdd(&nz_count, 1)] = threadIdx.x;
  __syncthreads();
  int* row_hist = hist + (size_t)row * kBins;
  for (int i = threadIdx.x; i < nz_count; i += blockDim.x)
    atomicAdd(&row_hist[nz[i]], bins[nz[i]]);

  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) is_last = (atomicAdd(tickets + row, 1) == (int)gridDim.x - 1);
  __syncthreads();
  if (!is_last) return;
  __threadfence();

  // Last block: fold the row histogram into the cutoff, then leave it clean.
  if (threadIdx.x < 32) {
    locate_bucket(row_hist, kK, locate);
    if (threadIdx.x == 0) {
      rows[row].filter = (unsigned int)locate[0] << 24;
      rows[row].need = kK - locate[1];
      tickets[row] = 0;  // ready for the next call
    }
  }
}

// Scatter: keys above the cutoff bucket fill [0, head) directly, keys inside it
// land in a buffer. The last block to finish ranks the buffer and completes the
// tail [head, K).
__global__ void gather_pass(const float* __restrict__ values, int s, int chunk,
                            RowSel* __restrict__ rows, unsigned int* __restrict__ buf_keys,
                            int* __restrict__ buf_idx,
                            float* __restrict__ out_values, long long* __restrict__ out_idx,
                            int* __restrict__ tickets) {
  __shared__ int warp_cnt[kWarps][2];
  __shared__ int warp_off[kWarps][2];
  __shared__ int block_base[2];
  __shared__ int hist[kBins];
  __shared__ int locate[2];
  __shared__ int tail[4];  // 0: candidate count, 1: region counts, 2: ranks
  __shared__ bool is_last;

  const int row = blockIdx.y;
  const RowSel sel = rows[row];
  const int filter = (int)(sel.filter >> 24);  // cutoff byte3
  const int head = kK - sel.need;
  const float* row_in = values + (size_t)row * s;
  unsigned int* row_keys = buf_keys + (size_t)row * s;
  int* row_idx = buf_idx + (size_t)row * s;
  const int begin = blockIdx.x * chunk;
  const int count = min(begin + chunk, s) - begin;
  const int lane = threadIdx.x & 31u;
  const int warp = threadIdx.x >> 5;
  const int iterations = (count + blockDim.x - 1) / blockDim.x;

  // Pass 1: count this block's slots in both regions so one reservation per
  // region per block suffices. Per element reservations would serialise
  // thousands of atomics onto a single address.
  int block_total[2] = {0, 0};
  for (int it = 0; it < iterations; ++it) {
    const int j = it * blockDim.x + threadIdx.x;
    unsigned int key = 0;
    if (j < count) key = mono_key(row_in[begin + j]);
    const int byte3 = (int)(key >> 24);
    block_total[0] += __popc(__ballot_sync(0xFFFFFFFFu, (j < count) && byte3 > filter));
    block_total[1] += __popc(__ballot_sync(0xFFFFFFFFu, (j < count) && byte3 == filter));
  }
  if (lane == 0) {
    warp_cnt[warp][0] = block_total[0];
    warp_cnt[warp][1] = block_total[1];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int offset_hi = 0;
    int offset_lo = 0;
    for (int w = 0; w < kWarps; ++w) {
      warp_off[w][0] = offset_hi;
      warp_off[w][1] = offset_lo;
      offset_hi += warp_cnt[w][0];
      offset_lo += warp_cnt[w][1];
    }
    block_base[0] = atomicAdd(&rows[row].reserve_hi, offset_hi);
    block_base[1] = atomicAdd(&rows[row].reserve_lo, offset_lo);
  }
  __syncthreads();
  const int base = block_base[0] + warp_off[warp][0];
  const int base_lo = block_base[1] + warp_off[warp][1];

  // Pass 2: identical offsets, so head slots never collide. Every warp keeps a
  // running rank: slots are assigned per warp, not per iteration.
  int rank = 0;
  int rank_lo = 0;
  for (int it = 0; it < iterations; ++it) {
    const int j = it * blockDim.x + threadIdx.x;
    unsigned int key = 0;
    if (j < count) key = mono_key(row_in[begin + j]);
    const bool active = j < count;
    const int byte3 = (int)(key >> 24);
    const bool top = active && (byte3 > filter);
    const bool inside = active && (byte3 == filter);
    const unsigned int top_mask = __ballot_sync(0xFFFFFFFFu, top);
    const unsigned int inside_mask = __ballot_sync(0xFFFFFFFFu, inside);
    if (top) {
      const int slot = base + rank + __popc(top_mask & ((1u << lane) - 1u));
      if (slot < head) {
        out_values[(size_t)row * kK + slot] = key_to_float(key);
        out_idx[(size_t)row * kK + slot] = (long long)(begin + j);
      }
    }
    if (inside) {
      const int slot = base_lo + rank_lo + __popc(inside_mask & ((1u << lane) - 1u));
      row_keys[slot] = key;
      row_idx[slot] = begin + j;
    }
    rank += __popc(top_mask);
    rank_lo += __popc(inside_mask);
  }

  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) is_last = (atomicAdd(tickets + row, 1) == (int)gridDim.x - 1);
  __syncthreads();
  if (!is_last) return;
  __threadfence();

  // Last block: the bucket buffer is complete. Resolve its byte2 then byte1 so
  // the cutoff reaches 24 key bits, and emit the tail in one final sweep.
  if (threadIdx.x == 0) {
    tail[0] = rows[row].reserve_lo;  // reserved bucket slots == candidates
    rows[row].reserve_hi = 0;
    rows[row].reserve_lo = 0;
    tickets[row] = 0;  // ready for the next call
  }
  __syncthreads();
  const int candidates = tail[0];
  const int need = sel.need;

  for (int i = threadIdx.x; i < kBins; i += blockDim.x) hist[i] = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < candidates; i += blockDim.x)
    atomicAdd(&hist[(row_keys[i] >> 16) & kMask], 1);
  __syncthreads();
  if (threadIdx.x < 32) locate_bucket(hist, need, locate);
  __syncthreads();
  const int digit2 = locate[0];
  const int need2 = need - locate[1];

  for (int i = threadIdx.x; i < candidates; i += blockDim.x) {
    const unsigned int key = row_keys[i];
    if (((key >> 16) & kMask) == (unsigned int)digit2) atomicAdd(&hist[(key >> 8) & kMask], 1);
  }
  __syncthreads();
  if (threadIdx.x < 32) locate_bucket(hist, need2, locate);
  __syncthreads();
  const int digit3 = locate[0];
  const int need3 = need2 - locate[1];

  // Two tail regions: every byte2 above digit2 or byte1 above digit3 is strictly
  // better than the cutoff bucket, and need3 members of that bucket close it.
  if (threadIdx.x == 0) {
    tail[1] = 0;              // rank of the better-than-cutoff region
    tail[2] = 0;              // rank inside the cutoff bucket
    tail[3] = head + need - need3;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < candidates; i += blockDim.x) {
    const unsigned int key = row_keys[i];
    const int byte2 = (int)((key >> 16) & kMask);
    const bool better = (byte2 > digit2) || (byte2 == digit2 && (int)((key >> 8) & kMask) > digit3);
    const bool at_cutoff = (byte2 == digit2) && (int)((key >> 8) & kMask) == digit3;
    if (!better && !at_cutoff) continue;
    const int rank = better ? atomicAdd(&tail[1], 1) : atomicAdd(&tail[2], 1);
    if (!better && rank >= need3) continue;
    const int slot = better ? head + rank : tail[3] + rank;
    out_values[(size_t)row * kK + slot] = key_to_float(key);
    out_idx[(size_t)row * kK + slot] = (long long)row_idx[i];
  }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

void check_inputs(const torch::Tensor& values, const torch::Tensor& top_values,
                  const torch::Tensor& indices) {
  TORCH_CHECK_VALUE(values.is_cuda(), "values must be a CUDA tensor");
  TORCH_CHECK_TYPE(values.scalar_type() == torch::kFloat32, "values must be float32");
  TORCH_CHECK_VALUE(values.dim() == 2, "values must have shape [B, S]");
  TORCH_CHECK_VALUE(values.is_contiguous(), "values must be contiguous");

  TORCH_CHECK_VALUE(top_values.is_cuda(), "top_values must be a CUDA tensor");
  TORCH_CHECK_TYPE(top_values.scalar_type() == torch::kFloat32, "top_values must be float32");
  TORCH_CHECK_VALUE(top_values.is_contiguous(), "top_values must be contiguous");
  TORCH_CHECK_TYPE(indices.scalar_type() == torch::kInt64, "indices must be int64");
  TORCH_CHECK_VALUE(indices.is_contiguous(), "indices must be contiguous");

  const int64_t rows = values.size(0);
  const int64_t s = values.size(1);
  TORCH_CHECK_VALUE(s >= kK, "sequence length must be at least K");
  TORCH_CHECK_VALUE(top_values.dim() == 2 && top_values.size(0) == rows &&
                        top_values.size(1) == kK,
                    "top_values must have shape [B, K]");
  TORCH_CHECK_VALUE(indices.dim() == 2 && indices.size(0) == rows && indices.size(1) == kK,
                    "indices must have shape [B, K]");
  TORCH_CHECK_VALUE(top_values.device() == values.device() && indices.device() == values.device(),
                    "device mismatch");
  TORCH_CHECK_VALUE(s <= INT32_MAX, "sequence length too large");
  TORCH_CHECK_VALUE(rows <= INT32_MAX, "batch too large");
}

// Persistent workspace: the kernels leave it zeroed across calls, so the steady
// state costs no allocation and no memset.
torch::Tensor& scratch(int device, int64_t rows, int64_t s) {
  static torch::Tensor buffer;
  static int64_t cached_rows = -1;
  static int64_t cached_s = -1;
  static int cached_device = -1;
  if (!buffer.defined() || cached_rows != rows || cached_s != s || cached_device != device) {
    const int64_t words = rows * (kBins + 4 + 2) + rows * 2 * s;
    buffer = torch::zeros({words}, torch::TensorOptions().dtype(torch::kInt32).device(
                                       torch::Device(torch::kCUDA, device)));
    cached_rows = rows;
    cached_s = s;
    cached_device = device;
  }
  return buffer;
}

void kernel(const torch::Tensor& values, const torch::Tensor& top_values,
            const torch::Tensor& indices) {
  check_inputs(values, top_values, indices);
  const int64_t rows = values.size(0);
  const int64_t s = values.size(1);
  if (rows == 0) return;

  c10::cuda::CUDAGuard guard(values.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(values.get_device()).stream();
  int* words = scratch(values.get_device(), rows, s).data_ptr<int>();

  int* hist = words;
  RowSel* rows_sel = reinterpret_cast<RowSel*>(hist + (size_t)rows * kBins);
  int* tickets = reinterpret_cast<int*>(rows_sel + rows);
  unsigned int* buf_keys = reinterpret_cast<unsigned int*>(tickets + 2 * rows);
  int* buf_idx = reinterpret_cast<int*>(buf_keys + (size_t)rows * s);

  const int chunk = (int)((s + kBlocks - 1) / kBlocks);
  const dim3 grid(kBlocks, (unsigned int)rows);
  const float* in = values.data_ptr<float>();
  float* out_values = top_values.data_ptr<float>();
  long long* out_idx = reinterpret_cast<long long*>(indices.data_ptr<int64_t>());

  hist_pass<<<grid, kThreads, 0, stream>>>(in, (int)s, chunk, hist, rows_sel, tickets);
  gather_pass<<<grid, kThreads, 0, stream>>>(in, (int)s, chunk, rows_sel, buf_keys, buf_idx,
                                             out_values, out_idx, tickets + rows);

  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "topk launch failed: ", cudaGetErrorString(error));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("kernel", &kernel);
}

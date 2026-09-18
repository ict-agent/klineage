#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

// Top-K selection for the DSA indexer: B rows of S FP32 scores, K largest per
// row. Each row is read twice.
//
//   pass 1  all blocks bin every element by the top 12 bits of its
//           order-preserving key; the block that closes the histogram turns
//           the counts into the bin holding the K-th largest value.
//   pass 2  all blocks write the elements above that bin straight into the
//           output and park the elements inside it in a candidate buffer; the
//           closing block resolves those exactly with two finer binning
//           rounds.
//
// A smooth score distribution spreads ~S/2^12 elements over the cut bin, so
// the refining stage stays small while adversarial inputs stay exact because
// the candidate buffer is sized for the whole row.

constexpr int kThreads = 1024;
constexpr int kTopBits = 12;
constexpr int kTopBins = 1 << kTopBits;  // 4096
constexpr int kLowBits = 8;
constexpr int kLowBins = 1 << kLowBits;  // 256
constexpr int kGridX = 128;

// Workspace words per row: histogram followed by bookkeeping.
constexpr int kMetaWords = 8;
constexpr int kHistWords = kTopBins + kMetaWords;

constexpr int kMetaCand = 0;   // elements parked in the candidate buffer
constexpr int kMetaOut = 1;    // output slots filled from above the cut bin
constexpr int kMetaDone = 2;   // blocks that closed pass 1
constexpr int kMetaDone2 = 3;  // blocks that closed pass 2
constexpr int kMetaCut = 4;    // cut bin (top 12 key bits)
constexpr int kMetaAbove = 5;  // elements strictly above the cut bin
constexpr int kMetaNeed = 6;   // elements still wanted from the cut bin

// Order-preserving key: unsigned compare matches float compare.
__device__ __forceinline__ unsigned key_of(float value) {
  const unsigned bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Exact inverse of key_of.
__device__ __forceinline__ float float_of(unsigned key) {
  const unsigned bits = (key & 0x80000000u) ? (key & 0x7fffffffu) : ~key;
  return __uint_as_float(bits);
}

// Locates the bin holding the rank-th smallest key: the first bin whose
// inclusive prefix count reaches rank. Writes that bin and its exclusive
// prefix count. lanes * per_lane == bins, lanes a multiple of 32.
__device__ void cut_from_hist(const unsigned* __restrict__ hist, int lanes, int per_lane,
                              unsigned rank, unsigned* __restrict__ sums,
                              unsigned* __restrict__ warps, int* __restrict__ out_bin,
                              int* __restrict__ out_above) {
  const int t = threadIdx.x;

  // Per-lane group counts.
  unsigned mine = 0;
  if (t < lanes) {
    const unsigned* group = hist + t * per_lane;
    for (int j = 0; j < per_lane; ++j) mine += group[j];
  }
  sums[t] = mine;
  __syncthreads();

  // Warp-inclusive scan of the group counts, then a scan of the warp totals.
  unsigned incl = mine;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const unsigned other = __shfl_up_sync(0xffffffffu, incl, off);
    if ((t & 31) >= off) incl += other;
  }
  if ((t & 31) == 31) warps[t >> 5] = incl;
  __syncthreads();
  if (t < 32) {
    unsigned v = (t < (lanes >> 5)) ? warps[t] : 0u;
    const unsigned keep = v;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const unsigned other = __shfl_up_sync(0xffffffffu, v, off);
      if (t >= off) v += other;
    }
    warps[t] = v - keep;
  }
  __syncthreads();

  if (t >= lanes) return;
  const unsigned before = (incl - mine) + ((t >> 5) ? warps[t >> 5] : 0u);
  if (before >= rank || before + mine < rank) return;

  // The rank falls inside this lane's group; walk it to the exact bin.
  unsigned acc = before;
  for (int j = 0; j < per_lane; ++j) {
    const unsigned count = hist[t * per_lane + j];
    if (acc + count >= rank) {
      *out_bin = t * per_lane + j;
      *out_above = (int)acc;
      return;
    }
    acc += count;
  }
}

// Pass 1: bin every element by its top key bits and, in the closing block,
// turn the row histogram into the cut bin and its rank.
__global__ void __launch_bounds__(kThreads) hist_kernel(const float* __restrict__ in,
                                                        unsigned* __restrict__ hist, int S, int K,
                                                        int stride) {
  const int row = blockIdx.y;
  const float* __restrict__ src = in + (size_t)row * S;
  unsigned* __restrict__ bins = hist + (size_t)row * stride;
  int* __restrict__ meta = (int*)(bins + kTopBins);
  const int t = threadIdx.x;

  __shared__ unsigned counter[kTopBins];
  __shared__ unsigned sums[kThreads];
  __shared__ unsigned warps[kThreads / 32];
  __shared__ int closing;

  for (int b = t; b < kTopBins; b += kThreads) counter[b] = 0;
  __syncthreads();

  for (int i = blockIdx.x * kThreads + t; i < S; i += kGridX * kThreads)
    atomicAdd(&counter[key_of(src[i]) >> (32 - kTopBits)], 1u);
  __syncthreads();

  // Merge the block histogram into the row histogram; empty bins cost nothing.
  for (int b = t; b < kTopBins; b += kThreads) {
    const unsigned count = counter[b];
    if (count) atomicAdd(&bins[b], count);
  }
  __threadfence();
  __syncthreads();

  if (t == 0) closing = (atomicAdd(&meta[kMetaDone], 1) == (int)gridDim.x - 1);
  __syncthreads();
  if (!closing) return;

  // Rank of the K-th largest counted from the bottom of the row.
  const unsigned rank = (unsigned)(S - K + 1);
  int bin = 0;
  int above = 0;
  cut_from_hist(bins, kThreads, kTopBins / kThreads, rank, sums, warps, &bin, &above);
  __syncthreads();
  if (t == 0) {
    meta[kMetaCut] = bin;
    meta[kMetaAbove] = above;
    meta[kMetaNeed] = (int)rank - above;
  }
}

// Pass 2: emit everything above the cut bin, park the cut bin's elements, and
// resolve them in the closing block.
__global__ void __launch_bounds__(kThreads) gather_kernel(
    const float* __restrict__ in, unsigned* __restrict__ hist, unsigned* __restrict__ cand,
    float* __restrict__ out_values, int64_t* __restrict__ out_indices, int S, int K, int stride) {
  const int row = blockIdx.y;
  const float* __restrict__ src = in + (size_t)row * S;
  unsigned* __restrict__ bins = hist + (size_t)row * stride;
  int* __restrict__ meta = (int*)(bins + kTopBins);
  unsigned* __restrict__ cand_key = cand + (size_t)row * 2 * S;
  unsigned* __restrict__ cand_idx = cand_key + S;
  float* __restrict__ values = out_values + (size_t)row * K;
  int64_t* __restrict__ indices = out_indices + (size_t)row * K;
  const int t = threadIdx.x;

  const unsigned cut = (unsigned)meta[kMetaCut];
  const int above = meta[kMetaAbove];
  const int need = meta[kMetaNeed];

  __shared__ unsigned counter[kTopBins];
  __shared__ unsigned sums[kThreads];
  __shared__ unsigned warps[kThreads / 32];
  __shared__ unsigned sub_pop;
  __shared__ int closing;
  __shared__ int sub_bin;
  __shared__ int sub_above;
  __shared__ int low_bin;
  __shared__ int low_above;
  __shared__ int slot_high;
  __shared__ int slot_low;

  for (int i = blockIdx.x * kThreads + t; i < S; i += kGridX * kThreads) {
    const float value = src[i];
    const unsigned key = key_of(value);
    const unsigned bin = key >> (32 - kTopBits);
    if (bin > cut) {
      const int slot = atomicAdd(&meta[kMetaOut], 1);
      values[slot] = value;
      indices[slot] = (int64_t)i;
      continue;
    }
    if (bin == cut) {
      const int slot = atomicAdd(&meta[kMetaCand], 1);
      cand_key[slot] = key;
      cand_idx[slot] = (unsigned)i;
    }
  }
  __threadfence();
  __syncthreads();

  if (t == 0) closing = (atomicAdd(&meta[kMetaDone2], 1) == (int)gridDim.x - 1);
  __syncthreads();
  if (!closing) return;

  // Remaining keys all share the cut bin; split them on bits 19..8.
  const int n = __ldcg(meta + kMetaCand);
  for (int b = t; b < kTopBins; b += kThreads) counter[b] = 0;
  __syncthreads();
  for (int j = t; j < n; j += kThreads) atomicAdd(&counter[(cand_key[j] >> 8) & 0xFFFu], 1u);
  __syncthreads();
  cut_from_hist(counter, kThreads, kTopBins / kThreads, (unsigned)(n - need + 1), sums, warps,
                &sub_bin, &sub_above);
  __syncthreads();
  const int sub_need = need - sub_above;

  // ... then on bits 7..0, which pins the exact threshold key.
  for (int b = t; b < kLowBins; b += kThreads) counter[b] = 0;
  __syncthreads();
  for (int j = t; j < n; j += kThreads) {
    const unsigned key = cand_key[j];
    if (((key >> 8) & 0xFFFu) == (unsigned)sub_bin) atomicAdd(&counter[key & (kLowBins - 1)], 1u);
  }
  __syncthreads();
  if (t < kThreads / 32) {
    unsigned sum = 0;
#pragma unroll
    for (int j = 0; j < kLowBins / (kThreads / 32); ++j)
      sum += counter[t * (kLowBins / (kThreads / 32)) + j];
    warps[t] = sum;
  }
  __syncthreads();
  if (t == 0) {
    unsigned sum = 0;
#pragma unroll
    for (int j = 0; j < kThreads / 32; ++j) sum += warps[j];
    sub_pop = sum;
  }
  __syncthreads();
  cut_from_hist(counter, kLowBins / 4, 4, (unsigned)((int)sub_pop - sub_need + 1), sums, warps,
                &low_bin, &low_above);
  __syncthreads();

  // Emit: keys above the threshold fill the middle slots, keys equal to it
  // fill the tail up to K.
  const unsigned threshold =
      (cut << (32 - kTopBits)) | ((unsigned)sub_bin << kLowBits) | (unsigned)low_bin;
  const int want_low = sub_need - low_above;
  const int slot_mid = above;
  const int slot_tail = above + sub_above + low_above;
  if (t == 0) {
    slot_high = 0;
    slot_low = 0;
  }
  __syncthreads();
  for (int j = t; j < n; j += kThreads) {
    const unsigned key = cand_key[j];
    if (key > threshold) {
      const int slot = atomicAdd(&slot_high, 1);
      values[slot_mid + slot] = float_of(key);
      indices[slot_mid + slot] = (int64_t)cand_idx[j];
      continue;
    }
    if (key == threshold) {
      const int slot = atomicAdd(&slot_low, 1);
      if (slot < want_low) {
        values[slot_tail + slot] = float_of(key);
        indices[slot_tail + slot] = (int64_t)cand_idx[j];
      }
    }
  }
}

void check(const torch::Tensor& values, const torch::Tensor& top_values,
           const torch::Tensor& indices) {
  TORCH_CHECK_VALUE(values.is_cuda() && top_values.is_cuda() && indices.is_cuda(),
                    "Expected CUDA tensors");
  TORCH_CHECK_TYPE(values.scalar_type() == torch::kFloat32, "Expected float32 values");
  TORCH_CHECK_TYPE(top_values.scalar_type() == torch::kFloat32, "Expected float32 top_values");
  TORCH_CHECK_TYPE(indices.scalar_type() == torch::kInt64, "Expected int64 indices");
  TORCH_CHECK_VALUE(values.dim() == 2 && top_values.dim() == 2 && indices.dim() == 2,
                    "Expected rank 2 tensors");
  TORCH_CHECK_VALUE(values.is_contiguous() && top_values.is_contiguous() && indices.is_contiguous(),
                    "Expected contiguous tensors");
  TORCH_CHECK_VALUE(values.device() == top_values.device() && values.device() == indices.device(),
                    "Device mismatch");
}

void kernel(const torch::Tensor& values, const torch::Tensor& top_values,
            const torch::Tensor& indices) {
  check(values, top_values, indices);

  const int64_t rows = values.size(0);
  const int64_t length = values.size(1);
  const int64_t k = top_values.size(1);
  TORCH_CHECK_VALUE(top_values.size(0) == rows && indices.size(0) == rows && indices.size(1) == k,
                    "Shape mismatch");
  TORCH_CHECK_VALUE(k >= 1 && k <= length, "K must be in [1, S]");
  if (rows == 0 || length == 0) return;

  c10::cuda::CUDAGuard guard(values.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(values.get_device()).stream();
  const auto int32 = torch::TensorOptions().device(values.device()).dtype(torch::kInt32);

  // Row bookkeeping is zeroed per call; the candidate buffer needs no init.
  auto hist = torch::zeros({rows, kHistWords}, int32);
  auto cand = torch::empty({rows, 2 * length}, int32);

  const dim3 grid(kGridX, (unsigned)rows);
  hist_kernel<<<grid, kThreads, 0, stream>>>(
      values.data_ptr<float>(), reinterpret_cast<unsigned*>(hist.data_ptr<int>()), (int)length,
      (int)k, kHistWords);
  const auto hist_error = cudaGetLastError();
  TORCH_CHECK(hist_error == cudaSuccess, cudaGetErrorString(hist_error));

  gather_kernel<<<grid, kThreads, 0, stream>>>(
      values.data_ptr<float>(), reinterpret_cast<unsigned*>(hist.data_ptr<int>()),
      reinterpret_cast<unsigned*>(cand.data_ptr<int>()), top_values.data_ptr<float>(),
      indices.data_ptr<int64_t>(), (int)length, (int)k, kHistWords);
  const auto gather_error = cudaGetLastError();
  TORCH_CHECK(gather_error == cudaSuccess, cudaGetErrorString(gather_error));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("kernel", &kernel); }

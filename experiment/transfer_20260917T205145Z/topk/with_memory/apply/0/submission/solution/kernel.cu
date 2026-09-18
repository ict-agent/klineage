// Cluster radix Top-K: one 8-CTA cluster per row.
//
//   stage 0  stream row chunk into shared, histogram radix digit [31:21]
//   stage 1  cluster-reduce histogram, pick crossing digit -> splitter
//   stage 2  repeat on digit [20:10] over keys matching the prefix
//   stage 3  repeat on digit [9:0]
//   stage 4  emit keys better than splitter, then splitter ties
//
// Shared layout (dynamic): State | hist A | hist B | cached chunk.
// Alternating histograms let one cluster barrier per stage publish the
// finished histogram and order the next overwrite of the other buffer.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace {

constexpr int kWarp = 32;
constexpr int kThreads = 512;
constexpr int kWarps = kThreads / kWarp;
constexpr int kCluster = 8;
constexpr int kBuckets = 1 << 11;
constexpr int kBucketsPerThread = kBuckets / kThreads;
constexpr unsigned kWarpMask = 0xffffffffu;

constexpr int kShiftA = 21;
constexpr int kShiftB = 10;
constexpr int kShiftC = 0;
constexpr unsigned kMaskA = (1u << 11) - 1u;
constexpr unsigned kMaskB = (1u << 11) - 1u;
constexpr unsigned kMaskC = (1u << 10) - 1u;

constexpr int kSelected = 2048;

// Largest float maps to the largest unsigned encoding: order preserving.
__device__ __forceinline__ unsigned ordered(float value) {
  const unsigned bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

constexpr int kStateBytes = 128;
constexpr int kHistBytes = 2 * kBuckets * int(sizeof(unsigned));

struct alignas(16) State {
  unsigned cross_bucket;
  unsigned cross_prefix;
  unsigned cross_count;
  unsigned warp_total[kWarps];
  unsigned local_better;
  unsigned local_tied;
  unsigned ctr_better;
  unsigned ctr_tied;
};
static_assert(sizeof(State) <= kStateBytes, "state layout");

__device__ __forceinline__ unsigned warp_scan(unsigned value) {
  const int lane = threadIdx.x & (kWarp - 1);
  #pragma unroll
  for (int offset = 1; offset < kWarp; offset <<= 1) {
    const unsigned other = __shfl_up_sync(kWarpMask, value, offset);
    if (lane >= offset) value += other;
  }
  return value;
}

// Thread `t` owns buckets [t*kBucketsPerThread, ...). Sum the cluster
// histograms for those buckets, block-scan the counts, and keep the bucket
// holding the discard boundary.
template <int SHIFT>
__device__ __forceinline__ void scan_level(
    const cg::cluster_group& cluster, const unsigned* hist, State* state,
    unsigned& need, unsigned& splitter) {
  const int tid = threadIdx.x;
  const int lane = tid & (kWarp - 1);
  const int warp = tid / kWarp;

  unsigned part[kBucketsPerThread];
  #pragma unroll
  for (int i = 0; i < kBucketsPerThread; ++i) part[i] = 0;
  #pragma unroll
  for (int rank = 0; rank < kCluster; ++rank) {
    const unsigned* remote = cluster.map_shared_rank(hist, rank);
    #pragma unroll
    for (int i = 0; i < kBucketsPerThread; ++i)
      part[i] += remote[tid + i * kThreads];
  }

  unsigned total = 0;
  #pragma unroll
  for (int i = 0; i < kBucketsPerThread; ++i) total += part[i];
  const unsigned inclusive = warp_scan(total);
  if (lane == kWarp - 1) state->warp_total[warp] = inclusive;
  __syncthreads();

  unsigned exclusive = inclusive - total;
  unsigned grand = 0;
  #pragma unroll
  for (int w = 0; w < kWarps; ++w) {
    const unsigned value = state->warp_total[w];
    if (w < warp) exclusive += value;
    grand += value;
  }

  const unsigned discard = grand - need;
  unsigned cumulative = exclusive;
  #pragma unroll
  for (int i = 0; i < kBucketsPerThread; ++i) {
    if (cumulative <= discard && discard < cumulative + part[i]) {
      state->cross_bucket = tid * kBucketsPerThread + i;
      state->cross_prefix = cumulative;
      state->cross_count = part[i];
    }
    cumulative += part[i];
  }
  __syncthreads();

  const unsigned better = grand - state->cross_prefix - state->cross_count;
  need -= better;
  splitter |= state->cross_bucket << SHIFT;
}

__global__ void __cluster_dims__(kCluster, 1, 1)
topk_cluster(const float* __restrict__ values, float* __restrict__ out_values,
    int64_t* __restrict__ out_indices, int row_len, int chunk) {
  extern __shared__ char raw[];
  State* state = reinterpret_cast<State*>(raw);
  unsigned* hist_a = reinterpret_cast<unsigned*>(raw + kStateBytes);
  unsigned* hist_b = hist_a + kBuckets;
  float* cache = reinterpret_cast<float*>(raw + kStateBytes + kHistBytes);

  const cg::cluster_group cluster = cg::this_cluster();
  const int tid = threadIdx.x;
  const int rank = cluster.block_rank();
  const int row = blockIdx.x / kCluster;
  const int begin = rank * chunk;
  const int count = (begin < row_len) ? min(row_len - begin, chunk) : 0;
  const float* row_in = values + int64_t(row) * row_len;

  #pragma unroll
  for (int i = 0; i < kBucketsPerThread; ++i) {
    hist_a[tid + i * kThreads] = 0;
    hist_b[tid + i * kThreads] = 0;
  }
  if (tid == 0) {
    state->local_better = 0;
    state->local_tied = 0;
    state->ctr_better = 0;
    state->ctr_tied = 0;
  }
  __syncthreads();

  // Stage 0: fill the shared cache and histogram radix digit [31:21].
  const bool vectorized =
      ((reinterpret_cast<uintptr_t>(row_in + begin) & 15u) == 0u) && count >= 4;
  if (vectorized) {
    const float4* source = reinterpret_cast<const float4*>(row_in + begin);
    float4* target = reinterpret_cast<float4*>(cache);
    const int quads = count >> 2;
    for (int i = tid; i < quads; i += kThreads) {
      const float4 value = source[i];
      target[i] = value;
      atomicAdd(&hist_a[(ordered(value.x) >> kShiftA) & kMaskA], 1u);
      atomicAdd(&hist_a[(ordered(value.y) >> kShiftA) & kMaskA], 1u);
      atomicAdd(&hist_a[(ordered(value.z) >> kShiftA) & kMaskA], 1u);
      atomicAdd(&hist_a[(ordered(value.w) >> kShiftA) & kMaskA], 1u);
    }
    for (int i = (count & ~3) + tid; i < count; i += kThreads) {
      const float value = row_in[begin + i];
      cache[i] = value;
      atomicAdd(&hist_a[(ordered(value) >> kShiftA) & kMaskA], 1u);
    }
  } else {
    for (int i = tid; i < count; i += kThreads) {
      const float value = row_in[begin + i];
      cache[i] = value;
      atomicAdd(&hist_a[(ordered(value) >> kShiftA) & kMaskA], 1u);
    }
  }
  cluster.sync();

  unsigned need = kSelected;
  unsigned splitter = 0;
  scan_level<kShiftA>(cluster, hist_a, state, need, splitter);

  // Stage 2: radix digit [20:10] over keys still matching the prefix.
  #pragma unroll
  for (int i = 0; i < kBucketsPerThread; ++i) hist_b[tid + i * kThreads] = 0;
  __syncthreads();
  for (int i = tid; i < count; i += kThreads) {
    const unsigned bits = ordered(cache[i]);
    if ((bits >> (kShiftB + 11)) != (splitter >> (kShiftB + 11))) continue;
    atomicAdd(&hist_b[(bits >> kShiftB) & kMaskB], 1u);
  }
  cluster.sync();
  scan_level<kShiftB>(cluster, hist_b, state, need, splitter);

  // Stage 3: radix digit [9:0]; completes the 32 bit encoding.
  #pragma unroll
  for (int i = 0; i < kBucketsPerThread; ++i) hist_a[tid + i * kThreads] = 0;
  __syncthreads();
  for (int i = tid; i < count; i += kThreads) {
    const unsigned bits = ordered(cache[i]);
    if ((bits >> (kShiftC + 10)) != (splitter >> (kShiftC + 10))) continue;
    atomicAdd(&hist_a[bits & kMaskC], 1u);
  }
  cluster.sync();
  scan_level<kShiftC>(cluster, hist_a, state, need, splitter);

  // Stage 4: rank survivors per cluster rank, then compact them densely.
  unsigned better = 0;
  unsigned tied = 0;
  for (int i = tid; i < count; i += kThreads) {
    const unsigned bits = ordered(cache[i]);
    if (bits < splitter) ++better;
    else if (bits == splitter) ++tied;
  }
  if (better != 0) atomicAdd(&state->local_better, better);
  if (tied != 0) atomicAdd(&state->local_tied, tied);
  cluster.sync();

  const unsigned better_prefix = kSelected - need;
  unsigned better_base = 0;
  unsigned tied_base = better_prefix;
  #pragma unroll
  for (int other = 0; other < kCluster; ++other) {
    if (other >= rank) break;
    const State* remote = cluster.map_shared_rank(state, other);
    better_base += remote->local_better;
    tied_base += remote->local_tied;
  }
  __syncthreads();
  if (tid == 0) {
    state->ctr_better = 0;
    state->ctr_tied = 0;
  }
  __syncthreads();

  float* row_values = out_values + int64_t(row) * kSelected;
  int64_t* row_indices = out_indices + int64_t(row) * kSelected;
  for (int i = tid; i < count; i += kThreads) {
    const float value = cache[i];
    const unsigned bits = ordered(value);
    unsigned slot;
    if (bits < splitter) slot = better_base + atomicAdd(&state->ctr_better, 1u);
    else if (bits == splitter) slot = tied_base + atomicAdd(&state->ctr_tied, 1u);
    else continue;
    if (slot >= kSelected) continue;
    row_values[slot] = value;
    row_indices[slot] = begin + i;
  }
}

void check_input(const torch::Tensor& values, int64_t rows, int64_t row_len) {
  TORCH_CHECK_VALUE(values.is_cuda(), "values must be CUDA");
  TORCH_CHECK_TYPE(values.scalar_type() == torch::kFloat32, "values must be float32");
  TORCH_CHECK_VALUE(values.dim() == 2, "values must be rank 2");
  TORCH_CHECK_VALUE(values.is_contiguous(), "values must be contiguous");
  TORCH_CHECK_VALUE(values.size(0) == rows && values.size(1) == row_len, "shape mismatch");
}

void kernel(const torch::Tensor& values, const torch::Tensor& top_values,
    const torch::Tensor& indices) {
  TORCH_CHECK_VALUE(values.dim() == 2, "values must be rank 2");
  const int64_t rows = values.size(0);
  const int64_t row_len = values.size(1);
  check_input(values, rows, row_len);
  TORCH_CHECK_VALUE(top_values.is_cuda() && indices.is_cuda(), "outputs must be CUDA");
  TORCH_CHECK_TYPE(top_values.scalar_type() == torch::kFloat32, "top_values must be float32");
  TORCH_CHECK_TYPE(indices.scalar_type() == torch::kInt64, "indices must be int64");
  TORCH_CHECK_VALUE(top_values.is_contiguous() && indices.is_contiguous(), "outputs must be contiguous");
  TORCH_CHECK_VALUE(top_values.size(0) == rows && top_values.size(1) == kSelected, "top_values shape");
  TORCH_CHECK_VALUE(indices.size(0) == rows && indices.size(1) == kSelected, "indices shape");
  TORCH_CHECK_VALUE(top_values.device() == values.device() && indices.device() == values.device(), "device mismatch");
  TORCH_CHECK_VALUE(row_len >= kSelected, "row shorter than K");
  if (rows == 0) return;

  const int chunk = int((row_len + kCluster - 1) / kCluster);
  const size_t smem = size_t(kStateBytes + kHistBytes) + size_t(chunk) * sizeof(float);

  static size_t configured = 0;
  if (smem > configured) {
    cudaError_t status = cudaFuncSetAttribute(
        topk_cluster, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
    TORCH_CHECK(status == cudaSuccess, "shared attribute: ", cudaGetErrorString(status));
    configured = smem;
  }

  c10::cuda::CUDAGuard guard(values.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(values.get_device()).stream();
  topk_cluster<<<unsigned(rows * kCluster), kThreads, smem, stream>>>(
      values.data_ptr<float>(), top_values.data_ptr<float>(),
      indices.data_ptr<int64_t>(), int(row_len), chunk);
  const auto error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("kernel", &kernel);
}

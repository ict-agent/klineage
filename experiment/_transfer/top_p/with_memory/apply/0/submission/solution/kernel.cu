// Top-p renormalization over dense row probabilities.
//
// One CTA cluster of kChunks CTAs owns a row; each CTA caches its slice of the
// row in shared memory once and every later pass reads that cache instead of
// global memory. Each level of the radix select is a shared-memory histogram of
// the cached slice, merged across the cluster through distributed shared memory.
//
//   key      : order preserving uint32 of a probability
//   level 0  : bins over key bits [31:21]  (2048 bins)
//   level 1  : bins over key bits [20:11]  (1024 bins, inside the level 0 bin)
//   level 2  : bins over key bits [10:0]   (2048 bins, exactly one key per bin)
//
//   cache slice -> level 0 histogram -> cluster merge -> bin b0, mass above a0
//                      level 1, filtered to b0 -> bin b1, a1 = a0 + above
//                          level 2, filtered to b0:b1 -> exact key, retained
//                              emit cached slice scaled by 1 / retained
//
// Global traffic is one read (the cache fill) and one write (the emit); the
// reference re-reads the row for every level.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 1024;
constexpr int kLanes = 32;
constexpr int kWarps = kThreads / kLanes;
constexpr int kChunks = 8;  // CTAs per row, equals the cluster size

// Radix split of the 32-bit monotone key.
constexpr int kRadix0 = 11;
constexpr int kRadix1 = 10;
constexpr int kRadix2 = 11;
static_assert(kRadix0 + kRadix1 + kRadix2 == 32, "radix split must cover the key");
constexpr int kShift1 = kRadix2;
constexpr int kShift2 = 0;
constexpr int kBins0 = 1 << kRadix0;
constexpr int kBins1 = 1 << kRadix1;
constexpr int kBins2 = 1 << kRadix2;
constexpr int kPositions = 2 * kThreads;
static_assert(kBins0 <= kPositions, "scan must cover level 0 bins");

constexpr unsigned kSign = 0x80000000u;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kAllRows = 0;  // sweep filter: keep every key

// Bin sums are held as fixed point integers split into two 32-bit halves, so
// every update is a native shared integer atomic (64-bit shared atomics compile
// to a CAS spin loop). Probabilities are bounded by one, so the low half of a
// whole row (kLength * 2^(kLoBits-1)) stays below 2^32 and the high half inside
// 2^31; the split is therefore exact, and the only loss is the per-term rounding
// to 2^-kFracBits, whose row-wide bound (kLength * 2^-(kFracBits+1)) stays two
// orders of magnitude below the smallest observed selection margin.
constexpr int kFracBits = 46;
constexpr int kLoBits = 15;
constexpr unsigned kLoMask = (1u << kLoBits) - 1;
constexpr unsigned long long kFracUnit = 1ull << kFracBits;
constexpr unsigned kMantissaMask = 0x7FFFFFu;
constexpr unsigned kMantissaBit = 0x800000u;
constexpr int kExponentBias = 127;
constexpr int kMantissaBits = 23;
constexpr int kShiftBase = kExponentBias + kMantissaBits - kFracBits;
constexpr int kMaxShift = kFracBits - 24;  // one is the largest representable value
constexpr int64_t kMaxWidth = 1 << 17;

using Fixed = unsigned long long;

// Distributed shared memory through the native cluster intrinsics. They exist
// only from compute capability 9.0 on, while the build also emits cubins for
// older targets; those passes compile the fallbacks below and can never launch
// (the host rejects devices under 9.0), so their code is unreachable.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define TOP_P_CLUSTER 1
#else
#define TOP_P_CLUSTER 0
#endif

// This CTA's rank inside its cluster.
__device__ __forceinline__ unsigned cluster_rank() {
#if TOP_P_CLUSTER
  return __clusterRelativeBlockRank();
#else
  return 0u;
#endif
}

// Cluster-wide barrier; every thread of every peer CTA arrives.
__device__ __forceinline__ void cluster_sync() {
#if TOP_P_CLUSTER
  asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
#endif
}

// Address of the same variable inside peer CTA `rank`'s shared memory.
template <typename T>
__device__ __forceinline__ T* rank_shared(T* local, unsigned rank) {
#if TOP_P_CLUSTER
  return reinterpret_cast<T*>(__cluster_map_shared_rank(local, rank));
#else
  return local;
#endif
}

__device__ __forceinline__ unsigned encode(float value) {
  const unsigned bits = __float_as_uint(value);
  return (bits & kSign) ? ~bits : (bits | kSign);
}

__device__ __forceinline__ float decode(unsigned key) {
  return __uint_as_float((key & kSign) ? (key & ~kSign) : ~key);
}

// Fixed-point image of a positive probability, rounded to nearest. Values at or
// below 2^-kFracBits collapse to zero, which no selection decision can reach.
__device__ __forceinline__ Fixed fixed_of(unsigned bits) {
  const int shift = (int)((bits >> kMantissaBits) & 0xFFu) - kShiftBase;
  const unsigned mantissa = (bits & kMantissaMask) | kMantissaBit;
  if (shift >= 0) {
    if (shift > kMaxShift) return kFracUnit - 1;  // saturate at one
    return (Fixed)mantissa << shift;
  }
  const int down = -shift;
  if (down > 63) return 0ull;
  return ((Fixed)mantissa + (1ull << (down - 1))) >> down;
}

__device__ __forceinline__ Fixed warp_scan(Fixed value, int lane) {
#pragma unroll
  for (int offset = 1; offset < kLanes; offset <<= 1) {
    const Fixed peer = __shfl_up_sync(kFullMask, value, offset);
    if (lane >= offset) value += peer;
  }
  return value;
}

// Suffix sums are compared against the target either inclusively (positive
// target) or exclusively (target pinned at zero), so empty bins above the data
// can never be selected when the target is not positive.
enum class Bound { inclusive, exclusive };

struct Scratch {
  unsigned high[kPositions];
  unsigned low[kPositions];
  Fixed warp_offset[kWarps];
  Fixed total;
  Fixed above;
  Fixed group;
  int crossing;
};

// Rank 0 publishes one of these per level; every CTA reads rank 0's copy.
struct Plan {
  Fixed total;
  Fixed above;
  Fixed group;
  int bin;
};

__device__ __forceinline__ Fixed bin_sum(const Scratch& scratch, int bin) {
  return ((Fixed)scratch.high[bin] << kLoBits) + scratch.low[bin];
}

__device__ __forceinline__ void reset(Scratch& scratch, int bins) {
  for (int i = threadIdx.x; i < bins; i += kThreads) {
    scratch.high[i] = 0u;
    scratch.low[i] = 0u;
  }
  if (threadIdx.x == 0) scratch.crossing = -1;
  __syncthreads();
}

template <int FilterBits>
__device__ __forceinline__ bool selected(unsigned key, unsigned filter) {
  if constexpr (FilterBits == kAllRows) return true;
  else return (key >> (32 - FilterBits)) == filter;
}

__device__ __forceinline__ void accumulate(Scratch& scratch, int bin, Fixed value) {
  atomicAdd(&scratch.high[bin], (unsigned)(value >> kLoBits));
  atomicAdd(&scratch.low[bin], (unsigned)(value & kLoMask));
}

// Histogram of the CTA's cached slice; the slice is shared memory, so no level
// pays a global read.
template <int BinShift, int BinMask, int FilterBits>
__device__ __forceinline__ void sweep(Scratch& scratch, const float* slice, int count,
                                      unsigned filter) {
  const int tid = threadIdx.x;
  const float4* packed = reinterpret_cast<const float4*>(slice);
  const int vectors = count >> 2;
  for (int i = tid; i < vectors; i += kThreads) {
    const float4 v = packed[i];
    const unsigned kx = encode(v.x);
    const unsigned ky = encode(v.y);
    const unsigned kz = encode(v.z);
    const unsigned kw = encode(v.w);
    if (selected<FilterBits>(kx, filter))
      accumulate(scratch, (kx >> BinShift) & BinMask, fixed_of(__float_as_uint(v.x)));
    if (selected<FilterBits>(ky, filter))
      accumulate(scratch, (ky >> BinShift) & BinMask, fixed_of(__float_as_uint(v.y)));
    if (selected<FilterBits>(kz, filter))
      accumulate(scratch, (kz >> BinShift) & BinMask, fixed_of(__float_as_uint(v.z)));
    if (selected<FilterBits>(kw, filter))
      accumulate(scratch, (kw >> BinShift) & BinMask, fixed_of(__float_as_uint(v.w)));
  }
  for (int i = (vectors << 2) + tid; i < count; i += kThreads) {
    const float value = slice[i];
    const unsigned key = encode(value);
    if (selected<FilterBits>(key, filter))
      accumulate(scratch, (key >> BinShift) & BinMask, fixed_of(__float_as_uint(value)));
  }
}

// Largest bin whose suffix sum still reaches the remaining target. Rank 0 calls
// this on its merged histogram; the result is published to the whole cluster.
__device__ void locate(Scratch& scratch, int bins, Fixed target, Bound bound) {
  const int tid = threadIdx.x;
  const int lane = tid % kLanes;
  const int warp = tid / kLanes;
  __syncthreads();  // the sweep's histogram atomics must be visible

  // Thread tid owns the ascending bin pair (2*tid, 2*tid+1).
  const Fixed first = (2 * tid < bins) ? bin_sum(scratch, 2 * tid) : 0ull;
  const Fixed second = (2 * tid + 1 < bins) ? bin_sum(scratch, 2 * tid + 1) : 0ull;
  const Fixed pair = first + second;

  const Fixed inclusive = warp_scan(pair, lane);
  if (lane == kLanes - 1) scratch.warp_offset[warp] = inclusive;
  __syncthreads();
  if (warp == 0) {
    const Fixed carry = (lane < kWarps) ? scratch.warp_offset[lane] : 0ull;
    const Fixed scanned = warp_scan(carry, lane);
    scratch.warp_offset[lane] = scanned - carry;
  }
  __syncthreads();

  const Fixed exclusive = scratch.warp_offset[warp] + (inclusive - pair);
  if (tid == kThreads - 1) scratch.total = exclusive + pair;
  const Fixed first_limit = exclusive;
  const Fixed second_limit = exclusive + first;
  __syncthreads();

  const Fixed total = scratch.total;
  const bool reachable = total >= target;
  const Fixed budget = reachable ? total - target : 0ull;
  int candidate = -1;
  if (reachable && 2 * tid < bins &&
      (bound == Bound::inclusive ? first_limit <= budget : first_limit < budget))
    candidate = 2 * tid;
  if (reachable && 2 * tid + 1 < bins &&
      (bound == Bound::inclusive ? second_limit <= budget : second_limit < budget))
    candidate = 2 * tid + 1;
  if (candidate >= 0) atomicMax(&scratch.crossing, candidate);
  __syncthreads();

  if (scratch.crossing == 2 * tid || scratch.crossing == 2 * tid + 1) {
    const Fixed limit = (scratch.crossing == 2 * tid) ? first_limit : second_limit;
    scratch.group = bin_sum(scratch, scratch.crossing);
    scratch.above = total - (limit + scratch.group);
  }
  __syncthreads();
}

// Rank 0 folds the peer histograms into its own before locating.
__device__ void merge_peers(Scratch& scratch, int bins) {
  for (int bin = threadIdx.x; bin < bins; bin += kThreads) {
    unsigned high = scratch.high[bin];
    unsigned low = scratch.low[bin];
#pragma unroll
    for (int rank = 1; rank < kChunks; ++rank) {
      high += *rank_shared(&scratch.high[bin], rank);
      low += *rank_shared(&scratch.low[bin], rank);
    }
    scratch.high[bin] = high;
    scratch.low[bin] = low;
  }
}

__device__ __forceinline__ const Plan& read_plan(const Plan& local) {
  return *rank_shared(const_cast<Plan*>(&local), 0u);
}

__device__ __forceinline__ void load_slice(float* slice, const float* source, int count) {
  const int tid = threadIdx.x;
  const int vectors = count >> 2;
  for (int i = tid; i < vectors; i += kThreads)
    __pipeline_memcpy_async(slice + (i << 2), source + (i << 2), 16);
  for (int i = (vectors << 2) + tid; i < count; i += kThreads)
    __pipeline_memcpy_async(slice + i, source + i, 4);
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncthreads();
}

__device__ __forceinline__ void emit(const float* slice, float* out, int count, float cutoff,
                                     double inverse, bool keep_all) {
  const int tid = threadIdx.x;
  const int vectors = count >> 2;
  const float4* packed = reinterpret_cast<const float4*>(slice);
  float4* stores = reinterpret_cast<float4*>(out);
  for (int i = tid; i < vectors; i += kThreads) {
    const float4 v = packed[i];
    float4 r;
    r.x = (keep_all || v.x >= cutoff) ? (float)((double)v.x * inverse) : 0.0f;
    r.y = (keep_all || v.y >= cutoff) ? (float)((double)v.y * inverse) : 0.0f;
    r.z = (keep_all || v.z >= cutoff) ? (float)((double)v.z * inverse) : 0.0f;
    r.w = (keep_all || v.w >= cutoff) ? (float)((double)v.w * inverse) : 0.0f;
    stores[i] = r;
  }
  for (int i = (vectors << 2) + tid; i < count; i += kThreads) {
    const float value = slice[i];
    out[i] = (keep_all || value >= cutoff) ? (float)((double)value * inverse) : 0.0f;
  }
}

// One level: local histogram, cluster merge, decision. Returns rank 0's plan.
__device__ __forceinline__ Plan run_level(Scratch& scratch, Plan& plan, int bins, int filter_bits,
                                          unsigned filter, int rank, Fixed target, Bound bound,
                                          const float* slice, int count) {
  reset(scratch, bins);
  switch (filter_bits) {
    case kAllRows:
      sweep<21, kBins0 - 1, kAllRows>(scratch, slice, count, filter);
      break;
    case kRadix0:
      sweep<kShift1, kBins1 - 1, kRadix0>(scratch, slice, count, filter);
      break;
    default:
      sweep<kShift2, kBins2 - 1, kRadix0 + kRadix1>(scratch, slice, count, filter);
      break;
  }
  __syncthreads();  // the slice is cache, not histogram scratch
  cluster_sync();   // every peer histogram is complete

  if (rank == 0) {
    merge_peers(scratch, bins);
    locate(scratch, bins, target, bound);
  }
  const Plan value = {scratch.total, scratch.above, scratch.group, scratch.crossing};
  if (rank == 0) plan = value;
  cluster_sync();  // the decision is published; peers may reset their histograms
  return read_plan(plan);
}

__global__ void __launch_bounds__(kThreads) top_p_kernel(const float* __restrict__ probs,
                                                         const float* __restrict__ top_p,
                                                         float* __restrict__ out,
                                                         int width, int chunk_floats) {
  extern __shared__ __align__(16) float cache[];
  __shared__ Scratch scratch;
  __shared__ Plan plan;

  const int row_index = blockIdx.y;
  const int rank = (int)cluster_rank();
  const float* row = probs + (size_t)row_index * width;
  float* out_row = out + (size_t)row_index * width;

  const int begin = rank * chunk_floats;
  const int count = (begin < width) ? min(chunk_floats, width - begin) : 0;
  load_slice(cache, row + begin, count);

  // A non-positive threshold keeps the single best token, so pin the target at
  // zero and require a strict suffix comparison.
  const double raw = (double)top_p[row_index];
  const Fixed target = (raw > 0.0) ? (Fixed)(raw * (double)kFracUnit + 0.5) : 0ull;
  const Bound bound = (target > 0ull) ? Bound::inclusive : Bound::exclusive;

  Plan state = run_level(scratch, plan, kBins0, kAllRows, 0u, rank, target, bound, cache, count);
  if (state.total == 0ull) {  // degenerate empty row
    emit(cache, out_row + begin, count, 1.0f, 0.0, false);
    return;
  }
  if (state.bin < 0) {  // the row never reaches the target: keep it all
    emit(cache, out_row + begin, count, 0.0f, (double)kFracUnit / (double)state.total, true);
    return;
  }

  const unsigned bin0 = (unsigned)state.bin;
  Fixed above = state.above;

  const Fixed row_total = state.total;
  state = run_level(scratch, plan, kBins1, kRadix0, bin0, rank, target - above, bound, cache, count);
  above += state.above;
  const unsigned bin1 = (unsigned)state.bin;

  state = run_level(scratch, plan, kBins2, kRadix0 + kRadix1, (bin0 << kRadix1) | bin1, rank,
                    target - above, bound, cache, count);
  const unsigned key = (bin0 << (kRadix1 + kRadix2)) | (bin1 << kShift1) | (unsigned)state.bin;
  const Fixed retained = above + state.above + state.group;

  if (retained > 0ull) {
    emit(cache, out_row + begin, count, decode(key), (double)kFracUnit / (double)retained, false);
    return;
  }
  emit(cache, out_row + begin, count, 0.0f, (double)kFracUnit / (double)row_total, true);
}

void kernel(const torch::Tensor& probs, const torch::Tensor& top_p, torch::Tensor& renorm_probs) {
  TORCH_CHECK(probs.is_cuda() && top_p.is_cuda() && renorm_probs.is_cuda(), "expected CUDA tensors");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32 && top_p.scalar_type() == torch::kFloat32 &&
                  renorm_probs.scalar_type() == torch::kFloat32,
              "expected float32");
  TORCH_CHECK(probs.dim() == 2 && top_p.dim() == 1 && renorm_probs.dim() == 2, "unexpected rank");
  TORCH_CHECK(probs.is_contiguous() && top_p.is_contiguous() && renorm_probs.is_contiguous(),
              "expected contiguous tensors");
  const int64_t rows = probs.size(0);
  const int64_t width = probs.size(1);
  TORCH_CHECK(top_p.size(0) == rows, "top_p shape mismatch");
  TORCH_CHECK(renorm_probs.size(0) == rows && renorm_probs.size(1) == width, "output shape mismatch");
  TORCH_CHECK(probs.device() == top_p.device() && probs.device() == renorm_probs.device(),
              "device mismatch");
  TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major >= 9,
              "top_p needs compute capability 9.0 for thread block clusters");
  if (rows == 0 || width == 0) return;
  TORCH_CHECK(width <= kMaxWidth, "row width exceeds the fixed-point split range");

  // Every row is padded to a whole number of 16-byte slices so the cache fill
  // and the emit can move float4 vectors.
  const int64_t chunk = ((width + kChunks - 1) / kChunks + 3) & ~int64_t(3);
  const size_t dynamic_bytes = (size_t)chunk * sizeof(float);

  c10::cuda::CUDAGuard guard(probs.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(probs.get_device()).stream();
  const cudaError_t attribute_error =
      cudaFuncSetAttribute(top_p_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           (int)dynamic_bytes);
  TORCH_CHECK(attribute_error == cudaSuccess, "shared memory attribute failed: ",
              cudaGetErrorString(attribute_error));

  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(kChunks, (unsigned)rows);
  config.blockDim = dim3(kThreads);
  config.dynamicSmemBytes = dynamic_bytes;
  config.stream = stream;
  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = kChunks;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;
  config.attrs = attribute;
  config.numAttrs = 1;
  const cudaError_t error =
      cudaLaunchKernelEx(&config, top_p_kernel, probs.data_ptr<float>(), top_p.data_ptr<float>(),
                         renorm_probs.data_ptr<float>(), (int)width, (int)chunk);
  TORCH_CHECK(error == cudaSuccess, "top_p launch failed: ", cudaGetErrorString(error));
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("kernel", &kernel); }

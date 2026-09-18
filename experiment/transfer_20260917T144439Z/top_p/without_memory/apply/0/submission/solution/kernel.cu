#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

// top-p renormalization of one probability row.
//
// Reference semantics: sort descending, cumulate in float64, take the value
// where the cumulative mass first reaches top_p, keep every token >= that
// value, renormalize by their summed mass.
//
// The pipeline never sorts the row.  It locates the cutoff value window with a
// value-key histogram, compacts that window, and only sorts the window.
//
//   probs ─► hist ─► band ─► gather ─► resolve ─► scale ─► renorm_probs
//
//   hist    per-row histogram over the 12-bit value key
//   band    crossing key cb: [key(cb), key(cb+1)) holds the cutoff value
//   gather  compaction of that window + exact float64 mass above/in it
//   resolve sort the window, walk it, emit cutoff and 1/mass
//   scale   masked rescale of the whole row

// A probability maps to a 12-bit key: sign, exponent, 3 mantissa bits.  Keys
// are monotone in value, so a window is a contiguous key range spanning 12.5%
// of its lower edge.
constexpr int kKeyShift = 20;
constexpr int kKeyBins = 1 << 12;
constexpr unsigned kMaxKey = 0x7F800000u;  // +inf: no token is above it

constexpr int kHistBlocks = 4;      // blocks per row, histogram pass
constexpr int kHistThreads = 1024;
constexpr int kBandThreads = 1024;
constexpr int kGatherBlocks = 8;    // blocks per row, compaction pass
constexpr int kGatherThreads = 1024;
constexpr int kScaleBlocks = 8;
constexpr int kScaleThreads = 256;
constexpr int kResolveThreads = 1024;

// Values gathered per row; the crossing window holds a few thousand tokens.
constexpr int kBandCap = 1 << 13;

constexpr unsigned kFull = 0xffffffffu;

__device__ __forceinline__ unsigned int key_of(float x) {
  return __float_as_uint(x) >> kKeyShift;
}

__device__ __forceinline__ float key_lo(unsigned int key) {
  return __uint_as_float(key << kKeyShift);
}

// Sum of v across a warp, broadcast to every lane.
__device__ __forceinline__ double warp_sum(double v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(kFull, v, o);
  return __shfl_sync(kFull, v, 0);
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(kFull, v, o));
  return __shfl_sync(kFull, v, 0);
}

// Sum of v across the block; scratch holds one double per warp.
__device__ double block_sum(double v, double* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nwarp = blockDim.x >> 5;

  v = warp_sum(v);

  if (lane == 0) scratch[warp] = v;
  __syncthreads();

  if (warp == 0) {
    double w = (lane < nwarp) ? scratch[lane] : 0.0;
    w = warp_sum(w);
    if (lane == 0) scratch[0] = w;
  }
  __syncthreads();

  const double total = scratch[0];
  __syncthreads();
  return total;
}

__device__ float block_max(float v, double* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nwarp = blockDim.x >> 5;

  v = warp_max(v);

  if (lane == 0) scratch[warp] = (double)v;
  __syncthreads();

  if (warp == 0) {
    float w = (lane < nwarp) ? (float)scratch[lane] : 0.0f;
    w = warp_max(w);
    if (lane == 0) scratch[0] = (double)w;
  }
  __syncthreads();

  const float total = (float)scratch[0];
  __syncthreads();
  return total;
}

// Block-wide exclusive scan of one double per thread (inclusive minus v).
__device__ double block_scan(double v, double* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int nwarp = blockDim.x >> 5;

  double incl = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const double t = __shfl_up_sync(kFull, incl, o);
    if (lane >= o) incl += t;
  }

  if (lane == 31) scratch[warp] = incl;
  __syncthreads();

  if (warp == 0) {
    double w = (lane < nwarp) ? scratch[lane] : 0.0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const double t = __shfl_up_sync(kFull, w, o);
      if (lane >= o) w += t;
    }
    if (lane < nwarp) {
      const double own = scratch[lane];
      scratch[lane] = w - own;
    }
  }
  __syncthreads();

  return scratch[warp] + (incl - v);
}

// ---------------------------------------------------------------------------
// Per-block histograms of the row's value keys.
// hist[block][key] holds the float32 mass landing in each key.
// ---------------------------------------------------------------------------
__global__ void hist_kernel(const float* __restrict__ probs, int vocab,
                            float* __restrict__ hist) {
  __shared__ float bins[kKeyBins];

  const int tid = threadIdx.x;
  for (int i = tid; i < kKeyBins; i += blockDim.x) bins[i] = 0.0f;
  __syncthreads();

  const int row = blockIdx.y;
  const int chunk = (vocab + kHistBlocks - 1) / kHistBlocks;
  const int begin = blockIdx.x * chunk;
  const int end = min(begin + chunk, vocab);
  if (begin >= end) return;
  const float* p = probs + (size_t)row * vocab + begin;

  const int nvec = (end - begin) >> 2;
  for (int v = tid; v < nvec; v += blockDim.x) {
    const float4 q = *reinterpret_cast<const float4*>(p + (v << 2));
    atomicAdd(&bins[key_of(q.x)], q.x);
    atomicAdd(&bins[key_of(q.y)], q.y);
    atomicAdd(&bins[key_of(q.z)], q.z);
    atomicAdd(&bins[key_of(q.w)], q.w);
  }

  for (int i = (nvec << 2) + tid; i < end - begin; i += blockDim.x) {
    const float x = p[i];
    atomicAdd(&bins[key_of(x)], x);
  }
  __syncthreads();

  float* dst = hist + ((size_t)row * kHistBlocks + blockIdx.x) * kKeyBins;
  for (int i = tid; i < kKeyBins; i += blockDim.x) dst[i] = bins[i];
}

// ---------------------------------------------------------------------------
// Merge the block histograms and pick the crossing key cb: the largest key
// whose suffix mass still reaches top_p.  The cutoff value lies in
// [key(cb), key(cb+1)).  Counters for the next pass are cleared here.
// ---------------------------------------------------------------------------
__global__ void band_kernel(const float* __restrict__ hist,
                            const float* __restrict__ top_p,
                            float* __restrict__ band_lo, float* __restrict__ band_hi,
                            unsigned int* __restrict__ cnt,
                            double* __restrict__ sum_high, double* __restrict__ sum_band) {
  constexpr int kPer = kKeyBins / kBandThreads;  // keys per thread

  __shared__ float bins[kKeyBins];
  __shared__ double seg[kBandThreads];
  __shared__ double warp_tot[32];
  __shared__ double warp_suf[32];
  __shared__ int best;

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  if (tid == 0) {
    cnt[row] = 0u;
    sum_high[row] = 0.0;
    sum_band[row] = 0.0;
    best = -1;
  }

  float s[kPer];
#pragma unroll
  for (int j = 0; j < kPer; ++j) s[j] = 0.0f;
  for (int b = 0; b < kHistBlocks; ++b) {
    const float* h = hist + ((size_t)row * kHistBlocks + b) * kKeyBins + tid * kPer;
#pragma unroll
    for (int j = 0; j < kPer; ++j) s[j] += h[j];
  }
#pragma unroll
  for (int j = 0; j < kPer; ++j) bins[tid * kPer + j] = s[j];
  __syncthreads();

  // Suffix mass over the whole key space, key by key.
  double g = 0.0;
#pragma unroll
  for (int j = 0; j < kPer; ++j) g += (double)s[j];
  double incl = g;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const double t = __shfl_up_sync(kFull, incl, o);
    if (lane >= o) incl += t;
  }
  if (lane == 31) warp_tot[warp] = incl;
  __syncthreads();
  if (tid == 0) {
    double acc = 0.0;
    for (int w = 31; w >= 0; --w) {
      const double t = warp_tot[w];
      warp_suf[w] = acc;
      acc += t;
    }
  }
  __syncthreads();
  seg[tid] = warp_suf[warp] + (incl - g);  // mass of every key above this thread
  __syncthreads();

  const double tp = (double)top_p[row];
  int hit = -1;
  double cum = seg[tid];
#pragma unroll
  for (int j = kPer - 1; j >= 0; --j) {
    cum += (double)s[j];
    if (cum >= tp) {
      hit = tid * kPer + j;
      break;
    }
  }
  if (hit >= 0) atomicMax(&best, hit);
  __syncthreads();

  if (tid == 0) {
    const unsigned int cb = (unsigned int)(best > 0 ? best : 0);
    band_lo[row] = key_lo(cb);
    band_hi[row] = key_lo(cb + 1u);
  }
}

// ---------------------------------------------------------------------------
// Compact the window [band_lo, band_hi) of one row and measure, in float64,
// the mass strictly above it and the mass inside it.  Both are needed to
// certify that the cutoff really sits inside the window.
// ---------------------------------------------------------------------------
__global__ void gather_kernel(const float* __restrict__ probs, int vocab,
                              const float* __restrict__ band_lo,
                              const float* __restrict__ band_hi,
                              float* __restrict__ buf, unsigned int* __restrict__ cnt,
                              double* __restrict__ sum_high,
                              double* __restrict__ sum_band) {
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;

  const float lo = band_lo[row];
  const float hi = band_hi[row];
  const int chunk = (vocab + kGatherBlocks - 1) / kGatherBlocks;
  const int begin = blockIdx.x * chunk;
  const int end = min(begin + chunk, vocab);
  const float* p = probs + (size_t)row * vocab + begin;
  float* dst = buf + (size_t)row * kBandCap;

  double high = 0.0;
  double band = 0.0;
  const unsigned int below = (1u << lane) - 1u;

  const int nvec = (end - begin) >> 2;
  for (int v = tid; v < nvec; v += blockDim.x) {
    const float4 q = *reinterpret_cast<const float4*>(p + (v << 2));
    const float x[4] = {q.x, q.y, q.z, q.w};
    bool in[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      in[j] = false;
      if (x[j] >= hi) {
        high += (double)x[j];
        continue;
      }
      if (x[j] < lo) continue;
      band += (double)x[j];
      in[j] = true;
    }

    // One ticket per warp iteration; ranks come from the lane ballots.
    const unsigned int m0 = __ballot_sync(kFull, in[0]);
    const unsigned int m1 = __ballot_sync(kFull, in[1]);
    const unsigned int m2 = __ballot_sync(kFull, in[2]);
    const unsigned int m3 = __ballot_sync(kFull, in[3]);
    const unsigned int n = __popc(m0) + __popc(m1) + __popc(m2) + __popc(m3);
    if (n == 0) continue;

    int base = 0;
    if (lane == 0) base = (int)atomicAdd(&cnt[row], n);
    base = __shfl_sync(kFull, base, 0);

    int rank = 0;
    if (in[0] && base + rank < kBandCap) dst[base + rank] = x[0];
    rank += __popc(m0);
    if (in[1] && base + rank + __popc(m1 & below) < kBandCap) dst[base + rank + __popc(m1 & below)] = x[1];
    rank += __popc(m1);
    if (in[2] && base + rank + __popc(m2 & below) < kBandCap) dst[base + rank + __popc(m2 & below)] = x[2];
    rank += __popc(m2);
    if (in[3] && base + rank + __popc(m3 & below) < kBandCap) dst[base + rank + __popc(m3 & below)] = x[3];
  }

  // Tail elements past the vector body.
  for (int i = (nvec << 2) + tid; i < end - begin; i += blockDim.x) {
    const float x = p[i];
    const bool inband = (x >= lo) && (x < hi);
    if (x >= hi) high += (double)x;
    else if (inband) band += (double)x;

    const unsigned int m = __ballot_sync(kFull, inband);
    const unsigned int n = __popc(m);
    if (n == 0) continue;
    int base = 0;
    if (lane == 0) base = (int)atomicAdd(&cnt[row], n);
    base = __shfl_sync(kFull, base, 0);
    if (inband) {
      const int slot = base + __popc(m & below);
      if (slot < kBandCap) dst[slot] = x;
    }
  }

  high = warp_sum(high);
  band = warp_sum(band);
  if (lane == 0 && (high != 0.0 || band != 0.0)) {
    atomicAdd(&sum_high[row], high);
    atomicAdd(&sum_band[row], band);
  }
}

// ---------------------------------------------------------------------------
// Sort the gathered window descending, walk it against top_p and emit the
// cutoff value plus the inverse mass of the retained set.  Falls back to an
// exact binary search over the key space when the certification fails.
// ---------------------------------------------------------------------------
__device__ void resolve_exact(const float* __restrict__ p, int vocab, double tp,
                              double* scratch, float* cutoff, double* mass) {
  const int tid = threadIdx.x;

  double total = 0.0;
  for (int i = tid; i < vocab; i += blockDim.x) total += (double)p[i];
  total = block_sum(total, scratch);

  // top_p above the whole row mass keeps every token; the cutoff is the minimum.
  if (total < tp) {
    float mn = INFINITY;
    for (int i = tid; i < vocab; i += blockDim.x) mn = fminf(mn, p[i]);
    mn = block_max(mn, scratch);
    if (tid == 0) {
      *cutoff = mn;
      *mass = total;
    }
    return;
  }

  unsigned int lo = 0u;
  unsigned int hi = kMaxKey;
  while (hi - lo > 1) {
    const unsigned int mid = lo + (hi - lo) / 2;
    const float t = key_lo(mid);
    double s = 0.0;
    for (int i = tid; i < vocab; i += blockDim.x) {
      const float x = p[i];
      if (x >= t) s += (double)x;
    }
    s = block_sum(s, scratch);
    if (s >= tp) lo = mid;
    else hi = mid;
  }

  // The cutoff is the largest token at or below the surviving key.
  const float bound = key_lo(lo);
  float best = 0.0f;
  for (int i = tid; i < vocab; i += blockDim.x) {
    const float x = p[i];
    if (x <= bound) best = fmaxf(best, x);
  }
  best = block_max(best, scratch);

  double s = 0.0;
  for (int i = tid; i < vocab; i += blockDim.x) {
    const float x = p[i];
    if (x >= best) s += (double)x;
  }
  s = block_sum(s, scratch);
  if (tid == 0) {
    *cutoff = best;
    *mass = s;
  }
}

__global__ void resolve_kernel(const float* __restrict__ probs, int vocab,
                               const float* __restrict__ buf,
                               const unsigned int* __restrict__ cnt,
                               const double* __restrict__ sum_high,
                               const double* __restrict__ sum_band,
                               const float* __restrict__ top_p,
                               float* __restrict__ cutoff, float* __restrict__ inv_s) {
  __shared__ float s[kBandCap];
  __shared__ double seg[kResolveThreads];
  __shared__ double scratch[32];
  __shared__ int hit;

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const float* p = probs + (size_t)row * vocab;
  const double tp = (double)top_p[row];
  const double above = sum_high[row];
  const double band_mass = sum_band[row];
  const unsigned int n = cnt[row];

  if (tid == 0) hit = -1;
  __syncthreads();

  // The window is only trusted when the cutoff provably sits inside it.
  bool slow = n == 0u || n > (unsigned int)kBandCap || above >= tp || above + band_mass < tp;

  if (!slow) {
    int size = 1;
    while (size < (int)n) size <<= 1;
    for (int i = tid; i < size; i += blockDim.x) {
      s[i] = (i < (int)n) ? buf[(size_t)row * kBandCap + i] : 0.0f;
    }
    __syncthreads();

    // Descending bitonic sort; each exchange pair is applied once.
    for (int k = 2; k <= size; k <<= 1) {
      for (int j = k >> 1; j > 0; j >>= 1) {
        for (int i = tid; i < size; i += blockDim.x) {
          const int ixj = i ^ j;
          if (ixj <= i) continue;
          const bool up = (i & k) == 0;
          const float a = s[i];
          const float b = s[ixj];
          if (up ? (a < b) : (a > b)) {
            s[i] = b;
            s[ixj] = a;
          }
        }
        __syncthreads();
      }
    }

    // Mass per thread run of sorted values; the exclusive scan yields the mass
    // carried by all earlier runs.
    const int per = max(1, size / (int)blockDim.x);
    const int start = tid * per;
    const int stop = min(start + per, size);
    double mine = 0.0;
    for (int i = start; i < stop; ++i) mine += (double)s[i];
    const double prefix = block_scan(mine, scratch);
    seg[tid] = prefix;
    __syncthreads();

    if (above + prefix + mine >= tp) {
      double cum = above + prefix;
      for (int i = start; i < stop; ++i) {
        cum += (double)s[i];
        if (cum >= tp) {
          atomicMin(&hit, i);
          break;
        }
      }
    }
    __syncthreads();

    if (hit < 0) slow = true;
  }

  if (!slow) {
    if (tid == 0) {
      const int i = hit;
      const float cut = s[i];

      // Tokens tied with the cutoff are all retained.
      int lo2 = i;
      int hi2 = ((int)n) - 1;
      while (lo2 < hi2) {
        const int mid = (lo2 + hi2 + 1) >> 1;
        if (s[mid] == cut) lo2 = mid;
        else hi2 = mid - 1;
      }

      int size = 1;
      while (size < (int)n) size <<= 1;
      const int per = max(1, size / (int)blockDim.x);
      const int run = lo2 / per;
      double mass = above + seg[run];
      for (int k = run * per; k <= lo2; ++k) mass += (double)s[k];

      cutoff[row] = cut;
      inv_s[row] = (float)(1.0 / mass);
    }
    return;
  }

  // Exact fallback: binary search the largest key whose suffix mass reaches
  // top_p, then report the largest token at or below it.
  float cut = 0.0f;
  double mass = 0.0;
  resolve_exact(p, vocab, tp, scratch, &cut, &mass);
  if (tid == 0) {
    cutoff[row] = cut;
    inv_s[row] = (float)(1.0 / mass);
  }
}

// ---------------------------------------------------------------------------
// Renormalize: keep the retained tokens, drop the rest.
// ---------------------------------------------------------------------------
__global__ void scale_kernel(const float* __restrict__ probs, float* __restrict__ out,
                             int vocab, const float* __restrict__ cutoff,
                             const float* __restrict__ inv_s) {
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  const float cut = cutoff[row];
  const float inv = inv_s[row];
  const int chunk = (vocab + kScaleBlocks - 1) / kScaleBlocks;
  const int begin = blockIdx.x * chunk;
  const int end = min(begin + chunk, vocab);
  if (begin >= end) return;

  const float* src = probs + (size_t)row * vocab + begin;
  float* dst = out + (size_t)row * vocab + begin;

  const int nvec = (end - begin) >> 2;
  for (int v = tid; v < nvec; v += blockDim.x) {
    const float4 q = *reinterpret_cast<const float4*>(src + (v << 2));
    float4 r;
    r.x = q.x >= cut ? q.x * inv : 0.0f;
    r.y = q.y >= cut ? q.y * inv : 0.0f;
    r.z = q.z >= cut ? q.z * inv : 0.0f;
    r.w = q.w >= cut ? q.w * inv : 0.0f;
    *reinterpret_cast<float4*>(dst + (v << 2)) = r;
  }
  for (int i = (nvec << 2) + tid; i < end - begin; i += blockDim.x) {
    const float x = src[i];
    dst[i] = x >= cut ? x * inv : 0.0f;
  }
}

void check_tensor(const torch::Tensor& t, int64_t rows, int64_t vocab, const char* name) {
  TORCH_CHECK_VALUE(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK_TYPE(t.scalar_type() == torch::kFloat32, name, " must be float32");
  TORCH_CHECK_VALUE(t.dim() == 2, name, " must be rank 2");
  TORCH_CHECK_VALUE(t.size(0) == rows && t.size(1) == vocab, name, " shape mismatch");
  TORCH_CHECK_VALUE(t.stride(1) == 1 && t.stride(0) == vocab, name, " must be row major");
}

void kernel(const torch::Tensor& probs, const torch::Tensor& top_p,
            torch::Tensor& renorm_probs) {
  TORCH_CHECK_VALUE(top_p.is_cuda(), "top_p must be a CUDA tensor");
  TORCH_CHECK_TYPE(top_p.scalar_type() == torch::kFloat32, "top_p must be float32");
  TORCH_CHECK_VALUE(top_p.dim() == 1, "top_p must be rank 1");

  const int64_t rows = probs.size(0);
  const int64_t vocab = probs.size(1);
  check_tensor(probs, rows, vocab, "probs");
  check_tensor(renorm_probs, rows, vocab, "renorm_probs");
  TORCH_CHECK_VALUE(top_p.size(0) == rows, "top_p length mismatch");
  TORCH_CHECK_VALUE(top_p.stride(0) == 1, "top_p must be contiguous");
  TORCH_CHECK_VALUE(probs.device() == renorm_probs.device(), "device mismatch");
  TORCH_CHECK_VALUE(probs.device() == top_p.device(), "device mismatch");
  if (rows == 0 || vocab == 0) return;

  // Device scratch: one flat allocation, sliced by hand.
  const int64_t hist_bins = rows * kHistBlocks * kKeyBins;
  const int64_t buf_vals = rows * kBandCap;
  const int64_t hist_bytes = hist_bins * (int64_t)sizeof(float);
  const int64_t buf_bytes = buf_vals * (int64_t)sizeof(float);
  const int64_t tail_bytes = rows * (int64_t)64;
  const int64_t total_bytes = hist_bytes + buf_bytes + tail_bytes + 64;

  auto workspace = torch::empty({total_bytes}, probs.options().dtype(torch::kUInt8));
  char* base = reinterpret_cast<char*>(workspace.data_ptr<uint8_t>());

  float* hist = reinterpret_cast<float*>(base);
  float* buf = reinterpret_cast<float*>(base + hist_bytes);
  char* tail = base + hist_bytes + buf_bytes;
  float* band_lo = reinterpret_cast<float*>(tail);
  float* band_hi = band_lo + rows;
  float* cutoff = band_hi + rows;
  float* inv_s = cutoff + rows;
  unsigned int* cnt = reinterpret_cast<unsigned int*>(inv_s + rows);
  char* aligned = reinterpret_cast<char*>(cnt + rows);
  aligned += (8 - (reinterpret_cast<uintptr_t>(aligned) & 7)) & 7;
  double* sum_high = reinterpret_cast<double*>(aligned);
  double* sum_band = sum_high + rows;

  c10::cuda::CUDAGuard guard(probs.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(probs.get_device()).stream();

  const float* p = probs.data_ptr<float>();
  const float* tp = top_p.data_ptr<float>();
  float* out = renorm_probs.data_ptr<float>();

  hist_kernel<<<dim3(kHistBlocks, (unsigned)rows), kHistThreads, 0, stream>>>(p, (int)vocab, hist);
  band_kernel<<<(unsigned)rows, kBandThreads, 0, stream>>>(
      hist, tp, band_lo, band_hi, cnt, sum_high, sum_band);
  gather_kernel<<<dim3(kGatherBlocks, (unsigned)rows), kGatherThreads, 0, stream>>>(
      p, (int)vocab, band_lo, band_hi, buf, cnt, sum_high, sum_band);
  resolve_kernel<<<(unsigned)rows, kResolveThreads, 0, stream>>>(
      p, (int)vocab, buf, cnt, sum_high, sum_band, tp, cutoff, inv_s);
  scale_kernel<<<dim3(kScaleBlocks, (unsigned)rows), kScaleThreads, 0, stream>>>(
      p, out, (int)vocab, cutoff, inv_s);

  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "top_p launch failed: ", cudaGetErrorString(error));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("kernel", &kernel);
}

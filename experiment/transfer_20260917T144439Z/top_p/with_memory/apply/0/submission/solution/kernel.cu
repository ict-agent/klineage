// Top-p (nucleus) renormalization of dense row probabilities.
//
// Reference semantics (float64 accumulate, float32 output):
//   ordered   = sort(probs, dim=-1, descending=True)          (stable)
//   cumulative= cumsum(ordered, float64)
//   boundary  = first index k with cumulative[k] >= top_p     (clamped to V-1)
//   cutoff    = ordered[boundary]
//   out       = where(probs >= cutoff, probs, 0) / sum(retained)
//
// Because cutoff is always an element of the row, the retention rule can be
// restated without any sort:
//   cutoff = largest element value v with S(v) >= top_p,
//   where  S(v) = sum of every row element with value >= v   (float64).
// The support {x >= cutoff} then matches the reference support exactly, which
// the checker enforces bit-for-bit (torch.equal(actual > 0, expected > 0)).
//
// Pipeline (one launch per stage, no library dependencies):
//   K1  read row once, build a per-row histogram of the top key bits carrying
//       exact float64 bucket sums; the last CTA per row scans buckets from the
//       top to locate the bucket holding the cutoff and the exact mass above it.
//   K2  read row again, gather the keys of that bucket only (~1% of a row).
//   K3  refine the gathered keys with two further histogram levels (11 + 10
//       bits) which together with the 13 level-1 bits resolve the full 32-bit
//       ordering key, i.e. the exact cutoff value, and the exact retained mass.
//   K4  read row, apply mask and scale, write output.
//
// Monotone ordering key: float bits remapped so that unsigned comparison of
// keys equals numeric comparison of values.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kL1Bits = 13;              // level-1 buckets: top 13 key bits
constexpr int kL1N = 1 << kL1Bits;
constexpr int kL1Shift = 32 - kL1Bits;
constexpr int kL2Bits = 11;              // level-2 buckets: bits [20:10]
constexpr int kL2N = 1 << kL2Bits;
constexpr int kL2Shift = 10;
constexpr int kL2Mask = kL2N - 1;
constexpr int kL3Bits = 10;              // level-3 buckets: bits [9:0]
constexpr int kL3N = 1 << kL3Bits;
constexpr int kL3Mask = kL3N - 1;

constexpr int kHistThreads = 1024;
constexpr int kGatherThreads = 1024;
constexpr int kRefineThreads = 256;
constexpr int kOutThreads = 256;
constexpr int kNChunk = 8;               // CTAs per row for the row passes
constexpr unsigned kInvalid = 0xFFFFFFFFu;

__device__ __forceinline__ unsigned to_key(float v) {
  unsigned b = __float_as_uint(v);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ float from_key(unsigned k) {
  unsigned b = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k;
  return __uint_as_float(b);
}

struct Params {
  unsigned b1;    // cutoff bucket, kInvalid when the row sum is below top_p
  unsigned c1;    // elements inside that bucket
  double sa;      // exact float64 mass of every bucket above b1
  double total;   // exact float64 row sum
};

struct Result {
  unsigned key;   // cutoff ordering key (0 means "keep everything")
  double rscale;  // 1 / retained mass
};

// Locate the largest bucket b with cum(b) >= threshold, where
// cum(b) = sum of all buckets >= b.  Returns -1 when the total is below the
// threshold, in which case the reference keeps every element.
template <int THREADS, int NB>
__device__ int find_cross(const double* __restrict__ bsum, double threshold,
                          double* above_out, double* total_out) {
  constexpr int G = NB / THREADS;
  __shared__ double gsum[THREADS];
  __shared__ double sfx[THREADS];
  __shared__ double total_sh;
  __shared__ int tstar_sh;
  const int t = threadIdx.x;

  // Sum every group of G consecutive buckets.
  const double* base = bsum + static_cast<size_t>(t) * G;
  double s = 0.0;
#pragma unroll
  for (int j = 0; j < G; ++j) s += base[j];
  gsum[t] = s;
  if (t == 0) tstar_sh = -1;
  __syncthreads();

  // Exclusive suffix scan over the groups, from the top bucket downwards.
  sfx[t] = gsum[THREADS - 1 - t];
  __syncthreads();
  for (int off = 1; off < THREADS; off <<= 1) {
    const double v = (t >= off) ? sfx[t - off] : 0.0;
    __syncthreads();
    sfx[t] += v;
    __syncthreads();
  }
  // sfx[t] now holds the mass of every group above group (THREADS-1-t).
  const double above_t = (t == THREADS - 1) ? 0.0 : sfx[THREADS - 2 - t];
  if (t == 0) total_sh = above_t + gsum[0];
  sfx[t] = above_t;
  __syncthreads();

  if (above_t + gsum[t] >= threshold) atomicMax(&tstar_sh, t);
  __syncthreads();

  __shared__ int found_b;
  __shared__ double found_above;
  const int ts = tstar_sh;
  if (t == 0) {
    double acc = 0.0;
    int found = -1;
    if (ts >= 0) {
      acc = sfx[ts];
      for (int j = G - 1; j >= 0; --j) {
        const int b = ts * G + j;
        if (acc + bsum[b] >= threshold) {
          found = b;
          break;
        }
        acc += bsum[b];
      }
    }
    found_b = found;
    found_above = acc;
    if (total_out) *total_out = total_sh;
  }
  __syncthreads();
  if (above_out) *above_out = found_above;
  return found_b;
}

// Stage 1: per-row histogram of the top key bits plus float64 bucket masses.
// The CTA that finishes last scans the merged histogram and publishes Params.
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
k_hist(const float* __restrict__ x, const long row_stride, const int V,
       const float* __restrict__ top, const int nchunk,
       unsigned* __restrict__ gcnt, double* __restrict__ gsum,
       unsigned* __restrict__ arrive, Params* __restrict__ params) {
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned* scnt = reinterpret_cast<unsigned*>(smem);
  double* ssum = reinterpret_cast<double*>(smem + kL1N * sizeof(unsigned));
  const int row = blockIdx.x;

  for (int i = threadIdx.x; i < kL1N; i += THREADS) {
    scnt[i] = 0u;
    ssum[i] = 0.0;
  }
  __syncthreads();

  const float* rp = x + static_cast<long>(row) * row_stride;
  const int chunk = (V + nchunk - 1) / nchunk;
  const int begin = blockIdx.y * chunk;
  const int end = min(begin + chunk, V);
  for (int i = begin + threadIdx.x; i < end; i += THREADS) {
    const float v = rp[i];
    const unsigned b = to_key(v) >> kL1Shift;
    atomicAdd(&scnt[b], 1u);
    atomicAdd(&ssum[b], static_cast<double>(v));
  }
  __syncthreads();

  unsigned* gc = gcnt + static_cast<size_t>(row) * kL1N;
  double* gs = gsum + static_cast<size_t>(row) * kL1N;
  for (int i = threadIdx.x; i < kL1N; i += THREADS) {
    const unsigned c = scnt[i];
    if (!c) continue;
    atomicAdd(&gc[i], c);
    atomicAdd(&gs[i], ssum[i]);
  }
  __threadfence();
  __syncthreads();

  __shared__ int is_last;
  if (threadIdx.x == 0) is_last = (atomicAdd(&arrive[row], 1u) == (unsigned)nchunk - 1);
  __syncthreads();
  if (!is_last) return;

  Params p;
  double total = 0.0;
  const int b1 = find_cross<THREADS, kL1N>(gs, static_cast<double>(__ldg(&top[row])),
                                           &p.sa, &total);
  p.b1 = (b1 < 0) ? kInvalid : static_cast<unsigned>(b1);
  p.c1 = (b1 < 0) ? 0u : gc[b1];
  p.total = total;
  if (threadIdx.x == 0) {
    params[row] = p;
    arrive[row] = 0u;  // leave the gate clean for the next call
  }
}

// Stage 2: keep the keys of the cutoff bucket.  Each CTA owns a private output
// region, so no cross-CTA counter or atomic ordering is required.
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
k_gather(const float* __restrict__ x, const long row_stride, const int V,
         const Params* __restrict__ params, const int nchunk, const int cap,
         uint32_t* __restrict__ cand, unsigned* __restrict__ ccnt) {
  __shared__ unsigned sn;
  const int row = blockIdx.x;
  const int chunk_id = blockIdx.y;
  if (threadIdx.x == 0) sn = 0u;
  __syncthreads();

  const unsigned b1 = params[row].b1;
  uint32_t* region = cand + (static_cast<size_t>(row) * nchunk + chunk_id) * cap;
  const int chunk = (V + nchunk - 1) / nchunk;
  const int begin = chunk_id * chunk;
  const int end = min(begin + chunk, V);
  if (b1 != kInvalid) {
    const float* rp = x + static_cast<long>(row) * row_stride;
    const unsigned lane = threadIdx.x & 31u;
    const unsigned full = 0xFFFFFFFFu;
    for (int i = begin + threadIdx.x; i < end; i += THREADS) {
      const unsigned k = to_key(rp[i]);
      const bool hit = (k >> kL1Shift) == b1;
      const unsigned mask = __ballot_sync(full, hit);
      if (!mask) continue;
      const unsigned n = __popc(mask);
      unsigned base = 0u;
      if (lane == static_cast<unsigned>(__ffs(mask) - 1)) base = atomicAdd(&sn, n);
      base = __shfl_sync(mask, base, __ffs(mask) - 1);
      if (hit) region[base + __popc(mask & ((1u << lane) - 1u))] = k;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) ccnt[row * nchunk + chunk_id] = sn;
}

// Stage 3: resolve the cutoff inside the gathered bucket.  Two more histogram
// levels (11 bits then 10 bits) complete the 32-bit ordering key.
__global__ void __launch_bounds__(kRefineThreads)
k_refine(const uint32_t* __restrict__ cand, const unsigned* __restrict__ ccnt,
         const Params* __restrict__ params, const float* __restrict__ top,
         const int nchunk, const int cap, Result* __restrict__ res) {
  extern __shared__ char rsmem[];
  unsigned* cnt2 = reinterpret_cast<unsigned*>(rsmem);
  double* sum2 = reinterpret_cast<double*>(cnt2 + kL2N);
  unsigned* cnt3 = reinterpret_cast<unsigned*>(sum2 + kL2N);
  double* sum3 = reinterpret_cast<double*>(cnt3 + kL3N);

  const int row = blockIdx.x;
  const Params p = params[row];
  const double threshold = static_cast<double>(__ldg(&top[row]));
  if (p.b1 == kInvalid) {
    if (threadIdx.x == 0) {
      res[row].key = 0u;
      res[row].rscale = 1.0 / p.total;
    }
    return;
  }

  for (int i = threadIdx.x; i < kL2N; i += kRefineThreads) {
    cnt2[i] = 0u;
    sum2[i] = 0.0;
  }
  __syncthreads();
  for (int c = 0; c < nchunk; ++c) {
    const uint32_t* region = cand + (static_cast<size_t>(row) * nchunk + c) * cap;
    const unsigned n = ccnt[row * nchunk + c];
    for (unsigned i = threadIdx.x; i < n; i += kRefineThreads) {
      const unsigned k = region[i];
      const unsigned sub = (k >> kL2Shift) & kL2Mask;
      atomicAdd(&cnt2[sub], 1u);
      atomicAdd(&sum2[sub], static_cast<double>(from_key(k)));
    }
  }
  __syncthreads();
  double sa2 = 0.0;
  const int b2 = find_cross<kRefineThreads, kL2N>(sum2, threshold - p.sa, &sa2, nullptr);
  __syncthreads();

  for (int i = threadIdx.x; i < kL3N; i += kRefineThreads) {
    cnt3[i] = 0u;
    sum3[i] = 0.0;
  }
  __syncthreads();
  if (b2 >= 0) {
    for (int c = 0; c < nchunk; ++c) {
      const uint32_t* region = cand + (static_cast<size_t>(row) * nchunk + c) * cap;
      const unsigned n = ccnt[row * nchunk + c];
      for (unsigned i = threadIdx.x; i < n; i += kRefineThreads) {
        const unsigned k = region[i];
        if (((k >> kL2Shift) & kL2Mask) != static_cast<unsigned>(b2)) continue;
        atomicAdd(&cnt3[k & kL3Mask], 1u);
        atomicAdd(&sum3[k & kL3Mask], static_cast<double>(from_key(k)));
      }
    }
  }
  __syncthreads();
  double sa3 = 0.0;
  const int b3 = find_cross<kRefineThreads, kL3N>(sum3, threshold - p.sa - sa2, &sa3, nullptr);
  if (threadIdx.x == 0 && b2 >= 0 && b3 >= 0) {
    res[row].key = (p.b1 << kL1Shift) | (static_cast<unsigned>(b2) << kL2Shift) |
                   static_cast<unsigned>(b3);
    res[row].rscale = 1.0 / (p.sa + sa2 + sa3 + sum3[b3]);
  } else if (threadIdx.x == 0) {
    // Unreachable for well-formed rows; keep every element rather than guess.
    res[row].key = 0u;
    res[row].rscale = 1.0 / p.total;
  }
}

// Stage 4: mask and rescale, vectorized when the row allows it.
template <int THREADS>
__global__ void __launch_bounds__(THREADS)
k_output_vec(const float4* __restrict__ x, float4* __restrict__ y, const int nv,
             const Result* __restrict__ res) {
  const int row = blockIdx.x;
  const unsigned key = res[row].key;
  const double rscale = res[row].rscale;
  const float4* rp = x + static_cast<size_t>(row) * nv;
  float4* op = y + static_cast<size_t>(row) * nv;
  for (int i = blockIdx.y * THREADS + threadIdx.x; i < nv; i += gridDim.y * THREADS) {
    const float4 v = __ldg(&rp[i]);
    float4 o;
    o.x = (to_key(v.x) >= key) ? static_cast<float>(static_cast<double>(v.x) * rscale) : 0.f;
    o.y = (to_key(v.y) >= key) ? static_cast<float>(static_cast<double>(v.y) * rscale) : 0.f;
    o.z = (to_key(v.z) >= key) ? static_cast<float>(static_cast<double>(v.z) * rscale) : 0.f;
    o.w = (to_key(v.w) >= key) ? static_cast<float>(static_cast<double>(v.w) * rscale) : 0.f;
    op[i] = o;
  }
}

template <int THREADS>
__global__ void __launch_bounds__(THREADS)
k_output_flat(const float* __restrict__ x, float* __restrict__ y, const long row_stride,
              const int V, const Result* __restrict__ res) {
  const int row = blockIdx.x;
  const unsigned key = res[row].key;
  const double rscale = res[row].rscale;
  const float* rp = x + static_cast<long>(row) * row_stride;
  float* op = y + static_cast<long>(row) * row_stride;
  for (int i = blockIdx.y * THREADS + threadIdx.x; i < V; i += gridDim.y * THREADS) {
    const float v = rp[i];
    op[i] = (to_key(v) >= key) ? static_cast<float>(static_cast<double>(v) * rscale) : 0.f;
  }
}

void check_tensor(const torch::Tensor& t, const char* name) {
  TORCH_CHECK_VALUE(t.is_cuda(), name, ": expected CUDA tensor");
  TORCH_CHECK_TYPE(t.scalar_type() == torch::kFloat32, name, ": expected float32");
}

// Scratch buffers live across calls; device memory only, never touched by host.
struct Workspace {
  int B = 0;
  int V = 0;
  int cap = 0;
  unsigned* gcnt = nullptr;
  double* gsum = nullptr;
  unsigned* arrive = nullptr;
  uint32_t* cand = nullptr;
  unsigned* ccnt = nullptr;
  Params* params = nullptr;
  Result* res = nullptr;

  bool fits(int b, int v) const { return gcnt && B >= b && V >= v; }

  void alloc(int b, int v, int c) {
    free_all();
    B = b;
    V = v;
    cap = c;
    const size_t n1 = static_cast<size_t>(B) * kL1N;
    C10_CUDA_CHECK(cudaMalloc(&gcnt, n1 * sizeof(unsigned)));
    C10_CUDA_CHECK(cudaMalloc(&gsum, n1 * sizeof(double)));
    C10_CUDA_CHECK(cudaMalloc(&arrive, static_cast<size_t>(B) * sizeof(unsigned)));
    C10_CUDA_CHECK(cudaMalloc(&cand, static_cast<size_t>(B) * kNChunk * cap * sizeof(uint32_t)));
    C10_CUDA_CHECK(cudaMalloc(&ccnt, static_cast<size_t>(B) * kNChunk * sizeof(unsigned)));
    C10_CUDA_CHECK(cudaMalloc(&params, static_cast<size_t>(B) * sizeof(Params)));
    C10_CUDA_CHECK(cudaMalloc(&res, static_cast<size_t>(B) * sizeof(Result)));
    C10_CUDA_CHECK(cudaMemset(gcnt, 0, n1 * sizeof(unsigned)));
    C10_CUDA_CHECK(cudaMemset(gsum, 0, n1 * sizeof(double)));
    C10_CUDA_CHECK(cudaMemset(arrive, 0, static_cast<size_t>(B) * sizeof(unsigned)));
  }

  void free_all() {
    if (gcnt) cudaFree(gcnt);
    if (gsum) cudaFree(gsum);
    if (arrive) cudaFree(arrive);
    if (cand) cudaFree(cand);
    if (ccnt) cudaFree(ccnt);
    if (params) cudaFree(params);
    if (res) cudaFree(res);
    gcnt = nullptr;
    gsum = nullptr;
    arrive = nullptr;
    cand = nullptr;
    ccnt = nullptr;
    params = nullptr;
    res = nullptr;
  }
};

Workspace& workspace() {
  static Workspace ws;
  return ws;
}

void kernel(const torch::Tensor& probs, const torch::Tensor& top_p, const torch::Tensor& out) {
  check_tensor(probs, "probs");
  check_tensor(top_p, "top_p");
  check_tensor(out, "renorm_probs");
  TORCH_CHECK_VALUE(probs.dim() == 2, "probs: expected rank 2");
  TORCH_CHECK_VALUE(top_p.dim() == 1, "top_p: expected rank 1");
  TORCH_CHECK_VALUE(out.dim() == 2, "renorm_probs: expected rank 2");
  const int B = static_cast<int>(probs.size(0));
  const int V = static_cast<int>(probs.size(1));
  TORCH_CHECK_VALUE(top_p.size(0) == B, "top_p: shape mismatch");
  TORCH_CHECK_VALUE(out.size(0) == B && out.size(1) == V, "renorm_probs: shape mismatch");
  TORCH_CHECK_VALUE(probs.device() == top_p.device() && probs.device() == out.device(),
                    "device mismatch");
  if (B == 0 || V == 0) return;

  const long rs = probs.stride(0);
  const long os = out.stride(0);
  const int chunk = (V + kNChunk - 1) / kNChunk;

  Workspace& ws = workspace();
  if (!ws.fits(B, V)) {
    ws.alloc(B, V, chunk);
    static bool attrs = false;
    if (!attrs) {
      attrs = true;
      const int s1 = kL1N * (sizeof(unsigned) + sizeof(double));
      const int s3 = kL2N * (sizeof(unsigned) + sizeof(double)) +
                     kL3N * (sizeof(unsigned) + sizeof(double));
      C10_CUDA_CHECK(cudaFuncSetAttribute(k_hist<kHistThreads>,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize, s1));
      C10_CUDA_CHECK(cudaFuncSetAttribute(k_refine,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize, s3));
    }
  }

  c10::cuda::CUDAGuard guard(probs.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(probs.get_device()).stream();

  const dim3 grid1(static_cast<unsigned>(B), kNChunk);
  const int s1 = kL1N * (sizeof(unsigned) + sizeof(double));
  k_hist<kHistThreads><<<grid1, kHistThreads, s1, stream>>>(
      probs.data_ptr<float>(), rs, V, top_p.data_ptr<float>(), kNChunk, ws.gcnt, ws.gsum,
      ws.arrive, ws.params);

  k_gather<kGatherThreads><<<grid1, kGatherThreads, 0, stream>>>(
      probs.data_ptr<float>(), rs, V, ws.params, kNChunk, ws.cap, ws.cand, ws.ccnt);

  const int s3 = kL2N * (sizeof(unsigned) + sizeof(double)) +
                 kL3N * (sizeof(unsigned) + sizeof(double));
  k_refine<<<dim3(static_cast<unsigned>(B)), kRefineThreads, s3, stream>>>(
      ws.cand, ws.ccnt, ws.params, top_p.data_ptr<float>(), kNChunk, ws.cap, ws.res);

  const bool vec = (rs == V) && (os == V) && (V % 4 == 0) &&
                   ((reinterpret_cast<uintptr_t>(probs.data_ptr<float>()) & 15u) == 0u) &&
                   ((reinterpret_cast<uintptr_t>(out.data_ptr<float>()) & 15u) == 0u);
  if (vec) {
    k_output_vec<kOutThreads><<<grid1, kOutThreads, 0, stream>>>(
        reinterpret_cast<const float4*>(probs.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()), V / 4, ws.res);
  } else {
    k_output_flat<kOutThreads><<<grid1, kOutThreads, 0, stream>>>(
        probs.data_ptr<float>(), out.data_ptr<float>(), rs, V, ws.res);
    (void)os;
  }
  const auto error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("kernel", &kernel); }

// Top-p (nucleus) renormalisation of dense row probabilities.
//
// Per row: c = max{v : S(v) >= p}, S(v) the float64 sum of the row entries
// >= v; the output is x / S(c) for x >= c, else 0.
//
// A row is split across kChunks blocks and three kernels:
//
//   K1 hist    per-chunk histogram of exact fixed-point mantissa codes
//   K2 gather  merge the chunk histograms, walk them to the bin holding c,
//              stage every entry of the 3-bin window around it into scratch
//   K3 refine  descents over the staged window pin c to one float pattern,
//              then the normalised write
//
// Only scalars travel between phases (bin, anchor, window mass), so the later
// kernels re-read them from scratch instead of synchronising blocks inside one
// launch.  K3 keeps the whole-row algorithm as a fallback for rows the staged
// window cannot describe (huge ties, values outside the histogram range).
//
// Shared-memory atomicAdd(float*) lowers to a compare-and-swap loop on this
// target, so bin masses are carried as 32-bit fixed-point codes split over two
// words:
//
//   code(x) = 2^23 + mantissa(x)              in [2^23, 2^24)
//   mass(b) = (hi * 2^9 + lo) * 2^(binade(b) - 150)
//
// kShift = 18 puts 32 bins in a binade, so no bin straddles one: a bin mass is
// the exact mantissa sum of its entries in a single exponent, no bit dropped.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>

namespace {

constexpr int kThreads = 1024;
constexpr int kWarps = kThreads / 32;
constexpr int kChunks = 4;  // row splits, one block each

constexpr int kShift = 18;                 // pattern bits dropped per bin
constexpr uint32_t kTopKey = 0x7F800000u;  // pattern of +inf
constexpr uint32_t kBinSpan = 1u << kShift;

// Whole-row bins: every finite pattern, used by the fallback path.
constexpr int kFullBins = int(kTopKey >> kShift) + 1;
constexpr int kFullPer = (kFullBins + kThreads - 1) / kThreads;

// Chunk bins: patterns clamped to 2^-64 .. 4, which holds every probability.
constexpr int kMinExp = 64;
constexpr int kMaxExp = 129;
constexpr uint32_t kMinKey = uint32_t(kMinExp) << 23;
constexpr uint32_t kMaxKey = uint32_t(kMaxExp) << 23;
constexpr int kNumBins = ((kMaxExp - kMinExp) << 5) + 1;
constexpr int kBinOff = int(kMinKey >> kShift);
constexpr int kPer = (kNumBins + kThreads - 1) / kThreads;

// Fixed-point code: 23 mantissa bits behind a set implicit bit.
constexpr int kCodeFrac = 23;
constexpr uint32_t kCodeOne = 1u << kCodeFrac;
constexpr uint32_t kCodeMask = kCodeOne - 1u;
constexpr int kHiShift = 9;  // low bits split off to keep both words in range
constexpr uint32_t kLoMask = (1u << kHiShift) - 1u;
constexpr double kExpBias = 127.0 + kCodeFrac;

constexpr int kRefineBits = 9;  // sub-bins per descent
constexpr int kMaxSub = 1 << kRefineBits;
constexpr int kGatherCap = kThreads * 8;  // staged window entries per row
constexpr double kMassEps = 1e-9;         // relative slack for the window check

// Shared control slots.
constexpr int kCtlCross = 0;  // int   : bin holding c, -1 if p is unreachable
constexpr int kCtlCount = 1;  // int   : staged entries
constexpr int kCtlSub = 2;    // int   : crossing sub-bin, -1 if absent
constexpr int kDctlHi = 0;    // double: S(top(bin))
constexpr int kDctlSubHi = 1; // double: S(hi) of the crossing sub-bin
constexpr int kDctlSubLo = 2; // double: S(lo) of the crossing sub-bin

// Per-row scratch: 4 ints then 4 doubles.
constexpr int kMetaCross = 0;
constexpr int kMetaAD = 0;      // S(top(bin))
constexpr int kMetaWinD = 1;    // mass of the staged window
constexpr int kMetaTotalD = 2;  // row mass
constexpr int kMetaStride = 4;

struct Bracket {
  uint32_t lo;
  uint32_t hi;
  double s_hi;  // S(hi)
  double s_lo;  // S(lo)
};

// Whole-row shared state; the fast path uses staged/sub/wsum/ctl only.
struct Shared {
  uint32_t bin_hi[kFullBins];
  uint32_t bin_lo[kFullBins];
  uint32_t staged[kGatherCap];
  uint32_t sub_hi[kMaxSub];
  uint32_t sub_lo[kMaxSub];
  double wsum[kWarps];
  double dctl[4];
  int ictl[4];
};

// Merge shared state: chunk bins, staged window, reduction scratch.
struct MergeShared {
  uint32_t bin_hi[kNumBins];
  uint32_t bin_lo[kNumBins];
  uint32_t staged[kGatherCap];
  double wsum[kWarps];
  double dctl[4];
  int ictl[4];
};

__device__ __forceinline__ int chunk_span(int V) {
  const int per = (V + kChunks - 1) / kChunks;
  return (per + 3) & ~3;
}

// Chunk ranges are clamped to the row so a chunk that starts past `V` keeps an
// empty [c0, c1) instead of a negative one.
__device__ __forceinline__ int chunk_beg(int V, int t) {
  const int b = t * chunk_span(V);
  return b < V ? b : V;
}

__device__ __forceinline__ int chunk_end(int V, int t) {
  const int e = t * chunk_span(V) + chunk_span(V);
  return e < V ? e : V;
}

__device__ __forceinline__ uint32_t clamp_key(uint32_t k) {
  return k > kTopKey ? kTopKey : k;
}

__device__ __forceinline__ double val_of(uint32_t k) {
  return double(__uint_as_float(k));
}

// Value of the exponent holding `base`, as a factor for fixed-point codes.
__device__ __forceinline__ double scale_of(uint32_t base) {
  const int e = int(base >> 23);
  return ldexp(1.0, (e > 1 ? e : 1) - kExpBias);
}

__device__ __forceinline__ double mass_of(uint32_t hi, uint32_t lo, double scale) {
  return (double(hi) * double(1u << kHiShift) + double(lo)) * scale;
}

// Key of the base of chunk bin b.
__device__ __forceinline__ uint32_t bin_key(int b) {
  return (uint32_t(b) << kShift) + kMinKey;
}

__device__ __forceinline__ double bin_mass(const MergeShared& sh, int b) {
  return mass_of(sh.bin_hi[b], sh.bin_lo[b], scale_of(bin_key(b)));
}

__device__ __forceinline__ double full_mass(const Shared& sh, int b) {
  return mass_of(sh.bin_hi[b], sh.bin_lo[b], scale_of(uint32_t(b) << kShift));
}

// Fixed-point code of a positive value's pattern.
__device__ __forceinline__ uint32_t code_of(uint32_t k) {
  return (k & kCodeMask) | kCodeOne;
}

// Sum of `v` over every lane/thread ranked above the caller.
__device__ __forceinline__ double block_suffix(double v, double* scratch) {
  constexpr unsigned kFull = 0xFFFFFFFFu;
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  double incl = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const double t = __shfl_down_sync(kFull, incl, o);
    if (lane + o < 32) incl += t;
  }
  if (lane == 0) scratch[wid] = incl;
  __syncthreads();
  double base = 0.0;
  if (lane == 0) {
    for (int i = wid + 1; i < kWarps; ++i) base += scratch[i];
  }
  base = __shfl_sync(kFull, base, 0);
  return base + (incl - v);
}

__device__ __forceinline__ double block_total(double v, double* scratch) {
  constexpr unsigned kFull = 0xFFFFFFFFu;
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  double incl = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const double t = __shfl_down_sync(kFull, incl, o);
    if (lane + o < 32) incl += t;
  }
  if (lane == 0) scratch[wid] = incl;
  __syncthreads();
  if (wid == 0) {
    double w = (lane < kWarps) ? scratch[lane] : 0.0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) w += __shfl_down_sync(kFull, w, o);
    if (lane == 0) scratch[0] = w;
  }
  __syncthreads();
  const double total = scratch[0];
  __syncthreads();
  return total;
}

// Crossing sub-bin: the unique t with S(hi_t) < p <= S(lo_t).  Returns false
// when the bracket holds no crossing, which leaves it unchanged.
__device__ __forceinline__ bool sub_walk(int nsub, uint32_t base, double s_hi,
                                         double p, Shared& sh) {
  const int t = threadIdx.x;
  const double scale = scale_of(base);
  const double s = (t < nsub) ? mass_of(sh.sub_hi[t], sh.sub_lo[t], scale) : 0.0;
  const double hi_sum = s_hi + block_suffix(s, sh.wsum);
  const double lo_sum = hi_sum + s;
  if (t < nsub && hi_sum < p && lo_sum >= p) {
    sh.ictl[kCtlSub] = t;
    sh.dctl[kDctlSubHi] = hi_sum;
    sh.dctl[kDctlSubLo] = lo_sum;
  }
  __syncthreads();
  return sh.ictl[kCtlSub] >= 0;
}

__device__ __forceinline__ bool descend(Bracket& br, int sub_shift, Shared& sh) {
  const int t = sh.ictl[kCtlSub];
  if (t < 0) return false;
  const uint32_t base = br.lo;
  br.lo = base + (uint32_t(t) << sub_shift);
  br.hi = base + (uint32_t(t + 1) << sub_shift);
  br.s_hi = sh.dctl[kDctlSubHi];
  br.s_lo = sh.dctl[kDctlSubLo];
  return true;
}

// Accumulate a bracket entry into its sub-bin as a fixed-point code relative to
// the bracket base.  Codes compare and sum exactly; only the double conversion
// at walk time rounds.
__device__ __forceinline__ void sub_add(Shared& sh, uint32_t k, uint32_t base_key,
                                        int slot) {
  const uint32_t code = k - base_key;
  atomicAdd(&sh.sub_hi[slot], code >> kHiShift);
  atomicAdd(&sh.sub_lo[slot], code & kLoMask);
}

__device__ __forceinline__ uint32_t bracket_base_key(uint32_t base) {
  const int e = int(base >> 23);
  return uint32_t(e > 1 ? e - 1 : 0) << 23;
}

__device__ __forceinline__ void clear_sub(Shared& sh, int nsub) {
  for (int i = threadIdx.x; i < nsub; i += kThreads) {
    sh.sub_hi[i] = 0u;
    sh.sub_lo[i] = 0u;
  }
  if (threadIdx.x == 0) sh.ictl[kCtlSub] = -1;
  __syncthreads();
}

__device__ __forceinline__ void hist_one(uint32_t* hi, uint32_t* lo, float x) {
  if (!(x > 0.0f)) return;
  const uint32_t k = __float_as_uint(x);
  const uint32_t kc = k < kMinKey ? kMinKey : (k > kMaxKey ? kMaxKey : k);
  const uint32_t code = code_of(k);
  const int b = int((kc >> kShift) - kBinOff);
  atomicAdd(&hi[b], code >> kHiShift);
  atomicAdd(&lo[b], code & kLoMask);
}

__device__ __forceinline__ bool vector_ok(const float* __restrict__ p, int64_t pitch) {
  return ((pitch & 3) == 0) && ((reinterpret_cast<uintptr_t>(p) & 15u) == 0);
}

// Histogram one chunk into shared bins.
__device__ void hist_scan(const float* __restrict__ p, int c0, int c1,
                          uint32_t* hi, uint32_t* lo) {
  const int n = c1 - c0;
  const int n4 = n >> 2;
  const bool vec = ((reinterpret_cast<uintptr_t>(p + c0) & 15u) == 0);
  if (vec) {
    const float4* q = reinterpret_cast<const float4*>(p + c0);
    for (int i = threadIdx.x; i < n4; i += kThreads) {
      const float4 v = q[i];
      hist_one(hi, lo, v.x);
      hist_one(hi, lo, v.y);
      hist_one(hi, lo, v.z);
      hist_one(hi, lo, v.w);
    }
  }
  const int tail = c0 + (vec ? (n4 << 2) : 0);
  for (int i = tail + threadIdx.x; i < c1; i += kThreads) hist_one(hi, lo, p[i]);
}

// Exact per-bin sums of the staged window, or -1 when the crossing lies
// outside the window.
__device__ int pin_bin(int centre, uint32_t win_lo, uint32_t top_centre, double p,
                       Shared& sh) {
  const double anchor = sh.dctl[kDctlHi];
  const int count = sh.ictl[kCtlCount];
  double prev = 0.0, cur = 0.0, next = 0.0;
  for (int i = threadIdx.x; i < count; i += kThreads) {
    const uint32_t k = sh.staged[i];
    if (k < win_lo + kBinSpan) prev += val_of(k);
    else if (k < top_centre) cur += val_of(k);
    else next += val_of(k);
  }
  const double s_prev = block_total(prev, sh.wsum);
  const double s_cur = block_total(cur, sh.wsum);
  const double s_next = block_total(next, sh.wsum);

  const double s_top_prev = anchor + s_cur;  // S(top(centre-1))
  const double s_top_next = anchor - s_next;  // S(top(centre+1))
  int bin = -1;
  double s_hi = 0.0;
  if (anchor < p && s_top_prev >= p) {
    bin = centre;
    s_hi = anchor;
  } else if (s_top_prev < p && s_top_prev + s_prev >= p) {
    bin = centre - 1;
    s_hi = s_top_prev;
  } else if (s_top_next < p && anchor >= p) {
    bin = centre + 1;
    s_hi = s_top_next;
  }
  if (threadIdx.x == 0 && bin >= 0) {
    sh.ictl[kCtlCross] = bin;
    sh.dctl[kDctlHi] = s_hi;
  }
  __syncthreads();
  return bin;
}

// Descents done purely on the staged window (only valid once complete).
__device__ bool refine_staged(int count, Bracket& br, double p, Shared& sh) {
  while (br.hi - br.lo > 1) {
    const int span_bits = 31 - __clz(br.hi - br.lo);
    const int bits = span_bits < kRefineBits ? span_bits : kRefineBits;
    const int sub_shift = span_bits - bits;
    const int nsub = 1 << bits;
    clear_sub(sh, nsub);
    const uint32_t base_key = bracket_base_key(br.lo);

    for (int i = threadIdx.x; i < count; i += kThreads) {
      const uint32_t k = sh.staged[i];
      if (k < br.lo || k >= br.hi) continue;
      sub_add(sh, k, base_key, int((k - br.lo) >> sub_shift));
    }
    __syncthreads();

    if (!sub_walk(nsub, br.lo, br.s_hi, p, sh)) return false;
    if (!descend(br, sub_shift, sh)) return false;
  }
  return true;
}

// One descent over the whole row; stages the bracket for the later descents.
__device__ bool refine_row(const float* __restrict__ in, int V, bool vec, Bracket& br,
                           double p, Shared& sh) {
  const int span_bits = 31 - __clz(br.hi - br.lo);
  const int bits = span_bits < kRefineBits ? span_bits : kRefineBits;
  const int sub_shift = span_bits - bits;
  const int nsub = 1 << bits;
  clear_sub(sh, nsub);
  if (threadIdx.x == 0) sh.ictl[kCtlCount] = 0;
  __syncthreads();

  const uint32_t lo = br.lo, hi = br.hi;
  const uint32_t base_key = bracket_base_key(lo);
  if (vec) {
    const int n4 = V >> 2;
    for (int i = threadIdx.x; i < n4; i += kThreads) {
      const float4 v = reinterpret_cast<const float4*>(in)[i];
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float x = (c == 0) ? v.x : (c == 1) ? v.y : (c == 2) ? v.z : v.w;
        if (!(x > 0.0f)) continue;
        const uint32_t k = clamp_key(__float_as_uint(x) & 0x7FFFFFFFu);
        if (k < lo || k >= hi) continue;
        sub_add(sh, k, base_key, int((k - lo) >> sub_shift));
        const int pos = atomicAdd(&sh.ictl[kCtlCount], 1);
        if (pos < kGatherCap) sh.staged[pos] = k;
      }
    }
    for (int i = (n4 << 2) + threadIdx.x; i < V; i += kThreads) {
      const float x = in[i];
      if (!(x > 0.0f)) continue;
      const uint32_t k = clamp_key(__float_as_uint(x) & 0x7FFFFFFFu);
      if (k < lo || k >= hi) continue;
      sub_add(sh, k, base_key, int((k - lo) >> sub_shift));
      const int pos = atomicAdd(&sh.ictl[kCtlCount], 1);
      if (pos < kGatherCap) sh.staged[pos] = k;
    }
  } else {
    for (int i = threadIdx.x; i < V; i += kThreads) {
      const float x = in[i];
      if (!(x > 0.0f)) continue;
      const uint32_t k = clamp_key(__float_as_uint(x) & 0x7FFFFFFFu);
      if (k < lo || k >= hi) continue;
      sub_add(sh, k, base_key, int((k - lo) >> sub_shift));
      const int pos = atomicAdd(&sh.ictl[kCtlCount], 1);
      if (pos < kGatherCap) sh.staged[pos] = k;
    }
  }
  __syncthreads();

  if (!sub_walk(nsub, br.lo, br.s_hi, p, sh)) return false;
  return descend(br, sub_shift, sh);
}

// Whole-row algorithm: histogram, walk, window pass, descents.  Used when the
// staged window cannot describe the row, so it covers every finite input.
__device__ void row_refine(const float* __restrict__ in, int64_t pitch, int V, double p,
                           Shared& sh, float& cut, double& s_cut) {
  const bool vec = vector_ok(in, pitch);
  for (int i = threadIdx.x; i < kFullBins; i += kThreads) {
    sh.bin_hi[i] = 0u;
    sh.bin_lo[i] = 0u;
  }
  if (threadIdx.x < 4) {
    sh.ictl[threadIdx.x] = (threadIdx.x == kCtlCross || threadIdx.x == kCtlSub) ? -1 : 0;
  }
  if (threadIdx.x < 4) sh.dctl[threadIdx.x] = 0.0;
  __syncthreads();

  // Histogram the whole row.
  const int n4 = V >> 2;
  if (vec) {
    for (int i = threadIdx.x; i < n4; i += kThreads) {
      const float4 v = reinterpret_cast<const float4*>(in)[i];
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float x = (c == 0) ? v.x : (c == 1) ? v.y : (c == 2) ? v.z : v.w;
        if (!(x > 0.0f)) continue;
        const uint32_t k = __float_as_uint(x);
        const uint32_t code = code_of(k);
        const int b = int(k >> kShift);
        atomicAdd(&sh.bin_hi[b], code >> kHiShift);
        atomicAdd(&sh.bin_lo[b], code & kLoMask);
      }
    }
    for (int i = (n4 << 2) + threadIdx.x; i < V; i += kThreads) {
      const float x = in[i];
      if (!(x > 0.0f)) continue;
      const uint32_t k = __float_as_uint(x);
      const uint32_t code = code_of(k);
      const int b = int(k >> kShift);
      atomicAdd(&sh.bin_hi[b], code >> kHiShift);
      atomicAdd(&sh.bin_lo[b], code & kLoMask);
    }
  } else {
    for (int i = threadIdx.x; i < V; i += kThreads) {
      const float x = in[i];
      if (!(x > 0.0f)) continue;
      const uint32_t k = __float_as_uint(x);
      const uint32_t code = code_of(k);
      const int b = int(k >> kShift);
      atomicAdd(&sh.bin_hi[b], code >> kHiShift);
      atomicAdd(&sh.bin_lo[b], code & kLoMask);
    }
  }
  __syncthreads();

  // Descending walk for the bin holding the cutoff.
  double group = 0.0;
  const int b0 = threadIdx.x * kFullPer;
#pragma unroll
  for (int j = 0; j < kFullPer; ++j) {
    const int b = b0 + j;
    if (b < kFullBins) group += full_mass(sh, b);
  }
  const double above_groups = block_suffix(group, sh.wsum);
  double acc = 0.0;
#pragma unroll
  for (int j = kFullPer - 1; j >= 0; --j) {
    const int b = b0 + j;
    if (b >= kFullBins) continue;
    const double s = full_mass(sh, b);
    const double hi_sum = above_groups + acc;
    if (hi_sum < p && hi_sum + s >= p) {
      sh.ictl[kCtlCross] = b;
      sh.dctl[kDctlHi] = hi_sum;
    }
    acc += s;
  }
  __syncthreads();

  const int cross = sh.ictl[kCtlCross];
  if (cross < 0) {
    // p is never reached: the reference clamps to the last token, which keeps
    // the whole row.  Cutoff 0 keeps every non-negative entry.
    cut = 0.0f;
    s_cut = block_total(group, sh.wsum);
    return;
  }

  // Window pass: exact mass above the centre bin plus the staged window.
  const uint32_t top_centre = (uint32_t(cross) + 1u) << kShift;
  const uint32_t win_lo = (cross >= 1 ? uint32_t(cross - 1) : 0u) << kShift;
  uint32_t win_hi = (uint32_t(cross) + 2u) << kShift;
  if (win_hi > kTopKey) win_hi = kTopKey + 1u;

  if (threadIdx.x == 0) sh.ictl[kCtlCount] = 0;
  __syncthreads();

  double above = 0.0;
  if (vec) {
    for (int i = threadIdx.x; i < n4; i += kThreads) {
      const float4 v = reinterpret_cast<const float4*>(in)[i];
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float x = (c == 0) ? v.x : (c == 1) ? v.y : (c == 2) ? v.z : v.w;
        if (!(x > 0.0f)) continue;
        const uint32_t k = clamp_key(__float_as_uint(x) & 0x7FFFFFFFu);
        if (k >= top_centre) above += double(x);
        if (k >= win_lo && k < win_hi) {
          const int pos = atomicAdd(&sh.ictl[kCtlCount], 1);
          if (pos < kGatherCap) sh.staged[pos] = k;
        }
      }
    }
    for (int i = (n4 << 2) + threadIdx.x; i < V; i += kThreads) {
      const float x = in[i];
      if (!(x > 0.0f)) continue;
      const uint32_t k = clamp_key(__float_as_uint(x) & 0x7FFFFFFFu);
      if (k >= top_centre) above += double(x);
      if (k >= win_lo && k < win_hi) {
        const int pos = atomicAdd(&sh.ictl[kCtlCount], 1);
        if (pos < kGatherCap) sh.staged[pos] = k;
      }
    }
  } else {
    for (int i = threadIdx.x; i < V; i += kThreads) {
      const float x = in[i];
      if (!(x > 0.0f)) continue;
      const uint32_t k = clamp_key(__float_as_uint(x) & 0x7FFFFFFFu);
      if (k >= top_centre) above += double(x);
      if (k >= win_lo && k < win_hi) {
        const int pos = atomicAdd(&sh.ictl[kCtlCount], 1);
        if (pos < kGatherCap) sh.staged[pos] = k;
      }
    }
  }
  const double anchor = block_total(above, sh.wsum);
  __syncthreads();
  if (threadIdx.x == 0) sh.dctl[kDctlHi] = anchor;
  __syncthreads();

  const int bin = pin_bin(cross, win_lo, top_centre, p, sh);
  Bracket br;
  const int centre = bin >= 0 ? bin : cross;
  br.lo = uint32_t(centre) << kShift;
  br.hi = br.lo + kBinSpan;
  br.s_hi = sh.dctl[kDctlHi];
  br.s_lo = 0.0;
  const int count = sh.ictl[kCtlCount];
  if (count <= kGatherCap && refine_staged(count, br, p, sh)) {
    cut = __uint_as_float(br.lo);
    s_cut = br.s_lo;
    return;
  }
  for (int i = threadIdx.x; i < kFullBins; i += kThreads) {
    sh.bin_hi[i] = 0u;
    sh.bin_lo[i] = 0u;
  }
  __syncthreads();
  if (!refine_row(in, V, vec, br, p, sh)) return;
  if (sh.ictl[kCtlCount] <= kGatherCap && refine_staged(sh.ictl[kCtlCount], br, p, sh)) {
    cut = __uint_as_float(br.lo);
    s_cut = br.s_lo;
    return;
  }
  refine_row(in, V, vec, br, p, sh);
  cut = __uint_as_float(br.lo);
  s_cut = br.s_lo;
}

// K1: exact bin histogram of one row chunk.
__global__ void hist_kernel(const float* __restrict__ in, int64_t pitch, int V,
                            uint32_t* __restrict__ hist_hi,
                            uint32_t* __restrict__ hist_lo) {
  extern __shared__ char raw[];
  uint32_t* hi = reinterpret_cast<uint32_t*>(raw);
  uint32_t* lo = hi + kNumBins;
  const int chunk = blockIdx.x;
  const int t = chunk % kChunks;
  const float* __restrict__ p = in + int64_t(chunk / kChunks) * pitch;

  for (int i = threadIdx.x; i < kNumBins; i += kThreads) {
    hi[i] = 0u;
    lo[i] = 0u;
  }
  __syncthreads();

  hist_scan(p, chunk_beg(V, t), chunk_end(V, t), hi, lo);
  __syncthreads();

  uint32_t* gh = hist_hi + int64_t(chunk) * kNumBins;
  uint32_t* gl = hist_lo + int64_t(chunk) * kNumBins;
  for (int i = threadIdx.x; i < kNumBins; i += kThreads) {
    gh[i] = hi[i];
    gl[i] = lo[i];
  }
}

// K2: merge the chunk histograms of one row, walk them for the cutoff bin and
// stage the entries of the window around it.
__global__ void gather_kernel(const float* __restrict__ in, int64_t pitch, int V,
                              const float* __restrict__ top_p,
                              const uint32_t* __restrict__ hist_hi,
                              const uint32_t* __restrict__ hist_lo,
                              uint32_t* __restrict__ staged,
                              uint32_t* __restrict__ counts, int* __restrict__ rmeta,
                              double* __restrict__ rmeta_d) {
  extern __shared__ char raw[];
  MergeShared& sh = *reinterpret_cast<MergeShared*>(raw);
  const int chunk = blockIdx.x;
  const int row = chunk / kChunks;
  const int t = chunk % kChunks;
  const float* __restrict__ p = in + int64_t(row) * pitch;
  const double p_row = double(top_p[row]);
  const int c0 = chunk_beg(V, t), c1 = chunk_end(V, t);

  const uint32_t* sh_hi = hist_hi + int64_t(row) * kChunks * kNumBins;
  const uint32_t* sh_lo = hist_lo + int64_t(row) * kChunks * kNumBins;
  if (threadIdx.x < 4) {
    sh.ictl[threadIdx.x] = (threadIdx.x == kCtlCross) ? -1 : 0;
  }
  for (int i = threadIdx.x; i < kNumBins; i += kThreads) {
    uint32_t h = 0u, l = 0u;
    for (int c = 0; c < kChunks; ++c) {
      h += sh_hi[int64_t(c) * kNumBins + i];
      l += sh_lo[int64_t(c) * kNumBins + i];
    }
    sh.bin_hi[i] = h;
    sh.bin_lo[i] = l;
  }
  __syncthreads();

  // Descending walk for the bin holding the cutoff.
  double group = 0.0;
  const int b0 = threadIdx.x * kPer;
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const int b = b0 + j;
    if (b < kNumBins) group += bin_mass(sh, b);
  }
  const double above_groups = block_suffix(group, sh.wsum);
  double acc = 0.0;
#pragma unroll
  for (int j = kPer - 1; j >= 0; --j) {
    const int b = b0 + j;
    if (b >= kNumBins) continue;
    const double s = bin_mass(sh, b);
    const double hi_sum = above_groups + acc;
    if (hi_sum < p_row && hi_sum + s >= p_row) {
      sh.ictl[kCtlCross] = b;
      sh.dctl[kDctlHi] = hi_sum;
    }
    acc += s;
  }
  const double total = block_total(group, sh.wsum);
  __syncthreads();

  const int cross = sh.ictl[kCtlCross];
  const double anchor = sh.dctl[kDctlHi];
  // Window: the crossing bin plus one bin either side.
  const int lo_bin = cross >= 1 ? cross - 1 : 0;
  const int hi_bin = cross + 2 < kNumBins ? cross + 2 : kNumBins;
  const uint32_t win_lo = bin_key(lo_bin);
  const uint32_t win_hi = (hi_bin >= kNumBins) ? kMaxKey + 1u : bin_key(hi_bin);
  double win_mass = 0.0;
  if (cross >= 0) {
    win_mass = mass_of(sh.bin_hi[cross], sh.bin_lo[cross], scale_of(bin_key(cross)));
    if (cross >= 1) {
      win_mass += mass_of(sh.bin_hi[cross - 1], sh.bin_lo[cross - 1],
                          scale_of(bin_key(cross - 1)));
    }
    if (cross + 1 < kNumBins) {
      win_mass += mass_of(sh.bin_hi[cross + 1], sh.bin_lo[cross + 1],
                          scale_of(bin_key(cross + 1)));
    }
  }
  if (t == 0 && threadIdx.x == 0) {
    rmeta[row * kMetaStride + kMetaCross] = cross;
    rmeta_d[row * kMetaStride + kMetaAD] = anchor;
    rmeta_d[row * kMetaStride + kMetaWinD] = win_mass;
    rmeta_d[row * kMetaStride + kMetaTotalD] = total;
  }

  // Stage the window entries of this chunk.
  if (threadIdx.x == 0) sh.ictl[kCtlCount] = 0;
  __syncthreads();
  if (cross >= 0) {
    const uint32_t lo = win_lo, hi = win_hi;
    const int n4 = (c1 - c0) >> 2;
    const bool vec = ((reinterpret_cast<uintptr_t>(p + c0) & 15u) == 0);
    if (vec) {
      const float4* q = reinterpret_cast<const float4*>(p + c0);
      for (int i = threadIdx.x; i < n4; i += kThreads) {
        const float4 v = q[i];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const float x = (c == 0) ? v.x : (c == 1) ? v.y : (c == 2) ? v.z : v.w;
          if (!(x > 0.0f)) continue;
          const uint32_t k = __float_as_uint(x);
          if (k < lo || k >= hi) continue;
          const uint32_t pos = atomicAdd(&sh.ictl[kCtlCount], 1);
          if (pos < kGatherCap) sh.staged[pos] = k;
        }
      }
    }
    const int tail = c0 + (vec ? (n4 << 2) : 0);
    for (int i = tail + threadIdx.x; i < c1; i += kThreads) {
      const float x = p[i];
      if (!(x > 0.0f)) continue;
      const uint32_t k = __float_as_uint(x);
      if (k < lo || k >= hi) continue;
      const uint32_t pos = atomicAdd(&sh.ictl[kCtlCount], 1);
      if (pos < kGatherCap) sh.staged[pos] = k;
    }
  }
  __syncthreads();

  const uint32_t n = uint32_t(sh.ictl[kCtlCount]);
  const uint32_t keep = n < uint32_t(kGatherCap) ? n : uint32_t(kGatherCap);
  counts[chunk] = n;
  uint32_t* dst = staged + int64_t(chunk) * kGatherCap;
  for (uint32_t i = threadIdx.x; i < keep; i += kThreads) dst[i] = sh.staged[i];
}

// Normalised write of one chunk.
__device__ void write_chunk(const float* __restrict__ p, float* __restrict__ op,
                            int c0, int c1, float cut, float inv) {
  const int n = c1 - c0;
  const int n4 = n >> 2;
  const bool vec = ((reinterpret_cast<uintptr_t>(p + c0) & 15u) == 0);
  if (vec) {
    const float4* q = reinterpret_cast<const float4*>(p + c0);
    float4* w = reinterpret_cast<float4*>(op + c0);
    for (int i = threadIdx.x; i < n4; i += kThreads) {
      const float4 v = q[i];
      float4 r;
      r.x = (v.x >= cut) ? v.x * inv : 0.0f;
      r.y = (v.y >= cut) ? v.y * inv : 0.0f;
      r.z = (v.z >= cut) ? v.z * inv : 0.0f;
      r.w = (v.w >= cut) ? v.w * inv : 0.0f;
      w[i] = r;
    }
  }
  const int tail = c0 + (vec ? (n4 << 2) : 0);
  for (int i = tail + threadIdx.x; i < c1; i += kThreads) {
    const float x = p[i];
    op[i] = (x >= cut) ? x * inv : 0.0f;
  }
}

// K3: pin the cutoff from the staged window and write the chunk.
__global__ void refine_kernel(const float* __restrict__ in, int64_t pitch,
                              const float* __restrict__ top_p,
                              float* __restrict__ out, int64_t opitch, int V,
                              const uint32_t* __restrict__ staged,
                              const uint32_t* __restrict__ counts,
                              const int* __restrict__ rmeta,
                              const double* __restrict__ rmeta_d) {
  extern __shared__ char raw[];
  Shared& sh = *reinterpret_cast<Shared*>(raw);
  const int chunk = blockIdx.x;
  const int row = chunk / kChunks;
  const int t = chunk % kChunks;
  const float* __restrict__ p = in + int64_t(row) * pitch;
  float* __restrict__ op = out + int64_t(row) * opitch;
  const double p_row = double(top_p[row]);
  const int cross = rmeta[row * kMetaStride + kMetaCross];
  const uint32_t* cnts = counts + int64_t(row) * kChunks;
  const uint32_t* src = staged + int64_t(row) * kChunks * kGatherCap;

  float cut = 0.0f;
  double s_cut = rmeta_d[row * kMetaStride + kMetaTotalD];
  bool staged_ok = cross >= 0;
  int total = 0;
  if (staged_ok) {
    for (int c = 0; c < kChunks; ++c) {
      const int n = int(cnts[c]);
      if (n >= kGatherCap) {
        staged_ok = false;
        break;
      }
      total += n;
    }
    staged_ok = staged_ok && total <= kGatherCap;
  }

  if (staged_ok) {
    // Merge the chunk windows and check they add up to the histogram mass.
    for (int c = 0; c < kChunks; ++c) {
      int base = 0;
      for (int d = 0; d < c; ++d) base += int(cnts[d]);
      const int n = int(cnts[c]);
      for (int i = threadIdx.x; i < n; i += kThreads) sh.staged[base + i] = src[c * kGatherCap + i];
    }
    if (threadIdx.x == 0) {
      sh.ictl[kCtlCount] = total;
      sh.ictl[kCtlSub] = -1;
      sh.dctl[kDctlHi] = rmeta_d[row * kMetaStride + kMetaAD];
    }
    __syncthreads();

    double m = 0.0;
    for (int i = threadIdx.x; i < total; i += kThreads) m += val_of(sh.staged[i]);
    const double got = block_total(m, sh.wsum);
    const double want = rmeta_d[row * kMetaStride + kMetaWinD];
    const double slack = kMassEps * want;
    if (got < want - slack || got > want + slack) staged_ok = false;
  }

  if (staged_ok) {
    const int lo_bin = cross >= 1 ? cross - 1 : 0;
    const uint32_t win_lo = bin_key(lo_bin);
    const uint32_t top_centre = bin_key(cross + 1);
    const int bin = pin_bin(cross, win_lo, top_centre, p_row, sh);
    if (bin >= 0) {
      Bracket br;
      br.lo = bin_key(bin);
      br.hi = br.lo + kBinSpan;
      br.s_hi = sh.dctl[kDctlHi];
      br.s_lo = 0.0;
      if (refine_staged(total, br, p_row, sh)) {
        cut = __uint_as_float(br.lo);
        s_cut = br.s_lo;
      } else {
        staged_ok = false;
      }
    } else {
      staged_ok = false;
    }
  }

  if (!staged_ok) row_refine(p, pitch, V, p_row, sh, cut, s_cut);

  const float inv = (s_cut > 0.0) ? float(1.0 / s_cut) : 0.0f;
  write_chunk(p, op, chunk_beg(V, t), chunk_end(V, t), cut, inv);
}

constexpr size_t kHistBytes = sizeof(uint32_t) * 2 * kNumBins;
constexpr size_t kMergeBytes = sizeof(MergeShared);
constexpr size_t kRefineBytes = sizeof(Shared);

void configure_once() {
  static std::once_flag once;
  std::call_once(once, [] {
    cudaError_t err = cudaFuncSetAttribute(
        hist_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kHistBytes));
    TORCH_CHECK(err == cudaSuccess, "shared attribute: ", cudaGetErrorString(err));
    err = cudaFuncSetAttribute(gather_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               int(kMergeBytes));
    TORCH_CHECK(err == cudaSuccess, "shared attribute: ", cudaGetErrorString(err));
    err = cudaFuncSetAttribute(refine_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               int(kRefineBytes));
    TORCH_CHECK(err == cudaSuccess, "shared attribute: ", cudaGetErrorString(err));
  });
}

void kernel(const torch::Tensor& probs, const torch::Tensor& top_p,
            const torch::Tensor& out) {
  TORCH_CHECK(probs.is_cuda() && top_p.is_cuda() && out.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(probs.scalar_type() == torch::kFloat32 && top_p.scalar_type() == torch::kFloat32 &&
                  out.scalar_type() == torch::kFloat32,
              "float32 required");
  TORCH_CHECK(probs.dim() == 2 && out.dim() == 2, "rank 2 inputs and outputs");
  TORCH_CHECK(top_p.dim() == 1, "rank 1 top_p");
  const int64_t rows = probs.size(0);
  const int64_t cols = probs.size(1);
  TORCH_CHECK(out.size(0) == rows && out.size(1) == cols, "output shape mismatch");
  TORCH_CHECK(top_p.size(0) == rows, "top_p shape mismatch");
  TORCH_CHECK(probs.stride(1) == 1 && out.stride(1) == 1 && top_p.stride(0) == 1,
              "unit inner stride required");
  TORCH_CHECK(probs.device() == out.device() && probs.device() == top_p.device(),
              "device mismatch");
  if (rows == 0 || cols == 0) return;
  TORCH_CHECK(cols <= INT32_MAX, "row too wide");

  c10::cuda::CUDAGuard guard(probs.device());
  configure_once();
  const auto opts = probs.options();
  const int64_t chunks = rows * kChunks;
  const int64_t bytes = chunks * kNumBins * 2 * int64_t(sizeof(uint32_t)) +
                        chunks * kGatherCap * int64_t(sizeof(uint32_t)) +
                        chunks * int64_t(sizeof(uint32_t)) +
                        rows * kMetaStride * (int64_t(sizeof(int)) + int64_t(sizeof(double)));
  auto buf = torch::empty({bytes}, opts.dtype(torch::kUInt8));
  uint8_t* base = buf.data_ptr<uint8_t>();
  uint32_t* hist_hi = reinterpret_cast<uint32_t*>(base);
  uint32_t* hist_lo = hist_hi + chunks * kNumBins;
  uint32_t* staged = hist_lo + chunks * kNumBins;
  uint32_t* counts = staged + chunks * kGatherCap;
  int* rmeta = reinterpret_cast<int*>(counts + chunks);
  double* rmeta_d = reinterpret_cast<double*>(rmeta + rows * kMetaStride);

  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(probs.get_device()).stream();
  const float* in = probs.data_ptr<float>();
  float* op = out.data_ptr<float>();
  const int V = int(cols);
  hist_kernel<<<uint32_t(chunks), kThreads, kHistBytes, stream>>>(in, probs.stride(0), V,
                                                                 hist_hi, hist_lo);
  gather_kernel<<<uint32_t(chunks), kThreads, kMergeBytes, stream>>>(
      in, probs.stride(0), V, top_p.data_ptr<float>(), hist_hi, hist_lo, staged, counts,
      rmeta, rmeta_d);
  refine_kernel<<<uint32_t(chunks), kThreads, kRefineBytes, stream>>>(
      in, probs.stride(0), top_p.data_ptr<float>(), op, out.stride(0), V, staged, counts,
      rmeta, rmeta_d);
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "top_p launch: ", cudaGetErrorString(err));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("kernel", &kernel);
}

// SM90 primitives: cluster shared-memory addressing, split cluster barriers,
// bulk async copies, and warp scan steps.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace native {

constexpr int kWarp = 32;
constexpr unsigned kWarpMask = 0xffffffffu;

__device__ __forceinline__ unsigned address(const void* ptr) {
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

// Address of `ptr` inside the shared memory of another CTA of the cluster.
__device__ __forceinline__ unsigned remote(const void* ptr, unsigned rank) {
  unsigned result;
  asm("mapa.shared::cluster.u32 %0, %1, %2;"
      : "=r"(result) : "r"(address(ptr)), "r"(rank));
  return result;
}

__device__ __forceinline__ void add_remote(unsigned ptr, unsigned value) {
  asm volatile("red.relaxed.cluster.shared::cluster.add.u32 [%0], %1;"
      : : "r"(ptr), "r"(value) : "memory");
}

// Rank 0 publishes its ballot; every peer spins until the generation matches.
__device__ __forceinline__ void store_release(uint64_t* ptr, uint64_t value) {
  asm volatile("st.release.cluster.shared::cta.b64 [%0], %1;"
      : : "r"(address(ptr)), "l"(value) : "memory");
}

__device__ __forceinline__ uint64_t poll(unsigned ptr, unsigned generation) {
  uint64_t value;
  asm volatile("{ .reg .u64 v, t; .reg .u32 g; .reg .pred p; "
      "poll_loop: ld.acquire.cluster.shared::cluster.u64 v, [%1]; "
      "shr.u64 t, v, 48; cvt.u32.u64 g, t; setp.ne.u32 p, g, %2; "
      "@p bra poll_loop; mov.u64 %0, v; }"
      : "=l"(value) : "r"(ptr), "r"(generation) : "memory");
  return value;
}

__device__ __forceinline__ uint64_t load_remote(unsigned ptr) {
  uint64_t result;
  asm volatile("ld.shared::cluster.u64 %0, [%1];" : "=l"(result) : "r"(ptr));
  return result;
}

// Split cluster barrier: arrive releases prior stores, wait acquires remote ones.
__device__ __forceinline__ void arrive() {
  asm volatile("barrier.cluster.arrive.release.aligned;" : : : "memory");
}

__device__ __forceinline__ void wait() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" : : : "memory");
}

__device__ __forceinline__ void sync() {
  arrive();
  wait();
}

__device__ __forceinline__ unsigned rank() {
  unsigned value;
  asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(value));
  return value;
}

// One lane of warp zero, so a single thread performs the CTA-wide work.
__device__ __forceinline__ bool elect() {
  const unsigned warp = __shfl_sync(kWarpMask, threadIdx.x / kWarp, 0);
  if (warp != 0) return false;

  unsigned elected;
  asm volatile("{ .reg .pred p; elect.sync _|p, %1; selp.u32 %0, 1, 0, p; }"
      : "=r"(elected) : "r"(kWarpMask));
  return elected != 0;
}

// Bulk asynchronous copy: one thread issues, every consumer waits on the barrier.
__device__ __forceinline__ void init_barrier(uint64_t* bar) {
  asm volatile("mbarrier.init.shared.b64 [%0], 1;" : : "r"(address(bar)) : "memory");
}

__device__ __forceinline__ void copy_async(float* dst, const float* src,
    unsigned bytes, uint64_t* bar) {
  const unsigned b = address(bar);
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];"
      : : "r"(address(dst)), "l"(src), "r"(bytes), "r"(b) : "memory");
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
      : : "r"(b), "r"(bytes) : "memory");
}

__device__ __forceinline__ void wait_copy(uint64_t* bar) {
  const unsigned b = address(bar);
  // Each resident slot is filled once, so every wait observes parity zero.
  asm volatile("{ .reg .pred p; wait_loop: "
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0; "
      "@!p bra wait_loop; }" : : "r"(b) : "memory");
}

__device__ __forceinline__ unsigned scan_step(unsigned value, int offset) {
  return __shfl_up_sync(kWarpMask, value, offset);
}
}  // namespace native

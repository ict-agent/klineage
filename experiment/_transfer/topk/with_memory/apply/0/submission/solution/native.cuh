// Low-level device primitives: shared/cluster addressing, DSMEM ops, barriers.
#pragma once

#include <cstdint>

namespace topk {
namespace native {

constexpr unsigned kWarp = 32u;
constexpr unsigned kWarpMask = 0xffffffffu;

__device__ __forceinline__ unsigned address(const void* ptr) {
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

// Map a local shared address onto the same offset of CTA `rank` (DSMEM).
__device__ __forceinline__ unsigned remote(const void* ptr, unsigned rank) {
  unsigned addr;
  asm("mapa.shared::cluster.u32 %0, %1, %2;"
      : "=r"(addr) : "r"(address(ptr)), "r"(rank));
  return addr;
}

// Fire-and-forget local increment; no dependency on a returned value.
__device__ __forceinline__ void red_shared(unsigned* ptr) {
  asm volatile("red.shared.add.u32 [%0], 1;" : : "r"(address(ptr)) : "memory");
}

__device__ __forceinline__ void red_remote(unsigned addr, unsigned value) {
  asm volatile("red.relaxed.cluster.shared::cluster.add.u32 [%0], %1;"
      : : "r"(addr), "r"(value) : "memory");
}

__device__ __forceinline__ void store_remote(unsigned addr, unsigned value) {
  asm volatile("st.shared::cluster.u32 [%0], %1;"
      : : "r"(addr), "r"(value) : "memory");
}

__device__ __forceinline__ unsigned load_remote(unsigned addr) {
  unsigned value;
  asm volatile("ld.shared::cluster.u32 %0, [%1];"
      : "=r"(value) : "r"(addr) : "memory");
  return value;
}

// Split-phase cluster barrier: overlap independent CTA-local work between them.
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

// One thread per CTA; used for single-owner shared updates.
__device__ __forceinline__ bool elect() {
  if (threadIdx.x >= kWarp) return false;
  unsigned elected;
  asm volatile("{ .reg .pred p; elect.sync _|p, %1; selp.u32 %0, 1, 0, p; }"
      : "=r"(elected) : "r"(kWarpMask));
  return elected != 0;
}

// Inclusive warp prefix step over a 32-lane warp.
__device__ __forceinline__ unsigned scan_step(unsigned value, int offset) {
  unsigned result;
  asm volatile("{ .reg .u32 v; .reg .pred p; "
      "shfl.sync.up.b32 v|p, %1, %2, 0, %3; "
      "@p add.u32 v, v, %1; mov.u32 %0, v; }"
      : "=r"(result) : "r"(value), "r"(offset), "r"(kWarpMask));
  return result;
}

}  // namespace native
}  // namespace topk

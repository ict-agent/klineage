/**
 * ub_to_gm_copy.cpp - UB to GM copy custom op
 *
 * Copies data from a UB buffer (with configurable src_offset and length)
 * to a GM memref destination.
 *
 * This allows storing a sub-region of a UB buffer directly to GM without
 * needing to allocate a separate UB output buffer (saves UB space).
 *
 * Compile:
 *   ccec -O2 -x cce ub_to_gm_copy.cpp -emit-llvm -c --cce-auto-sync=off \
 *     --cce-aicore-only --cce-generic-addrspace=off \
 *     --cce-aicore-arch=dav-c220-vec --cce-enable-print \
 *     --cce-enable-sanitizer -std=c++17 \
 *     -I <path>/bishengir/lib/Template/include -o ub_to_gm_copy.bc
 */

#include "DMA/DMAUtils.h"

extern "C" {

/// Copy a sub-region of a UB buffer to GM.
///
/// \param src:        UB memref<N x f32> — source UB buffer
/// \param src_offset: element offset in src where copy starts
/// \param copy_len:   number of f32 elements to copy
/// \param dst:        GM memref<M x f32> — destination in GM
__aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_ub_to_gm_copy_float(
    memref_t<__ubuf__ float, 1> *src,
    int64_t src_offset,
    int64_t copy_len,
    memref_t<__gm__ float, 1> *dst) {

  memref_t<__ubuf__ float, 1> src_ref{
      src->aligned, src->allocated,
      src->offset + src_offset,
      {copy_len}, {1}};
  memref_t<__gm__ float, 1> dst_ref{
      dst->aligned, dst->allocated, dst->offset,
      {copy_len}, {1}};

  INTRINSIC(pipe_barrier, PIPE_V);
  store_ubuf_to_gm_1d_core_with_contiguous_last_dim(&src_ref, &dst_ref);
  INTRINSIC(pipe_barrier, PIPE_MTE3);
}

} // extern "C"

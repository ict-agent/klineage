/**
 * gm_to_ub_copy.cpp - GM to UB copy custom op
 *
 * Copies a GM memref (1D, contiguous) into an existing UB buffer at dst_offset.
 * The GM input is passed as a memref (from tl.load of a block_ptr).
 *
 * Compile:
 *   ccec -O2 -x cce gm_to_ub_copy.cpp -emit-llvm -c --cce-auto-sync=off \
 *     --cce-aicore-only --cce-generic-addrspace=off \
 *     --cce-aicore-arch=dav-c220-vec --cce-enable-print \
 *     --cce-enable-sanitizer -std=c++17 \
 *     -I <path>/bishengir/lib/Template/include -o gm_to_ub_copy.bc
 */

#include "DMA/DMAUtils.h"

extern "C" {

/// Copy src (GM tensor) into dst (UB buffer) at dst_offset.
///
/// \param src:        GM memref<N x f32> — source data (from tl.load of block_ptr)
/// \param dst_offset: element offset in UB dst where data lands
/// \param dst:        UB memref<M x f32> — destination buffer
__aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_gm_to_ub_copy_float(
    memref_t<__gm__ float, 1> *src,
    int64_t dst_offset,
    memref_t<__ubuf__ float, 1> *dst) {

  int64_t copy_len = src->sizes[0];

  memref_t<__gm__ float, 1> src_ref{
      src->aligned, src->allocated, src->offset,
      {copy_len}, {1}};
  memref_t<__ubuf__ float, 1> dst_ref{
      dst->aligned, dst->allocated,
      dst->offset + dst_offset,
      {copy_len}, {1}};

  load_gm_to_ubuf_1d_core_with_contiguous_last_dim(&src_ref, &dst_ref, 0);
  INTRINSIC(pipe_barrier, PIPE_MTE2);
}

} // extern "C"

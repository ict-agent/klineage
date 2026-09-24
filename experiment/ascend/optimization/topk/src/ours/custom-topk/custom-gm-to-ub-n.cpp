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

/// Size-aware copy: copy exactly `copy_len` f32 elements from src (GM) into dst
/// (UB) at dst_offset, regardless of the block_ptr's block_shape.
///
/// Why this exists:
///   The caller's tl.make_block_ptr must use a *compile-time constant*
///   block_shape (triton constraint), so it is fixed at the worst-case size
///   (CHUNK*2). But the real number of valid f32 to move is a *runtime* scalar
///   (e.g. remaining proposals * 2) and is often smaller than the block_shape.
///   Using the fixed block_shape would over-read GM (reading invalid/garbage
///   data, and risking out-of-bounds at allocation boundaries).
///   This op copies only `copy_len` elements (never the full block_shape).
///
/// \param src:        GM memref<N x f32> — source (from tl.load of block_ptr).
///                    N == block_shape is the *capacity*, not the copy amount.
/// \param dst_offset: element offset in UB dst where data lands.
/// \param copy_len:   number of f32 elements to actually copy (runtime scalar).
///                    Always <= N (the block capacity); copies exactly this many.
/// \param dst:        UB memref<M x f32> — destination buffer.
__aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_gm_to_ub_copy_n_float(
    memref_t<__gm__ float, 1> *src,
    int64_t dst_offset,
    int64_t copy_len,
    memref_t<__ubuf__ float, 1> *dst) {

  // copy_len is a runtime scalar that is ALWAYS <= the block capacity
  // (src->sizes[0] == block_shape == CHUNK*2). We must copy exactly copy_len,
  // NOT the full block, otherwise we over-read GM past the valid data.
  // No upper clamp is needed (copy_len can never exceed the block). We only
  // guard the degenerate empty case.
  if (copy_len <= 0) {
    return;
  }

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

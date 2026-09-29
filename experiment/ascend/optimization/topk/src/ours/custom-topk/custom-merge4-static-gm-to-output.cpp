/**
 * custom-merge4-static-gm-to-output.cpp
 *
 * Merge up to 4 short sorted proposal runs and unpack directly to final
 * TopK value/index outputs.  This is intended for the small-K grouped path
 * where each input run length is <= 512 proposals.
 */

#include "DMA/DMAUtils.h"
#include "Vector/Sort/SortUtils.h"
#include "Vector/VecUtils.h"

constexpr uint64_t EXHAUST_BIT = (uint64_t)1 << 12;
constexpr int64_t NPP = PROPOSALS_BYTES / sizeof(float);
constexpr int64_t MAX_RUN_PROPS = 512;
constexpr int64_t MAX_RUN_F32 = MAX_RUN_PROPS * NPP;
constexpr int64_t OUT_PROPS = 4 * MAX_RUN_PROPS;
constexpr int64_t OUT_F32 = OUT_PROPS * NPP + 8;
constexpr int64_t UNPACK_CHUNK_PROPS = MAX_RUN_PROPS;

__aiv__ __attribute__((always_inline)) uint32_t
vms4_consumed_merge4(uint64_t sr, int i) {
  return (uint32_t)((sr >> (16 * i)) & 0xFFFF);
}

__aiv__ __attribute__((always_inline)) uint64_t
vmrgsort4_exhaust_merge4(__ubuf__ float *dst, __ubuf__ float **xn,
                         const uint32_t *lens, int ways) {
  uint64_t xm = 0;
  for (int i = 0; i < ways; ++i) {
    xm |= ((uint64_t)(lens[i] & 0xFFFF)) << (16 * i);
  }
  uint64_t mask = ((1ull << ways) - 1) & 0xF;
  uint64_t config = (mask << 8) | EXHAUST_BIT | 1;
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC(vmrgsort4, dst, xn, xm, config);
  return (uint64_t)get_vms4_sr();
}

__aiv__ __attribute__((always_inline)) void
load_gm_chunk_merge4(memref_t<__gm__ float, 1> *src, int64_t src_base_f32,
                     __ubuf__ float *dst_ub, int64_t copy_f32) {
  if (copy_f32 <= 0) {
    return;
  }
  memref_t<__gm__ float, 1> src_ref{
      src->aligned, src->allocated, src->offset + src_base_f32,
      {copy_f32}, {1}};
  memref_t<__ubuf__ float, 1> dst_ref{
      dst_ub, dst_ub, 0, {copy_f32}, {1}};
  load_gm_to_ubuf_1d_core_with_contiguous_last_dim(&src_ref, &dst_ref, 0);
  INTRINSIC(pipe_barrier, PIPE_MTE2);
  INTRINSIC(pipe_barrier, PIPE_ALL);
}

__aiv__ __attribute__((always_inline)) void
store_value_chunk_merge4(__ubuf__ float *src_ub, int64_t copy_num,
                         memref_t<__gm__ float, 1> *dst,
                         int64_t dst_off) {
  if (copy_num <= 0) {
    return;
  }
  memref_t<__ubuf__ float, 1> src_ref{
      src_ub, src_ub, 0, {copy_num}, {1}};
  memref_t<__gm__ float, 1> dst_ref{
      dst->aligned, dst->allocated, dst->offset + dst_off,
      {copy_num}, {1}};
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC(pipe_barrier, PIPE_ALL);
  store_ubuf_to_gm_1d_core_with_contiguous_last_dim(&src_ref, &dst_ref);
  INTRINSIC(pipe_barrier, PIPE_MTE3);
  INTRINSIC(pipe_barrier, PIPE_ALL);
}

__aiv__ __attribute__((always_inline)) void
store_index_chunk_merge4(__ubuf__ int32_t *src_ub, int64_t copy_num,
                         memref_t<__gm__ int32_t, 1> *dst,
                         int64_t dst_off) {
  if (copy_num <= 0) {
    return;
  }
  memref_t<__ubuf__ int32_t, 1> src_ref{
      src_ub, src_ub, 0, {copy_num}, {1}};
  memref_t<__gm__ int32_t, 1> dst_ref{
      dst->aligned, dst->allocated, dst->offset + dst_off,
      {copy_num}, {1}};
  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC(pipe_barrier, PIPE_ALL);
  store_ubuf_to_gm_1d_core_with_contiguous_last_dim(&src_ref, &dst_ref);
  INTRINSIC(pipe_barrier, PIPE_MTE3);
  INTRINSIC(pipe_barrier, PIPE_ALL);
}

__aiv__ __attribute__((always_inline)) void
unpack_proposals_merge4(memref_t<__ubuf__ float, 1> *src,
                        memref_t<__ubuf__ float, 1> *dst_value,
                        memref_t<__ubuf__ int32_t, 1> *dst_index,
                        int64_t real_num) {
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, real_num);

  memref_t<__ubuf__ int32_t, 1> src_int32;
  view_as<float, int32_t, 1>(src, &src_int32);
  vreducev2_1d_with_pattern_mode<int32_t, PatternMode::INDEX_1_FROM_2_ELEMENTS>(
      &src_int32, dst_index);

  vreducev2_1d_with_pattern_mode<float, PatternMode::INDEX_0_FROM_2_ELEMENTS>(
      src, dst_value);

  INTRINSIC_NO_ARGS(set_mask_norm);
}

__aiv__ __attribute__((always_inline)) void
write_output_props_merge4(__ubuf__ float *src_props, int64_t src_off_props,
                          int64_t copy_props,
                          memref_t<__gm__ float, 1> *dst_value,
                          memref_t<__gm__ int32_t, 1> *dst_index,
                          int64_t dst_off_props,
                          __ubuf__ float *dval,
                          __ubuf__ int32_t *didx) {
  if (copy_props <= 0) {
    return;
  }
  memref_t<__ubuf__ float, 1> src_view{
      src_props, src_props, src_off_props * NPP,
      {copy_props * NPP}, {1}};
  memref_t<__ubuf__ float, 1> dval_view{
      dval, dval, 0, {copy_props}, {1}};
  memref_t<__ubuf__ int32_t, 1> didx_view{
      didx, didx, 0, {copy_props}, {1}};

  INTRINSIC(pipe_barrier, PIPE_V);
  unpack_proposals_merge4(&src_view, &dval_view, &didx_view, copy_props);
  store_value_chunk_merge4(dval, copy_props, dst_value, dst_off_props);
  store_index_chunk_merge4(didx, copy_props, dst_index, dst_off_props);
}

extern "C" {

__aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_merge4_static_gm_to_output_float(
    memref_t<__gm__ float, 1> *src,
    int64_t run_len,
    int64_t topk,
    int64_t num_runs,
    memref_t<__gm__ float, 1> *dst_value,
    memref_t<__gm__ int32_t, 1> *dst_index) {
  __ubuf__ float in0[MAX_RUN_F32];
  __ubuf__ float in1[MAX_RUN_F32];
  __ubuf__ float in2[MAX_RUN_F32];
  __ubuf__ float in3[MAX_RUN_F32];
  __ubuf__ float out_props[OUT_F32];
  __ubuf__ float dval[UNPACK_CHUNK_PROPS];
  __ubuf__ int32_t didx[UNPACK_CHUNK_PROPS];
  __ubuf__ float *lanes[4] = {in0, in1, in2, in3};
  __ubuf__ float *xn[4];
  uint32_t lens[4] = {0, 0, 0, 0};

  int64_t active_runs = num_runs;
  if (active_runs > 4) {
    active_runs = 4;
  }
  if (active_runs < 0) {
    active_runs = 0;
  }

  int64_t copy_props = run_len;
  if (copy_props > topk) {
    copy_props = topk;
  }
  if (copy_props > MAX_RUN_PROPS) {
    copy_props = MAX_RUN_PROPS;
  }

  int active = 0;
  for (int i = 0; i < 4; ++i) {
    if (i >= active_runs || copy_props <= 0) {
      continue;
    }
    load_gm_chunk_merge4(src, i * run_len * NPP, lanes[active],
                         copy_props * NPP);
    xn[active] = lanes[active];
    lens[active] = (uint32_t)copy_props;
    active++;
  }

  if (active <= 0) {
    return;
  }

  int64_t need = topk;
  if (need > MAX_RUN_PROPS) {
    need = MAX_RUN_PROPS;
  }

  if (active == 1) {
    if (need > copy_props) {
      need = copy_props;
    }
    write_output_props_merge4(lanes[0], 0, need,
                              dst_value, dst_index, 0, dval, didx);
    return;
  }

  uint64_t sr = vmrgsort4_exhaust_merge4(out_props, xn, lens, active);
  int64_t produced = 0;
  for (int w = 0; w < active; ++w) {
    produced += (int64_t)vms4_consumed_merge4(sr, w);
  }
  if (need > produced) {
    need = produced;
  }

  write_output_props_merge4(out_props, 0, need,
                            dst_value, dst_index, 0, dval, didx);
}

} // extern "C"

/**
 * merge-sort-exhaust.cpp — Truncation-mode (exhaustion) 4-way merge sort
 * =====================================================================
 *
 * 与 merge-sort.cpp(全排序, config[12]=0)的区别:
 *   - 本文件使用 *截断/耗尽模式* vmrgsort4(config[12] = isAllStored = 1):
 *     归并进行到 "任意一路片上数据耗尽" 即停止,不要求把 4 路全部排完。
 *   - 每条 vmrgsort4 之后用 get_vms4_sr() 读回 VMS4_SR,
 *     该寄存器按 16-bit 打包记录 4 路各自实际消耗的 proposal 数:
 *         VMS4_SR[15:0]  = 路0 消耗
 *         VMS4_SR[31:16] = 路1 消耗
 *         VMS4_SR[47:32] = 路2 消耗
 *         VMS4_SR[63:48] = 路3 消耗
 *   - 软件据此推进各路读指针(cursor)与剩余量(remaining),并把消耗数写出,
 *     用于决定 "下一次 load 各路多少、从哪读"。
 *
 * 数据格式(与 merge-sort.cpp 一致):
 *   proposal = [value(f32), index(i32 重解释为 f32)] 共 8 字节。
 *   输入 src_proposals[N*2] f32,按 block_size 个 proposal 为一段,段内降序。
 *
 * config 位域(由 intrisics.h 的 vmrgsort4 友好封装确证):
 *   config = (repeat & 0xff) | ((maskSignal & 0xf) << 8) | ((isAllStored & 1) << 12)
 *   maskSignal: 0x3=2路 0x7=3路 0xf=4路
 *
 * Compile(与 merge-sort.cpp 相同):
 *  ccec -O2 -x cce merge-sort-exhaust.cpp -emit-llvm -c \
 *    --cce-auto-sync=off --cce-aicore-only --cce-generic-addrspace=off \
 *    -mllvm -disable-llvm-optzns --cce-aicore-arch=dav-c220-vec \
 *    --cce-enable-print --cce-enable-sanitizer -std=c++17 \
 *    -I <path>/bishengir/lib/Template/include -o merge_sort_exhaust.bc
 */

#include "Vector/Sort/SortUtils.h"
#include "Vector/VecUtils.h"
#include "DMA/DMAUtils.h"

// 截断模式 config[12] = isAllStored = 1
constexpr uint64_t EXHAUST_BIT = (uint64_t)1 << 12;

// 读 VMS4_SR(各路消耗,16-bit 打包)。get_vms4_sr 见 intrisics.h #2028。
__aiv__ __attribute__((always_inline)) uint64_t read_vms4_sr() {
  return (uint64_t)get_vms4_sr();
}

// 从打包寄存器取第 i 路(0..3)的消耗 proposal 数
__aiv__ __attribute__((always_inline)) uint32_t
vms4_consumed(uint64_t sr, int i) {
  return (uint32_t)((sr >> (16 * i)) & 0xFFFF);
}

// ============================================================================
// 一次截断式 K 路归并(K=2/3/4),返回 VMS4_SR。
//   dst:  输出 proposal 起始
//   xn:   K 路输入起始指针数组
//   lens: K 路输入长度(proposal 数)
//   ways: 2/3/4
// 输出长度 = 各路消耗之和(= 归并到某路耗尽时已输出的安全前缀)。
// ============================================================================
template <typename T>
__aiv__ __attribute__((always_inline)) uint64_t
vmrgsort4_exhaust(__ubuf__ T *dst, __ubuf__ T **xn, const uint32_t *lens,
                  int ways) {
  // src1 = 4×16bit 各路长度打包(未用的路长度填 0)
  uint64_t xm = 0;
  for (int i = 0; i < ways; ++i)
    xm |= ((uint64_t)(lens[i] & 0xFFFF)) << (16 * i);

  // maskSignal: ways 路有效 → 低 ways 位为 1
  uint64_t mask = ((1ull << ways) - 1) & 0xF;
  uint64_t config = (mask << 8) | EXHAUST_BIT | 1; // repeat=1, 截断模式

  INTRINSIC(pipe_barrier, PIPE_V);
  INTRINSIC(vmrgsort4, dst, xn, xm, config);

  // 关键:读回各路消耗(截断模式专用反馈通道)
  return read_vms4_sr();
}

// ============================================================================
// 主入口:streaming 截断归并,并把各路消耗写出。
//
//   src_proposals: [num_lists * list_len * 2] f32,num_lists 段各自降序
//   dst_proposals: [topk * 2] f32,归并后的前 topk 个 proposal
//   consumed_out:  [num_lists] i32,每路最终累计消耗的 proposal 数(关键产物!)
//   tmp_buf:       ping-pong 工作区
//   num_lists:     输入有序段数(2/3/4)
//   list_len:      每段 proposal 数
//   topk:          截断保留的 proposal 数
//
// 设计:单次 vmrgsort4(截断模式)即可演示 "归并 + 读回消耗"。
// 若 topk > 单次耗尽产出,则循环续做,推进 cursor[]/remaining[]。
// ============================================================================
__aiv__ __attribute__((always_inline)) void
merge_exhaust_with_consumed(memref_t<__ubuf__ float, 1> *src_proposals,
                            memref_t<__ubuf__ float, 1> *dst_proposals,
                            memref_t<__ubuf__ int32_t, 1> *consumed_out,
                            memref_t<__ubuf__ float, 1> *tmp_buf,
                            int64_t num_lists, int64_t list_len,
                            int64_t topk) {
  constexpr int64_t npp = PROPOSALS_BYTES / sizeof(float); // 2

  auto src_ptr = src_proposals->aligned + src_proposals->offset;
  auto dst_ptr = dst_proposals->aligned + dst_proposals->offset;
  auto cons_ptr = consumed_out->aligned + consumed_out->offset;

  // 各路 cursor(已消耗的 proposal 数)与 remaining(剩余 proposal 数)
  uint32_t cursor[4] = {0, 0, 0, 0};
  uint32_t remaining[4] = {0, 0, 0, 0};
  for (int i = 0; i < num_lists; ++i)
    remaining[i] = (uint32_t)list_len;

  int64_t produced = 0; // 已写入 dst 的 proposal 数

  while (produced < topk) {
    // 统计当前仍有数据的有效路,构造 xn / lens
    __ubuf__ float *xn[4];
    uint32_t lens[4] = {0, 0, 0, 0};
    int ways = 0;
    int active_idx[4];
    for (int i = 0; i < num_lists; ++i) {
      if (remaining[i] == 0)
        continue;
      // 该路在 src 中的基址 = (i*list_len + cursor[i]) 个 proposal
      int64_t off_props = (int64_t)i * list_len + cursor[i];
      xn[ways] = src_ptr + off_props * npp;
      lens[ways] = remaining[i];
      active_idx[ways] = i;
      ways++;
    }
    if (ways == 0)
      break;

    if (ways == 1) {
      // 只剩一路:无需比较,直接 copy 残余到 dst(标准 k 路归并尾段优化)
      int i = active_idx[0];
      int64_t cnt = remaining[i];
      if (produced + cnt > topk)
        cnt = topk - produced;
      int64_t off_props = (int64_t)i * list_len + cursor[i];
      memref_t<__ubuf__ float, 1> from{src_proposals->aligned,
                                       src_proposals->allocated,
                                       src_proposals->offset + off_props * npp,
                                       {cnt * npp},
                                       {1}};
      memref_t<__ubuf__ float, 1> to{dst_proposals->aligned,
                                     dst_proposals->allocated,
                                     dst_proposals->offset + produced * npp,
                                     {cnt * npp},
                                     {1}};
      copy_ubuf_to_ubuf_1d_core(&from, &to);
      cursor[i] += (uint32_t)cnt;
      remaining[i] -= (uint32_t)cnt;
      produced += cnt;
      break;
    }

    // 多路截断归并:输出到 tmp_buf 头部
    auto tmp_ptr = tmp_buf->aligned + tmp_buf->offset;
    uint64_t sr = vmrgsort4_exhaust<float>(tmp_ptr, xn, lens, ways);

    // 拆解各路消耗,更新 cursor/remaining,并累加产出
    int64_t batch_out = 0;
    for (int w = 0; w < ways; ++w) {
      uint32_t c = vms4_consumed(sr, w);
      int i = active_idx[w];
      cursor[i] += c;
      remaining[i] -= c;
      batch_out += c;
    }

    // 把本批安全前缀从 tmp 拷到 dst(可能被 topk 截断)
    int64_t copy_props = batch_out;
    if (produced + copy_props > topk)
      copy_props = topk - produced;
    if (copy_props > 0) {
      memref_t<__ubuf__ float, 1> from{tmp_buf->aligned, tmp_buf->allocated,
                                       tmp_buf->offset,
                                       {copy_props * npp},
                                       {1}};
      memref_t<__ubuf__ float, 1> to{dst_proposals->aligned,
                                     dst_proposals->allocated,
                                     dst_proposals->offset + produced * npp,
                                     {copy_props * npp},
                                     {1}};
      INTRINSIC(pipe_barrier, PIPE_V);
      copy_ubuf_to_ubuf_1d_core(&from, &to);
    }
    produced += copy_props;

    // 防御:若某次批产出为 0(异常),退出避免死循环
    if (batch_out == 0)
      break;
  }

  // 写出各路最终消耗(= cursor[i]),这是本 op 的关键可观测产物
  for (int i = 0; i < num_lists; ++i)
    *(cons_ptr + i) = (int32_t)cursor[i];
}

// ============================================================================
// C 接口:供 CustomOp 注册
// ============================================================================
extern "C" {

/// 截断模式归并 + 读回各路消耗(float32)
///
/// 注意:本接口遵循 CCE/MLIR custom op 约定,函数返回 void;
///       所有"输出"都经 memref 指针写回,而非 C 返回值。
///       本函数有 2 个输出:dst_proposals(归并结果)与 consumed_out(各路消耗),
///       调用方需预先分配好这两个 buffer 并把指针传入,kernel 直接往里写。
///
/// src_proposals: [输入] tensor<num_lists*list_len*2 x f32> — 多段已排序 proposal
/// tmp_buf:       [工作] tensor<num_lists*list_len*2 x f32> — ping-pong 工作区
/// num_lists:     [输入] 有序段数(2/3/4)
/// list_len:      [输入] 每段 proposal 数
/// topk:          [输入] 截断保留 proposal 数
/// dst_proposals: [输出,指针写回] tensor<topk*2 x f32>  — 归并后前 topk 个 proposal
/// consumed_out:  [输出,指针写回] tensor<num_lists x i32> — 各路实际消耗的 proposal 数(VMS4_SR 累计)
__aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_merge_exhaust_consumed_float(
    memref_t<__ubuf__ float, 1> *src_proposals,
    memref_t<__ubuf__ float, 1> *tmp_buf,
    int64_t num_lists,
    int64_t list_len,
    int64_t topk,
    memref_t<__ubuf__ float, 1> *dst_proposals,
    memref_t<__ubuf__ int32_t, 1> *consumed_out) {
  merge_exhaust_with_consumed(src_proposals, dst_proposals, consumed_out,
                              tmp_buf, num_lists, list_len, topk);
}

/// 单次截断归并(4 路等长)+ 读回消耗 — 最小可观测原语
///
/// 注意:函数返回 void;输出经 memref 指针写回(CCE custom op 约定):
///   - dst_proposals:[输出] 归并后的 proposal(安全前缀在前),由 vmrgsort4 直接写入该 UB;
///   - consumed_out: [输出] VMS4_SR 拆出的 4 路消耗。
/// 仅做一次 vmrgsort4(截断模式),便于单元测试核对 "消耗反馈" 行为。
///
/// src_proposals: [输入] tensor<4*list_len*2 x f32> — 4 段已排序 proposal(等长)
/// list_len:      [输入] 每段 proposal 数
/// dst_proposals: [输出,指针写回] tensor<4*list_len*2 x f32> — 归并输出(安全前缀在前)
/// consumed_out:  [输出,指针写回] tensor<4 x i32> — VMS4_SR 拆出的 4 路消耗
__aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_vmrgsort4_exhaust_once_float(
    memref_t<__ubuf__ float, 1> *src_proposals,
    int64_t list_len,
    memref_t<__ubuf__ float, 1> *dst_proposals,
    memref_t<__ubuf__ int32_t, 1> *consumed_out) {
  constexpr int64_t npp = PROPOSALS_BYTES / sizeof(float); // 2
  auto src_ptr = src_proposals->aligned + src_proposals->offset;
  auto dst_ptr = dst_proposals->aligned + dst_proposals->offset;
  auto cons_ptr = consumed_out->aligned + consumed_out->offset;

  __ubuf__ float *xn[4] = {
      src_ptr + (int64_t)0 * list_len * npp,
      src_ptr + (int64_t)1 * list_len * npp,
      src_ptr + (int64_t)2 * list_len * npp,
      src_ptr + (int64_t)3 * list_len * npp,
  };
  uint32_t lens[4] = {(uint32_t)list_len, (uint32_t)list_len,
                      (uint32_t)list_len, (uint32_t)list_len};

  uint64_t sr = vmrgsort4_exhaust<float>(dst_ptr, xn, lens, 4);

  for (int i = 0; i < 4; ++i)
    *(cons_ptr + i) = (int32_t)vms4_consumed(sr, i);
}

// ============================================================================
// unpack:proposal -> value + index(vreducev2)。供最终输出用。
// ============================================================================
__aiv__ __attribute__((always_inline)) void
unpack_proposals_ex(memref_t<__ubuf__ float, 1> *src,
                    memref_t<__ubuf__ float, 1> *dst_value,
                    memref_t<__ubuf__ int32_t, 1> *dst_index,
                    int64_t real_num) {
  INTRINSIC_NO_ARGS(set_mask_count);
  INTRINSIC(set_vector_mask, 0, real_num);
  // 先取 index(避免被 value 写覆盖)
  memref_t<__ubuf__ int32_t, 1> src_int32;
  view_as<float, int32_t, 1>(src, &src_int32);
  vreducev2_1d_with_pattern_mode<int32_t, PatternMode::INDEX_1_FROM_2_ELEMENTS>(
      &src_int32, dst_index);
  INTRINSIC(pipe_barrier, PIPE_V);
  // 再取 value
  vreducev2_1d_with_pattern_mode<float, PatternMode::INDEX_0_FROM_2_ELEMENTS>(
      src, dst_value);
  INTRINSIC_NO_ARGS(set_mask_norm);
}

/// 流式 topk —— 2 路截断归并取前 K(raw 输出,供下一轮)
///
/// 用于"挑战者 K 个 proposal" + "当前赢家 K 个 proposal" 的合并:
/// 两段各 K 长、各自降序,做一次 2 路截断归并(config[12]=1),
/// 由命题保证一次产出 >= K,故直接取前 K 个 proposal 作为新赢家。
///
/// 注意:函数返回 void,输出经 dst_proposals 指针写回。
///
/// src_proposals: [输入] tensor<2*K*2 x f32> — [挑战者K | 赢家K] 两段已排序 proposal
/// tmp_buf:       [工作] tensor<2*K*2 x f32> — 归并中转
/// topk:          [输入] K
/// dst_proposals: [输出,指针写回] tensor<K*2 x f32> — 前 K 个 proposal(新赢家)
__aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_merge_exhaust_topk_raw_float(
    memref_t<__ubuf__ float, 1> *src_proposals,
    memref_t<__ubuf__ float, 1> *tmp_buf,
    int64_t topk,
    memref_t<__ubuf__ float, 1> *dst_proposals) {
  constexpr int64_t npp = PROPOSALS_BYTES / sizeof(float); // 2
  auto src_ptr = src_proposals->aligned + src_proposals->offset;
  auto tmp_ptr = tmp_buf->aligned + tmp_buf->offset;

  // 2 路:list0 = src[0:K], list1 = src[K:2K],各 K 长
  __ubuf__ float *xn[4] = {
      src_ptr + (int64_t)0 * topk * npp,
      src_ptr + (int64_t)1 * topk * npp,
      src_ptr, src_ptr, // 未用
  };
  uint32_t lens[4] = {(uint32_t)topk, (uint32_t)topk, 0, 0};

  // 截断归并到 tmp(2 路 mask=0x3)
  (void)vmrgsort4_exhaust<float>(tmp_ptr, xn, lens, 2);

  // 取前 K 个 proposal 拷到 dst(命题保证产出 >= K)
  INTRINSIC(pipe_barrier, PIPE_V);
  memref_t<__ubuf__ float, 1> from{tmp_buf->aligned, tmp_buf->allocated,
                                   tmp_buf->offset, {topk * npp}, {1}};
  memref_t<__ubuf__ float, 1> to{dst_proposals->aligned,
                                 dst_proposals->allocated,
                                 dst_proposals->offset, {topk * npp}, {1}};
  copy_ubuf_to_ubuf_1d_core(&from, &to);
}

/// 流式 topk 最终步 —— 2 路截断归并取前 K + unpack 成 value/index
///
/// 注意:函数返回 void,输出经 dst_value / dst_index 指针写回。
///
/// src_proposals: [输入] tensor<2*K*2 x f32> — [挑战者K | 赢家K]
/// tmp_buf:       [工作] tensor<2*K*2 x f32>
/// topk:          [输入] K
/// dst_value:     [输出,指针写回] tensor<K x f32>
/// dst_index:     [输出,指针写回] tensor<K x i32>
__aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_merge_exhaust_topk_float(
    memref_t<__ubuf__ float, 1> *src_proposals,
    memref_t<__ubuf__ float, 1> *tmp_buf,
    int64_t topk,
    memref_t<__ubuf__ float, 1> *dst_value,
    memref_t<__ubuf__ int32_t, 1> *dst_index) {
  constexpr int64_t npp = PROPOSALS_BYTES / sizeof(float); // 2
  auto src_ptr = src_proposals->aligned + src_proposals->offset;
  auto tmp_ptr = tmp_buf->aligned + tmp_buf->offset;

  __ubuf__ float *xn[4] = {
      src_ptr + (int64_t)0 * topk * npp,
      src_ptr + (int64_t)1 * topk * npp,
      src_ptr, src_ptr,
  };
  uint32_t lens[4] = {(uint32_t)topk, (uint32_t)topk, 0, 0};

  (void)vmrgsort4_exhaust<float>(tmp_ptr, xn, lens, 2);

  // 取前 K 个 proposal 做 unpack
  INTRINSIC(pipe_barrier, PIPE_V);
  memref_t<__ubuf__ float, 1> merged{tmp_buf->aligned, tmp_buf->allocated,
                                     tmp_buf->offset, {topk * npp}, {1}};
  memref_t<__ubuf__ float, 1> dval{dst_value->aligned, dst_value->allocated,
                                   dst_value->offset, {topk}, {1}};
  memref_t<__ubuf__ int32_t, 1> didx{dst_index->aligned, dst_index->allocated,
                                     dst_index->offset, {topk}, {1}};
  unpack_proposals_ex(&merged, &dval, &didx, topk);
}

/// 单次截断归并(Python 驱动版)——最通用的最小原语
///
/// 由 Python 侧完全控制:哪几路、每路在 src 中的起始 proposal 偏移、每路长度、
/// 输出 buffer。C++ 只做一次 vmrgsort4(截断模式)并回报各路消耗。
/// 多级归并的循环、空间分配、指针推进全部在 Python 侧用此原语拼出。
///
/// 注意:函数返回 void,输出经 dst_proposals / consumed_out 指针写回。
///
/// src_proposals: [输入] UB proposal 缓冲(各路都在其中,起始由 off* 给出)
/// ways:          [输入] 有效路数(2/3/4)
/// off0..off3:    [输入] 各路起始 proposal 偏移(以 proposal 计,未用路填 0)
/// len0..len3:    [输入] 各路长度 proposal 数(未用路填 0)
/// dst_proposals: [输出,指针写回] 归并输出(安全前缀在前)
/// consumed_out:  [输出,指针写回] tensor<4 x i32> — 各路本次消耗(VMS4_SR 拆解)
__aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_vmrgsort4_exhaust_step_float(
    memref_t<__ubuf__ float, 1> *src_proposals,
    int64_t ways,
    int64_t off0, int64_t off1, int64_t off2, int64_t off3,
    int64_t len0, int64_t len1, int64_t len2, int64_t len3,
    memref_t<__ubuf__ float, 1> *dst_proposals,
    memref_t<__ubuf__ int32_t, 1> *consumed_out) {
  constexpr int64_t npp = PROPOSALS_BYTES / sizeof(float); // 2
  auto src_ptr = src_proposals->aligned + src_proposals->offset;
  auto dst_ptr = dst_proposals->aligned + dst_proposals->offset;
  auto cons_ptr = consumed_out->aligned + consumed_out->offset;

  int64_t offs[4] = {off0, off1, off2, off3};
  int64_t lns[4] = {len0, len1, len2, len3};

  // ★ 关键:压紧(compact)非空路到连续车道 0..(active-1)。
  //   vmrgsort4_exhaust 用 mask = (1<<ways)-1,假设活跃路从车道 0 连续排布。
  //   但 caller 把 4 路放在固定槽位、用长度 0 表示空路,归并尾段可能出现
  //   "中间某路先耗尽"(如 l0>0,l1=0,l2>0)的空洞。若直接按固定槽位送,
  //   mask 会选中长度为 0 的车道 → vmrgsort4 非法配置(VEC illegal config)。
  //   故这里只把 len>0 的路收集进连续车道,记录其原始下标,
  //   归并后再把各路消耗 scatter 回原始槽位。
  __ubuf__ float *xn[4];
  uint32_t lens[4] = {0, 0, 0, 0};
  int orig_idx[4] = {0, 0, 0, 0};   // 压紧车道 w → 原始槽位
  int active = 0;
  for (int i = 0; i < 4; ++i) {
    if (lns[i] > 0) {
      xn[active] = src_ptr + offs[i] * npp;
      lens[active] = (uint32_t)lns[i];
      orig_idx[active] = i;
      active++;
    }
  }

  // 先把 4 路消耗清 0(空路消耗恒为 0)
  for (int i = 0; i < 4; ++i)
    *(cons_ptr + i) = 0;

  if (active == 0) {
    return;  // 无数据,空操作
  }

  if (active == 1) {
    // 只剩一路:无需归并,直接 copy 残余到 dst(标准 k 路归并尾段优化)。
    // 注意:vmrgsort4 不支持 1 路(mask=0x1 非法),必须单独处理。
    int64_t cnt = lens[0];
    memref_t<__ubuf__ float, 1> from{
        src_proposals->aligned, src_proposals->allocated,
        src_proposals->offset + offs[orig_idx[0]] * npp,
        {cnt * npp}, {1}};
    memref_t<__ubuf__ float, 1> to{
        dst_proposals->aligned, dst_proposals->allocated,
        dst_proposals->offset, {cnt * npp}, {1}};
    INTRINSIC(pipe_barrier, PIPE_V);
    copy_ubuf_to_ubuf_1d_core(&from, &to);
    *(cons_ptr + orig_idx[0]) = (int32_t)cnt;
    return;
  }

  // active >= 2:正常截断归并(压紧后车道连续,mask 合法)
  uint64_t sr = vmrgsort4_exhaust<float>(dst_ptr, xn, lens, active);

  // 把压紧车道 w 的消耗 scatter 回原始槽位 orig_idx[w]
  for (int w = 0; w < active; ++w)
    *(cons_ptr + orig_idx[w]) = (int32_t)vms4_consumed(sr, w);
}

/// unpack:前 topk 个 proposal → value/index(供最终输出)
///
/// 注意:函数返回 void,输出经 dst_value / dst_index 指针写回。
/// src_proposals: [输入] 前 topk 个有序 proposal
/// topk:          [输入] K
/// dst_value:     [输出,指针写回] tensor<topk x f32>
/// dst_index:     [输出,指针写回] tensor<topk x i32>
__aiv__ __attribute__((always_inline)) void
_mlir_ciface_custom_unpack_topk_float(
    memref_t<__ubuf__ float, 1> *src_proposals,
    int64_t topk,
    memref_t<__ubuf__ float, 1> *dst_value,
    memref_t<__ubuf__ int32_t, 1> *dst_index) {
  constexpr int64_t npp = PROPOSALS_BYTES / sizeof(float); // 2
  // CRITICAL: vreducev2_1d_with_pattern_mode uses src->sizes[0] to set mask,
  // NOT the topk parameter. If src buffer is larger than topk*npp (which it
  // always is — it's the full UNPACK_CHUNK*2 buffer), vreducev2 will process
  // garbage past the valid data and may trigger "illegal configurations".
  // Solution: create a src view limited to exactly topk*npp elements.
  memref_t<__ubuf__ float, 1> src_view{
      src_proposals->aligned, src_proposals->allocated, src_proposals->offset,
      {topk * npp}, {1}};
  memref_t<__ubuf__ float, 1> dval{dst_value->aligned, dst_value->allocated,
                                   dst_value->offset, {topk}, {1}};
  memref_t<__ubuf__ int32_t, 1> didx{dst_index->aligned, dst_index->allocated,
                                     dst_index->offset, {topk}, {1}};
  unpack_proposals_ex(&src_view, &dval, &didx, topk);
}

} // extern "C"
# Triton TopK 当前实现

本文只描述当前正式实现，不包含已删除的实验分支。

源码：

```text
/Users/rudy/codex/topk/workspace/tutorials/topk-triton.py
```

## 接口

```python
topk(x: torch.Tensor, k: int, seg_len: int = 4096, debug: bool = False)
```

| 项 | 说明 |
| --- | --- |
| `x` | `(B, N)`，`torch.float32`，NPU tensor |
| `k` | 每行取 top-k，要求 `0 < k <= N` |
| `seg_len` | 默认 `4096`，主路径按 4096 元素切 segment |
| 返回值 | `(values, indices)` |
| `values` | `(B, K)`，降序排列 |
| `indices` | `(B, K)`，每个 value 在原 row 内的列下标 |

当前实现固定输出降序；没有实现升序分支。

## 符号

| 符号 | 含义 |
| --- | --- |
| `B` | batch size |
| `N` | 每行长度 |
| `K` | top-k |
| `S` | `ceil(N / 4096)` |
| `S2048` | `ceil(N / 2048)`，只用于 2048 小 N 路径 |
| `sort_run_len` | `min(K, 4096)` |
| `out_segs` | batch-local 阶段每行输出的 local run 数 |
| proposal | `[value(fp32), index(int32 as fp32)]`，占 8B |

关键常量：

| 常量 | 值 | 对应参数 | 说明 |
| --- | ---: | --- | --- |
| `NUM_CORES` | `48` | grid / `out_segs` | 目标 vector core 数；sort grid 最多 48，`out_segs` 选择也按 48 个 core 的占用来打分。 |
| `MAX_WAYS` | `4` | merge ways | 单次归并最多 4 路；影响 `vmrgsort4_exhaust_step`、`final_merge_unpack_kernel`、`generic_merge_unpack_kernel` 和 core-local 内部归并树。 |
| `CHUNK` | `2048` | merge chunk length | 通用/final merge 每路一次最多读入的 proposal 数；对应 kernel 参数 `CHUNK_C`。 |
| `CORE_LOCAL_CHUNK` | `1152` | core-local chunk length | `core_local_nocopy_merge_kernel` 每路一次最多处理的 proposal 数；对应 kernel 参数 `CHUNK_C`；当前默认值来自 batch=48、K=8192 场景的稳定配置。 |
| `TINY_S_SMALL_BATCH_MAX_K` | `512` | `K` | tiny-S small-batch 子场景阈值；`K<=512` 时可走 `SEG_LEN=2048 + static_merge4_unpack_kernel`。 |
| `STREAM16_SMALLK_MERGE_MAX_K` | `2048` | `K` | small-K stream 阈值；`K<=2048` 且 `sort_run_len==K`、`S>16` 时进入 small-K stream 判断。 |
| `DIRECT_GROUP_MIN_SEGS` | `16` | `S=num_seg_per_row` | no core-local 与 batch-local 的段数分界下界；`S<16` 更倾向直接 sort 后 final，`S>16` 才进入 large-K direct final 的额外判断。 |
| `DIRECT_GROUP_MAX_SEGS` | `128` | `S=num_seg_per_row` | no core-local 与小 batch `out_segs=4` 特化上界；`S>128` 不再走 large-K direct final 判断，且小 batch 不再强制 `out_segs<=4`。 |

## 路径总览

`topk()` 的主路径按 K 大小和是否使用 core-local 划分。

| 优先级 | 路径 | 触发条件 | 执行 kernel |
| ---: | --- | --- | --- |
| 1 | small-K direct final | `K<=2048`，但不启用 small-K batch-local-stream + final merge；包括 tiny-S、小 N、小 S base 等直接 final 场景 | `sort_kernel -> final output kernel` |
| 2 | small-K batch-local-stream + final merge | `K<=2048`，启用 small-K batch-local-stream + final merge stream | `sort_kernel -> smallk_core_local_stream_merge_kernel -> smallk stream final` |
| 3 | large-K direct final | `K>2048`，sort 后直接 final 输出；包括小 N 2048、`out_segs==S` 和其他 base 情况 | `sort_kernel -> final merge unpack` |
| 4 | large-K batch-local + final merge | `K>2048`，先局部归并降 run 数，再 final 输出 | `sort_kernel -> core_local_nocopy_merge_kernel -> final output` |

## Sort 规则

主路径使用 `sort_kernel`：

```text
grid = min(B*S, 48)
program cid 处理 segment cid, cid+48, cid+96 ...
```

每个 4096 segment 最多写回 `sort_run_len=min(K,4096)` 个 proposal。

`sort_kernel` 根据 `sort_run_len` 选择 custom sort：

| sort impl | 条件 | custom op |
| --- | --- | --- |
| `layered` | `seg_len=4096`，`K<=128` 或 `K=2048`；当单 program 处理超过 2 个 segment 时，只保留 `K<=128` | `sort_1d_topk_proposals_layered_4096` |
| `smallk` | `seg_len=4096`，`129<=K<=512` | `sort_1d_topk_proposals_4x1024` |
| `base` | 其他 | `sort_1d_topk_proposals` |

no core-local final 是所有 `sort_kernel -> final merge unpack` 场景的合并说明。tiny-S small-batch 子场景使用 `SEG_LEN=2048` 和 `static_merge4_unpack_kernel`；small-N 子场景使用 `SEG_LEN=2048`；其他子场景使用传入的 `seg_len`。除 tiny-S small-batch 外，当 final run 数 `<=4` 时使用 `final_merge_unpack_kernel`，连续布局下传 `RUN_STRIDE=RUN_LEN`；当 final run 数 `>4` 时使用 `generic_merge_unpack_kernel`。

## out_segs 规则

`out_segs` 只用于 batch-local 和 small-K batch-local-stream + final merge。

```text
max_runs = min(48, S)

if max_runs <= 1:
    out_segs = max_runs
elif B == 2:
    out_segs = min(16, max_runs)
elif B <= 8 and S <= 128 and K <= 4096:
    out_segs = min(4, max_runs)
elif B == 16 and S == 16 and K == 4096:
    out_segs = min(4, max_runs)
else:
    在 1..max_runs 中选择 runs，使：
      waves = ceil(B * runs / 48) 最小
      idle_tail = (-B * runs) % 48 最小
      runs 本身最小
```

目标是让 `B*out_segs` 贴近 48 的整数倍，同时限制 local run 数，避免后续 GM 中间结果过大。

## Kernel 作用

| kernel | 作用 | 主要路径 |
| --- | --- | --- |
| `sort_kernel` | round-robin sort；grid 为 `min(真实 segment 数,48)`，一个 program 处理一个或多个 segment，每段最多输出 `min(K,SEG_LEN)` 个 proposal。 | small-K direct final、small-K batch-local-stream + final merge、large-K direct final、large-K batch-local + final merge |
| `generic_merge_unpack_kernel` | 通用 4-way 多轮归并，最后 unpack 到 `values/indices`。 | no core-local 中 final run 数 `>4` 的场景、large-K batch-local + final merge 的 generic final |
| `static_merge4_unpack_kernel` | 最多 4 个 run 的静态归并，直接输出。 | small-K direct final 的 tiny-S small-batch 子场景 |
| `smallk_core_local_stream_merge_kernel` | small-K batch-local-stream + final merge 流式归并；每个 `(row,cid)` 归并一个连续 segment range，输出一个 K-length local run。 | small-K batch-local-stream + final merge |
| `smallk_stream_merge4_unpack_kernel` | small-K final，最多 4 个 K-length local run 流式归并并输出。 | small-K batch-local-stream + final merge final |
| `smallk_stream_merge_unpack_kernel` | small-K final，N-way rolling stream 归并并输出。 | small-K direct final、small-K batch-local-stream + final merge final |
| `core_local_nocopy_merge_kernel` | batch-local 归并；第一轮直接读 `sort_gm`，省掉 SortGM 到 TmpGM 的预拷贝。 | large-K batch-local + final merge |
| `final_merge_unpack_kernel` | 归并最多 4 个 run 并 unpack 输出；`RUN_STRIDE=RUN_LEN` 时读取连续 run，`RUN_STRIDE>RUN_LEN` 时读取 strided tmp run。 | no core-local 中 final run 数 `<=4` 的场景、large-K batch-local + final merge direct final |
| `unpack_kernel` | core-local 已产出单个有序 run 时，只做 proposal 拆包，不再做 merge。 | large-K batch-local + final merge direct final |

## Merge 细节

当前实现里 merge 可以按执行方式分成三类：小 K 流式归并、大 K 分阶段归并、特化路径归并。三类输入都是 `sort_kernel` 产出的有序 proposal run，proposal 格式为 `[value, index_as_f32]`。

### 小 K 流式归并

小 K 流式归并服务于 `K<=2048` 且 `sort_run_len==K` 的场景。因为每个 sort run 已经被截断到 K，后续每次 merge 只需要保留 K 个 proposal，不需要产出完整段长。

run 长度：

| 阶段 | run 来源 | 每个 run 的长度 |
| --- | --- | --- |
| sort 输出 | `sort_kernel` 写入 `sort_gm` 的每个 segment run | `run_len = sort_run_len = K` 个 proposal |
| direct final 输入 | `smallk_stream_merge_unpack_kernel` / `smallk_stream_merge4_unpack_kernel` 直接读取 `sort_gm` | 每路 `K` 个 proposal |
| batch-local 输入 | `smallk_core_local_stream_merge_kernel` 读取一个 local range 内的 sort run | 每路 `K` 个 proposal |
| batch-local 输出 | 每个 `(row,cid)` 写入 `local_gm` 的 local run | `K` 个 proposal |
| batch-local final 输入 | final kernel 读取 `out_segs` 个 local run | 每路 `K` 个 proposal |

因此小 K 流式路径的所有 merge 输入 run 都按 K-length 处理；`stream_cap = next_power_of_2(4 * K * 2)`，其中 `*2` 是 proposal 的 `[value,index]` 两个 f32 槽位。

| 子路径 | 触发条件 | kernel |
| --- | --- | --- |
| direct final | `K<=2048`，`num_runs>1`，并且不进入 batch-local；典型是 `B` 较大或 `out_segs==1` | `smallk_stream_merge_unpack_kernel` 或 `smallk_stream_merge4_unpack_kernel` |
| batch-local stream | `K<=2048`，`B<=24`，`num_runs>16`，`out_segs>1` | `smallk_core_local_stream_merge_kernel -> smallk_stream_merge*_unpack_kernel` |

核心归并方式是 rolling accumulator：

```text
前 4 个 run 做 4-way merge，产出 K 个 accumulator
后续每次读取最多 3 个新 run
accumulator + 3 个新 run 再做 4-way merge
重复直到所有 run 被消费
最后把 accumulator unpack 成 values/indices
```

`smallk_stream_merge_unpack_kernel` 是每个 row 一个 program。它先从 `WorkGM` 读取前 4 个 run 到 UB，用 `vmrgsort4_exhaust_step` 产出 accumulator；后续每轮把新 3 个 run 追加到 accumulator 后面，再做一次 4-way merge。中间 accumulator 在 `buf_a/buf_b` 之间切换，不写回 GM，最后在同一个 kernel 内调用 `unpack_topk_float` 并写最终输出。

小 K 调用归并指令时，UB buffer 布局是连续的 K-length proposal run。第一次归并把前 4 个 run 放到输入 buffer：

```text
run0: offset = 0
run1: offset = K
run2: offset = 2*K
run3: offset = 3*K
len0..len3 = K 或 0
```

调用形式等价于：

```text
vmrgsort4_exhaust_step(
  src_buf,
  ways=4,
  off0=0, off1=K, off2=2*K, off3=3*K,
  len0, len1, len2, len3,
  dst_buf,
  consumed_out
)
```

第一次输出的 `dst_buf` 就是 K-length accumulator。后续每轮把 accumulator 放在 offset 0，再把最多 3 个新 run 搬到 `K、2*K、3*K` 位置，因此仍然是同一个 4-way 指令：

```text
accumulator: offset = 0,   len = K
new run 0 : offset = K,   len = K 或 0
new run 1 : offset = 2*K, len = K 或 0
new run 2 : offset = 3*K, len = K 或 0
```

`buf_a` 和 `buf_b` 轮换承担 `src_buf/dst_buf`，所以小 K direct final 的归并中间结果不写回 GM。`consumed_out` 在这条路径里主要由 intrinsic 产生，但由于每次只需要保留 K 个 accumulator，kernel 不依赖它推进多路 cursor。

`smallk_stream_merge4_unpack_kernel` 是最多 4 个 run 的简化版。它一次性读取最多 4 个 K-length run，做一次 4-way merge，然后直接 unpack 输出；没有多轮 accumulator。

`smallk_core_local_stream_merge_kernel` 用在小 K 的 batch-local 场景。grid 是 `(B, out_segs)`，每个 program 处理一个 row 的一个连续 segment range。它内部也使用 `4 + 3 + 3 + ...` 的 rolling accumulator，但每个 program 最终只把一个 K-length local run 写入 `local_gm`。随后 final kernel 再把 `out_segs` 个 local run 合成最终 topK；当 `out_segs<=4` 时走 `smallk_stream_merge4_unpack_kernel`，否则走 `smallk_stream_merge_unpack_kernel`。

这类路径的主要收益是减少中间 GM 读写：direct final 没有 merge 中间写回；batch-local 只有 local run 这一层必须写回，local range 内部的多轮 merge 不反复写 GM。

### 大 K 归并

大 K 指 `K>2048` 或未命中小 K stream 的 final 场景。大 K 下单个 accumulator 不能总是完整放进 UB，因此主要使用 chunked 4-way merge：每路每次最多读取 `CHUNK` 或 `CORE_LOCAL_CHUNK` 个 proposal，merge 后把本批结果写回 GM，再继续下一批。

no core-local 的大 K direct final：

```text
sort_kernel
  -> final_merge_unpack_kernel     # final run 数 <= 4
     或 generic_merge_unpack_kernel # final run 数 > 4
```

`final_merge_unpack_kernel` 处理最多 4 个 run。它为每路维护 cursor，每轮从 GM 读入最多 `CHUNK=2048` 个 proposal 到 UB，调用 `vmrgsort4_exhaust_step`，根据 `consumed` 更新每路 cursor，再把本轮输出写到 WorkGM 的另一个 phase。循环结束后，它在同一个 kernel 内从最终 phase 读取 proposal，调用 `unpack_topk_float` 写 `values/indices`。

大 K 4-way merge 的 UB 分配如下。这里的 `CHUNK=2048` 是每路 proposal 数；一个 proposal 占两个 f32 槽位 `[value,index_as_f32]`。

```text
in_buf  = 4 * CHUNK * 2 f32
out_buf = 4 * CHUNK * 2 f32 + 8 f32
cons    = 4 int32
```

以 `CHUNK=2048` 计算：

```text
in_buf  = 4 * 2048 * 2 f32 = 16384 f32 = 64 KB
out_buf = 4 * 2048 * 2 f32 = 16384 f32 = 64 KB
合计约 128 KB，再加少量 cons/cursor/scalar 临时量
```

因此确实存在一次读入 `4 * 2048` 个 proposal 的 case，例如 `(16,16384), K=8192` 的 large-K direct final，或者 batch-local final 中 `out_segs=4` 且 `core_out_len>=2048` 的 case。当前普通大 K merge 不会因为这一步溢出 UB，因为它只同时保留一组 4-way 输入窗口和一个输出窗口；每批结果立即写回 GM。会带来 UB 风险的是同时保留更多 group buffer/final buffer 的融合式多层归并。

大 K 调用归并指令时，`in_buf` 里只放每路当前窗口，而不是完整 run。每路 cursor 分别是 `c0..c3`，本轮剩余量是 `r0..r3`，实际搬入长度是：

```text
l0 = min(r0, CHUNK)
l1 = min(r1, CHUNK)
l2 = min(r2, CHUNK)
l3 = min(r3, CHUNK)
```

4 路窗口在 `in_buf` 中的布局是：

```text
run0 window: offset = 0 * CHUNK
run1 window: offset = 1 * CHUNK
run2 window: offset = 2 * CHUNK
run3 window: offset = 3 * CHUNK
len0..len3 = l0..l3
```

调用形式等价于：

```text
vmrgsort4_exhaust_step(
  in_buf,
  ways=aw,
  off0=0*CHUNK, off1=1*CHUNK, off2=2*CHUNK, off3=3*CHUNK,
  l0, l1, l2, l3,
  out_buf,
  consumed_out
)
```

其中 `aw` 是本轮非空输入路数。`consumed_out` 返回 4 路本次实际消耗量：

```text
c0 += consumed_out[0]
c1 += consumed_out[1]
c2 += consumed_out[2]
c3 += consumed_out[3]
```

`out_buf` 中得到的是本轮已经有序的 proposal batch，kernel 取不超过当前 group 剩余需求的 `take` 个 proposal 写回 GM。下一轮再根据更新后的 cursor 继续从 GM 读下一批窗口。因此大 K 路径是“分块读入、单次 4-way merge、写回 GM、推进 cursor”的循环。

`generic_merge_unpack_kernel` 处理超过 4 个 run。它按 4 路一组做多轮归并树：

```text
cur_n = S
while cur_n > 1:
  每 4 个 run 归并成 1 个 run
  每个新 run 长度截断到 K
  WorkGM phase 0/1 ping-pong
  cur_n = ceil(cur_n / 4)
最后 unpack 输出
```

这里的每个 group 内部仍然是 chunked 4-way merge，不会一次把完整 run 放进 UB。由于每一轮归并树都会把新 run 写回 WorkGM，再由下一轮读回，所以它比小 K 流式路径有更多 GM 中间读写，但 UB 占用稳定，适合大 K 或 run 数较多的通用情况。

batch-local 的大 K 路径：

```text
sort_kernel
  -> core_local_nocopy_merge_kernel[(B, out_segs)]
  -> final kernel
```

`core_local_nocopy_merge_kernel` 先把每个 row 切成 `out_segs` 个连续 local range。每个 `(row,cid)` program 负责一个 local range，把其中多个 sort run 归并成一个 local run，local run 长度为：

```text
core_out_len = min(ceil(S / out_segs) * sort_run_len, K)
```

它的第一轮直接从 `sort_gm` 读取原始 sort run，写到 `TmpGM` phase 1，因此省掉原先额外的 `SortGM -> TmpGM` 预拷贝。后续轮次在 `TmpGM` phase 0/1 之间 ping-pong。每一轮仍按最多 4-way 分组，单组输出截断到 K。

batch-local final 有三种收尾：

| 条件 | 收尾方式 | 说明 |
| --- | --- | --- |
| `out_segs==1` 且 `core_out_len==K` | `unpack_kernel` | core-local 已经得到最终单 run，只需拆 proposal |
| `2<=out_segs<=4` 且 local run 长度一致 | `final_merge_unpack_kernel` | 直接把最多 4 个 local run final merge 并 unpack |
| 其他 | `generic_merge_unpack_kernel` | 对 `out_segs` 个 local run 再走通用 4-way 归并树 |

### 特化路径归并

当前保留的特化归并是 tiny-S 静态 4 路归并。它把 GM 读取、归并、unpack、写最终输出合在一个 custom op 调用里，目标是减少通用路径的 kernel 固定开销和中间 GM ping-pong。

tiny-S 静态 4 路归并：

```text
触发：B<=4，K<=512，S<=2，按 2048 切分后 S2048<=4
路径：sort_kernel(seg_len=2048) -> static_merge4_unpack_kernel
custom op：merge4_static_gm_to_output_float
```

`static_merge4_unpack_kernel` 每个 row 一个 program。它把最多 4 个 run 的 block pointer 传给 `merge4_static_gm_to_output_float`，custom op 在内部完成 GM->UB 读取、最多 4-way merge、proposal unpack，并直接写最终 `values/indices`。这条路径不走 `generic_merge_unpack_kernel`，也不使用 WorkGM ping-pong。


## Custom Op 依赖

默认 bitcode 目录：

```text
TOPK_BC_DIR=/home/topk/topk_import_check/custom-topk
```

可选覆盖：

| 环境变量 | 作用 |
| --- | --- |
| `TOPK_BC_DIR` | base custom op `.bc` 目录 |
| `TOPK_SMALLK_BC_DIR` | `sort_topk_smallk.bc` 目录；未设置时优先使用 `/home/topk/topk/custom-topk`，否则回退到 `TOPK_BC_DIR` |
| `TOPK_LAYERED_BC_DIR` | `sort_topk_layered.bc` 目录；未设置时优先使用 `/home/topk/topk/custom-topk`，否则回退到 `TOPK_BC_DIR` |

| custom op | bitcode | 输入 | 输出 | 详细作用 | 调用位置 |
| --- | --- | --- | --- | --- | --- |
| `sort_1d_topk_proposals` | `sort_topk.bc` | 一个 UB 内的原始 value segment，长度通常为 `4096`，small-N 路径可为 `2048`；参数包含 `topk` 和 `index_offset`。 | topK proposal run，格式为 `[value, index_as_f32]` 交错存储。 | base sort。它负责从无序 value 生成带原始列下标的有序 proposal，因此不是简单 merge。适合 K 较大或 small-K 特化收益不明显的场景。 | `sort_kernel` 的 base sort 分支 |
| `sort_1d_topk_proposals_4x1024` | `sort_topk_smallk.bc` | 一个 `4096` value segment；内部切成 `4 x 1024` 子块。 | 一个长度为 K 的 proposal run。 | small-K sort。每个 1024 子块先各自产生 K 个候选 proposal，再做 4-way merge，只保留最终 topK。减少 `129<=K<=512` 时对完整 4096 排序的无效工作。 | `sort_kernel` 的 smallk sort 分支 |
| `sort_1d_topk_proposals_layered_4096` | `sort_topk_layered.bc` | 一个 `4096` value segment；参数包含 `topk` 和 `index_offset`。 | 一个长度不超过 `min(K,2048)` 的 proposal run。 | layered sort。先构造多个较小有序 run，再做截断式分层归并，目标是在 `K<=128` 和部分 `K=2048` 场景减少完整排序开销。 | `sort_kernel` 的 layered sort 分支 |
| `vmrgsort4_exhaust_step` | `merge-sort-exhaust-1.bc` | UB 内最多 4 个已经有序的 proposal run，以及每路 offset/len。 | 合并后的 proposal buffer，以及每路本次消耗量 `consumed_out`。 | 4-way merge 基元。它不接收原始 value，也不生成 index；只能归并已经排好序的 proposal run。通用 merge、small-K stream、core-local merge 都用它做每一步归并。 | `generic_merge_unpack_kernel`、`smallk_*merge*`、`core_local_nocopy_merge_kernel`、`final_merge_unpack_kernel` |
| `unpack_topk_float` | `merge-sort-exhaust-1.bc` | UB 内 proposal run，格式为 `[value, index_as_f32]`。 | 独立的 `values` 向量和 `indices` 向量。 | proposal unpack。把中间 proposal 表示拆成最终 `torch.topk` 兼容输出，其中 index 恢复为 int32。 | 所有 `*_unpack_kernel` 的最终输出阶段 |
| `gm_to_ub_copy_float` | `gm-to-ub.bc` | GM block pointer，整块长度由 block shape 决定。 | UB buffer。 | 定长 GM 到 UB 搬运。主要用于满 segment 或满块 proposal 读取。 | sort 输入读取、部分 merge 输入读取 |
| `gm_to_ub_copy_n_float` | `gm-to-ub-n.bc` | GM block pointer、目标 UB offset、运行时 `copy_len`。 | UB buffer。 | 变长 GM 到 UB 搬运。用于尾段、最后一个 chunk、chunked merge 里不足满块的读取，避免按 block shape 越界读。 | sort 尾段读取、merge streaming 读取、unpack 读取 |
| `ub_to_gm_copy_float` | `ub-to-gm.bc` | UB buffer、源 offset、`copy_len`、GM block pointer。 | 写回 GM。 | UB 到 GM 搬运。用于 sort proposal 写回、通用 merge 中间结果写回、core-local local run 写回。 | sort 写 `sort_gm`，通用/core-local merge 写 tmp/final GM |
| `merge4_static_gm_to_output_float` | `merge4-static-gm-to-output.bc` | GM 中最多 4 个有序 proposal run，以及 `run_len/topk/num_runs`。 | 直接写最终 `values/indices`。 | 小段静态归并输出。把 GM 读取、最多 4-run merge、unpack、写 output 合在一个 custom op 内，避免通用 merge kernel 的多轮 GM ping-pong。 | `static_merge4_unpack_kernel` |

其中 `sort_topk*.bc` 负责“无序 value -> 有序 proposal run”；`merge-sort-exhaust-1.bc` 负责“有序 proposal run -> 归并/拆包”。因此 `vmrgsort4_exhaust_step` 不能单独替代 sort custom op，只能作为 sort 后各级 merge 的基础操作。

## 性能数据

数据来源：

```text
/Users/rudy/codex/topk/reports/topk/triton_compare_benchmark/20260630_163157_28cases/triton_benchmark.json
```

配置：

| 项 | 值 |
| --- | --- |
| dtype | `fp32` |
| sorted | `true` |
| `seg_len` | `4096` |
| warmup | `2` |
| active | `5` |

| shape | K | S | 路径 | torch_npu(us) | Triton(us) | 加速比 | 精度 | Triton kernels |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| `(1,8192)` | `2048` | `2` | small-K direct final | `13.164` | `10.108` | `1.302x` | PASS | `final_merge_unpack_kernel=5.276, sort_kernel=4.832` |
| `(1,16384)` | `2048` | `4` | small-K direct final | `14.381` | `13.308` | `1.081x` | PASS | `final_merge_unpack_kernel=5.068, sort_kernel=8.240` |
| `(1,65536)` | `4096` | `16` | large-K batch-local + final merge | `37.853` | `17.697` | `2.139x` | PASS | `core_local_nocopy_merge_kernel=4.076, final_merge_unpack_kernel=5.436, sort_kernel=8.184` |
| `(2,8192)` | `64` | `2` | small-K direct final | `7.440` | `6.420` | `1.159x` | PASS | `sort_kernel=4.492, static_merge4_unpack_kernel=1.928` |
| `(2,131072)` | `64` | `32` | small-K batch-local-stream + final merge | `18.432` | `17.737` | `1.039x` | PASS | `smallk_core_local_stream_merge_kernel=2.296, smallk_stream_merge_unpack_kernel=2.924, sort_kernel=12.517` |
| `(2,262144)` | `32` | `64` | small-K batch-local-stream + final merge | `28.560` | `22.345` | `1.278x` | PASS | `smallk_core_local_stream_merge_kernel=2.460, smallk_stream_merge_unpack_kernel=2.872, sort_kernel=17.013` |
| `(4,8192)` | `128` | `2` | small-K direct final | `8.140` | `6.812` | `1.195x` | PASS | `sort_kernel=4.568, static_merge4_unpack_kernel=2.244` |
| `(4,8192)` | `512` | `2` | small-K direct final | `8.368` | `7.384` | `1.133x` | PASS | `sort_kernel=4.596, static_merge4_unpack_kernel=2.788` |
| `(4,8192)` | `1024` | `2` | small-K direct final | `9.164` | `8.308` | `1.103x` | PASS | `final_merge_unpack_kernel=3.784, sort_kernel=4.524` |
| `(4,131072)` | `128` | `32` | small-K batch-local-stream + final merge | `28.572` | `22.597` | `1.264x` | PASS | `smallk_core_local_stream_merge_kernel=2.760, smallk_stream_merge4_unpack_kernel=2.360, sort_kernel=17.477` |
| `(4,131072)` | `4096` | `32` | large-K batch-local + final merge | `43.113` | `38.944` | `1.107x` | PASS | `core_local_nocopy_merge_kernel=10.540, final_merge_unpack_kernel=6.084, sort_kernel=22.320` |
| `(8,65536)` | `4096` | `16` | large-K batch-local + final merge | `41.581` | `33.393` | `1.245x` | PASS | `core_local_nocopy_merge_kernel=4.428, final_merge_unpack_kernel=6.516, sort_kernel=22.448` |
| `(8,524288)` | `1024` | `128` | small-K batch-local-stream + final merge | `152.167` | `156.203` | `0.974x` | PASS | `smallk_core_local_stream_merge_kernel=10.076, smallk_stream_merge4_unpack_kernel=3.540, sort_kernel=142.587` |
| `(16,8192)` | `64` | `2` | small-K direct final | `12.572` | `9.068` | `1.386x` | PASS | `smallk_stream_merge4_unpack_kernel=1.888, sort_kernel=7.180` |
| `(16,8192)` | `128` | `2` | small-K direct final | `12.840` | `9.352` | `1.373x` | PASS | `smallk_stream_merge4_unpack_kernel=2.132, sort_kernel=7.220` |
| `(16,16384)` | `256` | `4` | small-K direct final | `18.596` | `16.961` | `1.096x` | PASS | `smallk_stream_merge4_unpack_kernel=2.192, sort_kernel=14.769` |
| `(16,16384)` | `8192` | `4` | large-K direct final | `30.773` | `26.185` | `1.175x` | PASS | `final_merge_unpack_kernel=10.396, sort_kernel=15.788` |
| `(16,65536)` | `256` | `16` | small-K direct final | `44.221` | `41.913` | `1.055x` | PASS | `smallk_stream_merge_unpack_kernel=3.412, sort_kernel=38.501` |
| `(16,65536)` | `4096` | `16` | large-K batch-local + final merge | `55.869` | `57.681` | `0.969x` | PASS | `core_local_nocopy_merge_kernel=7.324, final_merge_unpack_kernel=8.084, sort_kernel=42.273` |
| `(16,131072)` | `1024` | `32` | small-K batch-local-stream + final merge | `84.129` | `81.710` | `1.030x` | PASS | `smallk_core_local_stream_merge_kernel=6.112, smallk_stream_merge4_unpack_kernel=3.112, sort_kernel=72.486` |
| `(16,131072)` | `4096` | `32` | large-K batch-local + final merge | `95.038` | `95.862` | `0.991x` | PASS | `core_local_nocopy_merge_kernel=15.788, final_merge_unpack_kernel=6.108, sort_kernel=73.966` |
| `(16,524288)` | `128` | `128` | small-K batch-local-stream + final merge | `282.446` | `218.688` | `1.292x` | PASS | `smallk_core_local_stream_merge_kernel=6.352, smallk_stream_merge4_unpack_kernel=2.248, sort_kernel=210.088` |
| `(32,8192)` | `512` | `2` | small-K direct final | `19.829` | `17.892` | `1.108x` | PASS | `smallk_stream_merge4_unpack_kernel=2.668, sort_kernel=15.224` |
| `(32,32768)` | `128` | `8` | small-K direct final | `57.325` | `35.068` | `1.635x` | PASS | `smallk_stream_merge_unpack_kernel=3.132, sort_kernel=31.936` |
| `(32,65536)` | `4096` | `16` | large-K batch-local + final merge | `124.526` | `92.046` | `1.353x` | PASS | `core_local_nocopy_merge_kernel=13.892, sort_kernel=74.046, unpack_kernel=4.108` |
| `(32,262144)` | `64` | `64` | small-K direct final | `412.528` | `216.716` | `1.904x` | PASS | `smallk_stream_merge_unpack_kernel=7.016, sort_kernel=209.700` |
| `(32,524288)` | `1024` | `128` | small-K direct final | `842.953` | `577.203` | `1.460x` | PASS | `smallk_stream_merge_unpack_kernel=28.869, sort_kernel=548.335` |
| `(48,16384)` | `128` | `4` | small-K direct final | `33.429` | `25.065` | `1.334x` | PASS | `smallk_stream_merge4_unpack_kernel=2.980, sort_kernel=22.085` |
| `(48,16384)` | `256` | `4` | small-K direct final | `33.745` | `29.680` | `1.137x` | PASS | `smallk_stream_merge4_unpack_kernel=3.208, sort_kernel=26.472` |
| `(48,131072)` | `8192` | `32` | large-K batch-local + final merge | `274.273` | `281.429` | `0.975x` | PASS | `core_local_nocopy_merge_kernel=59.401, sort_kernel=213.756, unpack_kernel=8.272` |
| `(48,262144)` | `8192` | `64` | large-K batch-local + final merge | `534.419` | `549.495` | `0.973x` | PASS | `core_local_nocopy_merge_kernel=115.342, sort_kernel=425.724, unpack_kernel=8.429` |
| `(48,524288)` | `8192` | `128` | large-K batch-local + final merge | `1154.287` | `1166.639` | `0.989x` | PASS | `core_local_nocopy_merge_kernel=303.934, sort_kernel=853.741, unpack_kernel=8.964` |

汇总：

| 项 | 结果 |
| --- | ---: |
| 通过数 | `32/32` |
| 平均加速比 | `1.227x` |
| 中位数加速比 | `1.148x` |

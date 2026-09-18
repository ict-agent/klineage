# triton-ascend 编程规则（写 KDA 前先读）

来源：本机 `~/codex/triton-tle/references/triton-ascend/`（整理自官方文档，固定提交
`872fa1ed68c0fc82846d20bb4393a8a35ae831b3`）。版本敏感行为必须以容器内实际包为准。

## 1. 环境自检

```bash
npu-smi info
python3 -m pip show triton-ascend triton torch torch-npu
python3 -c "import torch, torch_npu, triton; print(torch.__version__, triton.__version__)"
```

- 稳定组合之一：triton-ascend 3.2.2 / triton 3.2.0 / CANN 9.1.0 / torch 2.7.1；
  本任务容器 `vllm0.23.0-zcj` 实测：`import triton` -> 3.5.1，`triton.backends.ascend` 可导入
  （`pip show triton-ascend` 报 not found，版本以 `import triton` 为准）。
- 新扩展 API（`triton.language.extra.cann.extension`、`sync_block_*`）先写最小 kernel 验证
  再进主实现；容器已确认 `sync_block_set/wait/all` 存在。

## 2. 编程模型

- 一个 program 处理一个/多个 tile；NPU 物理核只有几十个，**不要把 grid 开到成百上千**。
  固定有限 grid + 核内 stride 循环：

```python
pid, num_core = tl.program_id(0), tl.num_programs(0)
for tile in range(pid, num_tiles, num_core):
    ...
```

- 包含跨块同步/执行顺序的 kernel 不能开 `TRITON_ALL_BLOCKS_PARALLEL=1`；纯独立任务才可折叠。
- UB 预算参考 192 KB（A2/A3，随编译路径浮动）；tile 设计先算「输入+输出+中间量+双缓冲」。
  UB overflow 的处理顺序：缩小 tile -> 缩短大中间量生命周期 -> 拆 accumulator -> 拆 kernel。
- 长轴必须核内分块，不要把整张 `[4096, 128]` 搬进片上。

## 3. 常见退化点

- **int64 运算**：A2/A3 上部分 Vector ADD/CMP 的 int64 会退化为 Scalar。索引/偏移优先 int32
  （本算子 T*H*K = 5e7 量级，int32 足够；确需 int64 时只放在地址计算里）。
- **离散访问**：gather/scatter/不规则 mask 会被降级为标量循环；能先批量搬进 UB 再 select 就别直接 gather。
- **非对齐**：尾轴尽量 32B 对齐；KDA 的 D=128（bf16 256 B）天然满足，别引入不对齐的中间布局。
- **小 mask 大代价**：复杂 mask 会打断流水，能证明无尾块时就走无 mask 快路径。
- 过小 BLOCK 会增加搬运指令与循环控制；过大直接 UB overflow。

## 4. Cube（tl.dot）

- 2D tile + fp32 accumulator；输入 bf16/fp16，累加 fp32（`tl.dot(a, b, acc)` 或 `out_dtype=tl.float32`）。
- tile 形状参考：M/N 128、K 64 起步；满 K/满 V 的整 tile 优于源码层拆分后相加（见设计文档）。
- 每个 `tl.dot` 的 layout/dtype/累加类型先确认能落到 Cube，再做循环变换与流水。

## 5. Autotune 与编译选项

```python
import triton.backends.ascend.runtime   # 必须导入才启用 Ascend autotune 扩展

@triton.autotune(configs=[], key=["T", "H"])
@triton.jit
def kernel(...): ...
```

- `@triton.autotune` 必须紧贴 `@triton.jit` 外层；`configs=[]` 表示自动生成 tiling 候选。
- 想调优的 `tl.constexpr` 不要给默认值、不要在 launch 时固定；grid 写成依赖 `meta` 的 lambda。
- 也可手写 `triton.Config({...})`，并可携带 Ascend 编译选项（`multibuffer` 等）。
- 关键编译选项：`multibuffer`（默认开）、`limit_auto_multi_buffer_*`、`enable_hivm_auto_cv_balance`、
  `tile_mix_vector_loop`/`tile_mix_cube_loop`、`sync_solver`/`unit_flag`/`inject_barrier_all`、
  `enable_auto_blockify`/`auto_blockify_size`、`enable_nd2nz_on_vector`。
- 事实核对：本仓 GDN 正式实现使用 `num_warps=4, num_stages=2, enable_sync_block_lock=True,
  disable_auto_inject_block_sync=True, multibuffer=False`。不要照抄，先用目标 shape 复测。

## 6. 调试与性能

- 编译/精度：`TRITON_DEBUG=1`、`TRITON_ALWAYS_COMPILE=1`、`MLIR_ENABLE_DUMP=1`、
  `TRITON_REPRODUCER_PATH=<path>`；缓存 `~/.triton/cache`，dump `~/.triton/dump`。
- 运行时打印：`tl.static_print`（编译期）、`tl.device_print`（需 `TRITON_DEVICE_PRINT=1`）；
  大量 device print 会改变行为。
- 精度：先 PyTorch golden，再逐阶段对齐（先把 Prepare 的输出 Aqk/Akk/u/w 对到，再看 FwdH）。
  `TRITON_INTERPRET=1` 只能辅助定位，不能当性能结论。
- 性能：只用 `msprof op`/设备侧数据，不用 host 计时跨 session 比较；每次只改一个变量。

## 7. 最常见的 8 个坑（迁移 GPU Triton 代码时逐条核对）

1. grid 远大于物理核数；2. `num_warps`/`num_stages` 在目标版本语义不同或被忽略；
3. i64 运算落到 Scalar；4. 尾轴/单次搬运不对齐；5. 离散访问被降级为标量；
6. tile 超出 UB/L1；7. 依赖 GPU 调度顺序的跨块同步；8. `tl.dot` 输入不满足 Cube 路径条件。

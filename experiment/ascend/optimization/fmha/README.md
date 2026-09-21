# FMHA：Ascend 910B1 固定 Shape 优化

独立 Artifact。包含内核源码、Python wrapper、构建工具、正确性校验和 benchmark；不依赖原开发目录、其他算子或预编译的自定义二进制。

## 1. 算子定义

每条序列独立计算 `output = softmax(Q @ Kᵀ / sqrt(128)) @ V`。非因果 self-attention，无 dropout；输入输出均为连续 packed-NHD（CANN 名称 TND）。

| 输入/输出 | Shape | Dtype | 约束 |
|---|---|---|---|
| `q`, `k`, `v` | `[16384, 64, 128]` | FP16 | 同一 NPU，连续 |
| `cu_seqlens_q`, `cu_seqlens_k` | `[9]` | int32 | 同一 NPU，连续，值为 `[0,2048,...,16384]` |
| `output` | `[16384, 64, 128]` | FP16 | 连续 |

唯一测试 case：**batch=8、heads=64、seq=2048、head_dim=128、FP16**。入口 `fmha.run(q, k, v, cu_seqlens_q, cu_seqlens_k)` 每次校验 shape、dtype、device、连续性和序列边界，不按指针缓存边界内容。本实现不支持其他 shape、ragged 序列、GQA、causal、dropout 或 BF16。

## 2. Baseline

采用昇腾官方 **`torch_npu.npu_fused_infer_attention_score`，TND 布局**，由 CANN 9.1.0 内置高性能 FusedInferAttentionScore 内核执行。Baseline 与优化实现使用相同 Python API、输入、host tiling 和调度 key；优化实现替换该 key 的 AscendC 设备内核。

- 编译/调度 key：`5000000000000200100`。
- 官方二进制：`FusedInferAttentionScore_8cd36e66e4ceb2d60ef96db29a147347.o`。
- 官方 SHA256：`91644f785e1b618c1aa56f8b083d0523017091d5737fdb22ff13fdf5c21a80d3`。
- `build.py` 创建两个私有 OPP，清空第三方 vendor 优先级，baseline 直接使用上述官方文件，candidate 使用现场编译的文件。系统 OPP 不作修改。两者在独立进程加载，避免缓存混用。
- CPU FP32 PyTorch SDPA 仅作为数值 oracle，不作为性能 baseline。

本 Artifact 是对官方内核的派生优化，不是从零重写，也不把调用官方 API 本身算作优化。完整上游差异见 `kernel_vs_upstream.patch`；内核改动 **8 个文件，+371/-48 行**，包括 3 个新模块。复制的上游源码不计入改动量。

## 3. 性能结果

下表为原正式验收记录，完整数据及逐组计时保存在 `results/published/`。

| Seed | 官方 baseline（µs） | 优化实现（µs） | 加速比 |
|---|---:|---:|---:|
| 20260921 | 7517.813 | 6527.474 | 1.1517× |
| 123 | 7530.431 | 6525.557 | 1.1540× |
| 777 | 7537.358 | 6524.020 | 1.1553× |

三个 seed 的加速比几何平均为 **1.1537×**。计时采用 NPU Event：每条路径预热 **20 次**，测量 **7 组 × 50 次**；每组总时间除以 50，最终取 7 组的中位数。上表为 FIA API 调用路径，排除输入生成、正确性校验、CPU oracle 及 wrapper 的边界检查，不等同于 profiler 的单 kernel duration。

另外报告包含每次 `cu_seqlens.cpu().tolist()` 与边界检查的完整 wrapper 路径；其 baseline 分别为 7814.741、7826.709、7823.213 µs，优化后为 6685.654、6693.871、6697.522 µs，加速比几何平均 **1.1687×**。此路径同样使用 NPU Event，包含期间的设备空闲等待，不是独立 CPU wall-clock 指标。

正确性：3 个 seed，每个 seed 对全部 **134,217,728** 个输出元素与官方结果进行 3 次逐位比较，均一致；检查 FP16、连续性和有限值。首个 seed 另对全部输出验证 CPU FP32 SDPA，`atol=rtol=1e-3`，最大绝对误差 **0.000202656**。Profiler 与性能验收分开运行。机器未声明独占，耗时会受频率、温度和其他负载影响；复现脚本按当前实测结果判定，不保证不同环境必然复现同一数值。

### 独立目录复现（2026-09-21）

将本 Artifact 单独复制到干净目录后，运行 `ASCEND_RT_VISIBLE_DEVICES=3 ./run_benchmark.sh`，从源码重新编译并完成全部默认测试，退出码 **0**。

| Seed | 官方 baseline（µs） | 优化实现（µs） | 加速比 |
|---|---:|---:|---:|
| 20260921 | 7561.118 | 6512.346 | 1.1610× |
| 123 | 7549.175 | 6515.814 | 1.1586× |
| 777 | 7523.749 | 6521.216 | 1.1537× |

算子加速比几何平均 **1.1578×**，完整入口 **1.1722×**；全部正确性检查通过。源码与生成二进制的 hash 均与原验收相同。原始计时、日志和构建清单见 `results/reproduced/`。这组结果验证独立目录复现，统计配置与上表相同。

## 4. 环境

验证环境：Ubuntu 22.04.5 LTS / aarch64，Python 3.12.13，Ascend **910B1（24 AIC / 48 AIV）**，CANN **9.1.0**，PyTorch **2.10.0+cpu**，torch-npu **2.10.0.post4**。使用带有上述依赖的 `quay.io/ascend/vllm-ascend:v0.23.0` 容器环境。

需要可访问的 910B1、兼容驱动、CANN 开发工具 `asc_opc` 及上述 Python 包。建议预留至少 16 GiB 空闲 NPU 内存、16 GiB 主机内存。依赖已安装后，构建不需要网络。由于这里只打包一个官方调度 key，脚本会拒绝官方二进制 hash 不同的 CANN 安装；不宣称跨版本/跨芯片兼容。

## 5. 一键复现

在符合上述环境的 Linux 主机/容器中执行：

```bash
cd experiment/ascend/optimization/fmha
./run_benchmark.sh
```

默认使用可见物理 NPU 0；指定其他卡和 CANN 路径：

```bash
ASCEND_RT_VISIBLE_DEVICES=3 \
ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 \
./run_benchmark.sh
```

脚本从任意工作目录均可运行，自动加载 CANN 环境、编译源码、创建私有 OPP，然后运行三个 seed 的全量校验及 benchmark（包含首个 seed 的 CPU FP32 oracle）。首次 CPU oracle 需要数分钟。可用 `PYTHON=/path/to/python` 选择解释器。

产物：

- `build/manifest.json`：源代码、编译参数、官方及优化二进制 SHA256。
- `results/latest/summary.json`：绝对耗时、加速比、统计配置和 `passed`。
- `results/latest/{baseline,candidate}_*.json`：每组原始时间与正确性记录。
- `results/latest/*.log`：构建、运行日志。

仅当全部 seed 的正确性通过，且算子及完整入口加速比均 **>1.05×** 时退出码为 0；否则返回非零。原记录在 `results/published/`，不会被覆盖。只用于快速排障可运行 `./run_benchmark.sh --seeds 20260921 --trials 3 --repeat 20`，这不等价于完整验收配置。

单独使用 wrapper 时，必须在新的 Python 进程导入 torch-npu 之前选择私有 OPP：

```bash
./run_with_opp.sh candidate python3 your_program.py
# your_program.py 中：from fmha import run
```

私有 OPP 仅服务本 Artifact 的固定 case，不应注入通用推理服务。

## 6. 目录与实现

```text
fmha/
├── README.md
├── src/                         # AscendC 内核与必要编译依赖
│   └── fused_infer_attention_score/op_kernel/
├── fmha.py                      # packed-NHD Python wrapper
├── build.py                     # 编译单 key，隔离官方/优化 OPP
├── compile_param.json
├── benchmark.py                 # 正确性与性能验收
├── run_benchmark.sh              # 一键构建和验收
├── run_with_opp.sh
├── kernel_vs_upstream.patch      # 可审阅的完整内核改动
├── UPSTREAM_COMMIT
├── DESIGN.md                    # 内核结构与资源布局
├── results/published/            # 原验收数据
├── results/reproduced/           # 独立目录复现数据
└── LICENSE
```

优化保留 Q256/KV512 分块和 FP32 数值运算顺序，通过 FP32 输出累加器常驻 UB、共享 S 缓冲生命周期、固定任务映射及统一 UB 布局，消除多轮 running-O 的 GM 往返。具体资源约束与同步见 `DESIGN.md`。

源码基于 CANN ops-transformer，版本见 `UPSTREAM_COMMIT`；上游版权声明保留，许可证见 `LICENSE`。已随目录包含编译所需源码，不需要克隆上游或应用补丁后才能构建。

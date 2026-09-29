# FMHA：Packed-NHD Attention

Ascend 910B1 上的 FP16 packed attention Artifact，包含源码、Python wrapper、构建脚本、正确性测试和一键 benchmark。支持运行时 shape 分派：对齐输入使用常驻 UB 的 FIA 路径，其他输入使用独立的官方通用 attention 路径。

## 1. 算子定义与支持范围

每条序列独立计算 `output = softmax(Q @ Kᵀ / sqrt(D)) @ V`。非因果、无 dropout，连续 packed-NHD（CANN 名称 TND）布局。

| 输入/输出 | Shape | Dtype | 约束 |
|---|---|---|---|
| `q` | `[Tq, H, D]` | FP16 | 连续 NPU tensor |
| `k`, `v` | `[Tk, H, D]` | FP16 | 连续，与 Q 同设备，K/V shape 相同 |
| `cu_seqlens_q`, `cu_seqlens_k` | `[B+1]` | int32 | 连续，可在 CPU 或 Q 所在 NPU |
| `output` | `[Tq, H, D]` | FP16 | 连续 NPU tensor |

`B`、`H` 和序列长度为运行时参数；支持等长和变长序列、Q/K 不同边界及不同 token 总数。`D` 支持 **16–256 之间的 16 的整数倍**。Q/K/V 的 H、D 必须匹配；序列非空，累积长度从 0 开始、严格递增，末值分别为 Tq、Tk。当前接口不提供 GQA、causal、dropout 或其他 dtype。

入口：`fmha.run(q, k, v, cu_seqlens_q, cu_seqlens_k)`。每次调用都检查输入与边界，不按 tensor 指针缓存校验结果。

执行路径：

- **常驻 UB 路径**：D=128，Q/K 边界相同，各序列等长，长度为 512 的倍数且至少 1536。batch、heads、序列长度、任务数和 workspace 分段由运行时数据计算；固定的是 Q256/KV512 的硬件分块。CANN 负责选择 FIA 的实际调度 key，本 Artifact 替换其中的非 split-KV key。
- **通用路径**：尾块、ragged、短序列、其他 D 等输入调用官方 `torch_npu.npu_fusion_attention` 的 TND 实现。它是独立于 FIA 的算子，不会误命中被替换的 FIA key。

泛化是“运行时优化内核 + 官方通用路径”的组合，不宣称所有输入都执行自定义内核，也不对其他 shape 作加速比承诺。

## 2. 性能 Baseline 与测试 Shape

性能验收 shape：**batch=8、heads=64、seq=2048、head_dim=128、FP16**，Q/K/V 均为 `[16384,64,128]`，cu_seqlens 为 `[0,2048,...,16384]`。

Baseline 采用昇腾官方 **`torch_npu.npu_fused_infer_attention_score`，TND 布局**，执行 CANN 9.1.0 内置高性能实现。Baseline 与优化版使用相同输入、Python FIA API、host tiling 和调度 key；优化版替换该 key 的 AscendC 设备内核。

- Key：`5000000000000200100`。
- 官方文件：`FusedInferAttentionScore_8cd36e66e4ceb2d60ef96db29a147347.o`。
- 官方 SHA256：`91644f785e1b618c1aa56f8b083d0523017091d5737fdb22ff13fdf5c21a80d3`。
- 两个私有 OPP 排除第三方 vendor 覆盖，在独立进程加载；系统安装不变。
- CPU FP32 PyTorch SDPA 仅作正确性 oracle，不作性能 baseline。

本实现基于官方内核派生优化。内核相对上游的完整改动见 `kernel_vs_upstream.patch`（8 个文件，+335/-42 行）；上游源码复制不计为优化工作量。

## 3. 性能结果

以下性能数据仅对应上述验收 shape。泛化版本的验收数据保存在 `results/generalized/`；历史版本记录保留在 `results/published/`、`results/reproduced/`。

2026-09-21，泛化版一键验收通过，算子加速比几何平均 **1.1554×**。

| Seed | 官方算子（µs） | 优化算子（µs） | 加速比 |
|---|---:|---:|---:|
| 20260921 | 7525.407 | 6515.754 | 1.1550× |
| 123 | 7532.360 | 6514.926 | 1.1562× |
| 777 | 7535.256 | 6524.122 | 1.1550× |

完整入口加速比几何平均 **1.1677×**：

| Seed | 官方完整入口（µs） | 优化完整入口（µs） | 加速比 |
|---|---:|---:|---:|
| 20260921 | 7834.656 | 6744.930 | 1.1616× |
| 123 | 7891.907 | 6734.667 | 1.1718× |
| 777 | 7879.244 | 6735.221 | 1.1699× |

计时采用 **NPU Event**：每条路径预热 **20 次**，测量 **7 组 × 50 次**；每组总时间除以 50，最终取 7 组的中位数，再对三个 seed 的加速比取几何平均。

算子计时排除输入生成、正确性校验、CPU oracle 和 wrapper 边界验证，计量 FIA API 路径，不等同于 profiler 的单 kernel duration。另报告包含 `cu_seqlens.cpu().tolist()` 和边界检查的完整 wrapper 路径；它也使用 NPU Event，包含期间设备空闲等待，不是独立 CPU wall-clock 指标。

性能验收同时检查每个 seed 的全部 134,217,728 个输出元素，与官方结果进行三次逐位比较；首个 seed 另做全量 FP32 SDPA 校验（atol=rtol=1e-3）。机器未声明独占；实测时间受频率、温度和负载影响。

## 4. 正确性泛化

`check_shapes.py` 覆盖 **28 个正向 case**，只做正确性测试，不测量或报告其他 shape 的性能：

| 类别 | 覆盖内容 |
|---|---|
| 常驻路径候选 | B/H/S 分别为 1/8/1536、2/16/2048、3/8/3072、1/4/4096、1/1/1536；D=128 |
| 尾块 | B=2、H=3、S=257、D=128 |
| 变长序列 | 长度 `[17,129,513]` 与 `[1536,2048]` |
| Q/K 边界不同 | Q=`[31,65]`、KV=`[47,49]` |
| Q/K 总长度不同 | Q=`[1,33]`、KV=`[17,65]` |
| batch 与短序列 | B=8、S=65；以及 S=1024 |
| head_dim | D=16,32,...,256，每个 D 测长度 `[1,17,65]` |

所有 case 对照逐序列 **CPU FP32 SDPA** 的全部输出，容差 `atol=rtol=1e-3`；也与 baseline 进程输出比较，并检查 shape、FP16、连续性、有限值，以及 CPU/NPU 边界 tensor 与重复调用一致性。额外检查非零起点、空序列、错误末值和倒序边界均被拒绝。28 个 case 全部通过，4 类非法边界均正确拒绝。结果见 `results/generalized/shapes/summary.json`。

## 5. 环境与一键复现

验证环境：Ubuntu 22.04.5 LTS / aarch64，Python 3.12.13，Ascend **910B1（24 AIC / 48 AIV）**，CANN **9.1.0**，PyTorch **2.10.0+cpu**，torch-npu **2.10.0.post4**；容器环境为 `quay.io/ascend/vllm-ascend:v0.23.0`。

需要兼容驱动、CANN 开发工具 `asc_opc` 和上述 Python 包。建议预留 16 GiB 空闲 NPU 内存及 16 GiB 主机内存。依赖已安装后构建无需网络。脚本校验官方二进制 hash，不宣称跨 CANN 版本或跨芯片兼容。

```bash
cd experiment/ascend/optimization/fmha
./run_benchmark.sh
```

默认使用可见物理 NPU 0；也可指定：

```bash
ASCEND_RT_VISIBLE_DEVICES=3 \
ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 \
./run_benchmark.sh
```

脚本加载 CANN 环境，从源码编译，创建私有 OPP，运行指定 shape 的三 seed 性能验收与 FP32 oracle，再执行 28 个 shape 正确性测试。可用 `PYTHON=/path/to/python` 选择解释器。只有性能验收与泛化测试均通过才返回 0；其他 shape 不参与 >1.05× 的性能门槛。

产物：`build/manifest.json` 记录源码、编译参数、官方与优化二进制 hash；`results/latest/summary.json` 保存性能结果；`results/latest/shapes/summary.json` 保存泛化校验；各目录保留原始 JSON 与日志。归档结果不会被覆盖。

已构建后只运行泛化检查：

```bash
ASCEND_RT_VISIBLE_DEVICES=3 python3 check_shapes.py
```

单独调用 wrapper 时，在新的 Python 进程导入 torch-npu 前选择 OPP：

```bash
./run_with_opp.sh candidate python3 your_program.py
# your_program.py: from fmha import run
```

私有 candidate OPP 必须配合本 wrapper 使用；不要绕过输入分派直接对任意 shape 调用 FIA。

## 6. 目录与源码

```text
fmha/
├── README.md
├── src/                         # AscendC 内核与上游编译依赖
│   └── fused_infer_attention_score/op_kernel/
├── fmha.py                      # 校验、运行时分派与 Python wrapper
├── build.py
├── compile_param.json
├── benchmark.py                 # 指定 shape 的性能验收
├── check_shapes.py              # 其他 shape 的正确性检查
├── run_benchmark.sh
├── run_with_opp.sh
├── kernel_vs_upstream.patch
├── diff_upstream.json
├── UPSTREAM_COMMIT
├── DESIGN.md
├── results/generalized/
└── LICENSE
```

常驻路径保留 FP32 运算顺序，以 UB 常驻输出累加、S 缓冲生命周期复用和运行时任务映射减少 GM 往返，详见 `DESIGN.md`。源码基于 CANN ops-transformer，版本见 `UPSTREAM_COMMIT`，保留上游版权和 `LICENSE`。已包含构建所需源码，不需要另行克隆上游或应用补丁。

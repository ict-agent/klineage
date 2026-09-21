# Conv2d：Ascend / FlagTree TLE

## 算子定义

| 项目 | 规格 |
| --- | --- |
| Shape | N=8，C=64，H=W=56，F=128，K=3 |
| 输入 | 连续 NHWC `[8,56,56,64]`，FP16 |
| 权重 | 连续 `[576,128]`，FP16；HWCF 按 Kh、Kw、C 顺序展平 |
| 输出 | 连续 NHWC `[8,56,56,128]`，FP16 |
| 运算 | 前向 cross-correlation；stride=1，padding=1，dilation=1，groups=1，bias=None |

入口：`src.conv2d.run(x, weight)`，接口对应 `problems/definitions/conv2d.json`。
实现使用 FlagTree Triton TLE：`tle.scope` 划分 Vector/Cube，
`tle.dsa.ascend.L1` 与 buffer frontend 组织片上数据；Cube 按 256 个空间位置分块，
执行九次矩阵乘法，FP32 累加、FP16 输出。

## Baseline

采用 [Ascend 官方 torch-npu](https://github.com/Ascend/pytorch) 的
`torch.nn.functional.conv2d`，底层为 CANN `aclnnConvolution`，使用库默认调度。
`src/baseline.py` 按题目接口调用该实现。

比较对象为同一 Shape 卷积的两个核心设备 task：

| 实现 | profiler task 名称 |
| --- | --- |
| torch-npu / CANN | `aclnnConvolution_Conv2dWithFlag_Conv2D` |
| FlagTree TLE | `_conv_cube` |

## 性能结果与测试口径

2026-09-21，Ascend 910B1 物理 NPU 2；正确性验证通过。

| Shape / Dtype | Baseline 核心耗时 | TLE 核心耗时 | 加速比 |
| --- | ---: | ---: | ---: |
| N8 C64 H56 W56 F128 K3 / FP16 | 26.5391 μs | 17.7352 μs | **1.496×** |

- **计时边界**：运行题目接口，从 torch-npu profiler 的 `kernel_details.csv` 提取上表指定核心 task 的设备 `Duration(us)`；每次调用各对应一个核心 task。
- **预热与重复**：完成编译和正确性校验后，各预热 20 次；正式采集七轮，交替 baseline/TLE 的先后顺序。每轮每个实现紧邻测量前预热 20 次，然后测量 20 次；预热块和测量块结束时均同步设备。
- **统计方式**：CSV 按每轮调用顺序剔除预热，每个实现保留 `7×20=140` 次有效样本。主耗时为这 140 次的算术平均值；加速比为 baseline 均值 / TLE 均值。JSON 同时保存逐次耗时、七个轮均值及轮均值的中位数、最小值、最大值。
- **数据与缓存**：输入和权重均为 FP16，计时数据随机种子为 9182，固定设备地址、warm-cache。计时前完成 JIT 编译、数据传输和 CPU 精度参考计算；CPU 线程数为 4。
- **运行条件**：`torch.inference_mode()`；采集开始及每轮前后检查 NPU 健康和进程占用，要求目标卡独占。CANN 默认策略，实测 `allow_hf32_conv=True`、`allow_hf32_matmul=False`。
- **正确性**：以 CPU FP32 convolution 为参考，逐元素阈值为 `abs(a-b) <= 0.01 + 0.01*abs(b)`。同一 Shape 覆盖三个随机种子、零值、全 1、角落脉冲及非对齐连续 view；每组额外重复五次检查逐 bit 一致，并验证非默认 stream。
- **可核查记录**：`kernels.json` 保存样本、统计量、源码 SHA256、环境和设备占用；`kernels.csv` 与 profiler trace 保存原始设备任务。

## 环境与复现

实测环境：Linux aarch64，Ascend 910B1，驱动 25.5.2，CANN 9.1.0；
Python 3.12.13，torch 2.10.0+cpu，torch-npu 2.10.0.post4；
FlagTree `0.6.0+ascend.gitf56cd1bd`，Triton 3.5.1；jemalloc 主机分配器。

在已安装上述 torch、torch-npu 和 CANN 的环境中安装固定版本编译器：

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
git clone https://github.com/flagos-ai/FlagTree.git /tmp/conv2d-flagtree
git -C /tmp/conv2d-flagtree checkout f56cd1bdb98a88eedc9e3367620a6f8d68ab585b
python -m pip install 'setuptools>=40.8' 'cmake>=3.20,<4' 'ninja>=1.11.1' 'pybind11>=2.13.1' wheel
FLAGTREE_BACKEND=ascend python -m pip install /tmp/conv2d-flagtree --no-build-isolation
```

提交目录为 `experiment/ascend/optimization/conv2d/`：

```text
conv2d/
├── README.md
├── benchmark.py
├── benchmark_kernels.py
├── run_benchmark.sh
└── src/
    ├── __init__.py
    ├── baseline.py
    ├── conv2d.py
    └── tle_kernels.py
```

在该目录一键运行正确性验证与性能测试：

```bash
TRITON_CACHE_DIR=$(mktemp -d /tmp/conv2d-triton.XXXXXX) \
LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2 \
CANN_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh \
PYTHON=python bash run_benchmark.sh --device 0
```

结果保存到 `results/run.XXXXXX/`；`--output-dir DIR` 可指定新的输出目录。
`--device` 为逻辑卡号，物理卡号按 `ASCEND_RT_VISIBLE_DEVICES` 映射，
也可用 `--physical-device` 指定。Python 与 `npu-smi` 使用相同 PID 命名空间。

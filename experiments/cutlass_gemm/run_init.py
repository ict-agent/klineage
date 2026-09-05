"""Run one init action; change only the three experiment inputs below."""

from pathlib import Path

from klineage.action import init

ROOT = Path(__file__).resolve().parents[2]
PROBLEM = ROOT / "problems/gemm/reference.py"
REPO = "https://github.com/NVIDIA/cutlass.git"
EXPERT_KERNEL = ROOT / "experiments/cutlass_gemm/sm80_bf16_gemm.cu"

if __name__ == "__main__":
    kernel = init(PROBLEM, REPO, EXPERT_KERNEL)
    print(kernel.artifact_path)

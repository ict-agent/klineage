#!/bin/bash
# One-click benchmark for TopK B=64 S=4096 K=512 FP32 on Ascend 910B1.
#
# Usage: bash run_benchmark.sh [device_id]
#
# Both sides report the same number: the sum of the device-side kernel
# Duration(us) of one invocation, profiled through torch_npu over 2 warm-up +
# 5 active calls.
#   baseline : torch-npu torch.topk  (src/baseline/torch_bench.py)
#   ours     : Triton-Ascend topk    (src/ours/topk-triton.py --bench)
# Requirements: CANN set_env.sh sourced, NPU device, python3 with torch,
#   torch_npu, pandas and Ascend triton, bishengir-compile on PATH, and the
#   custom-op bitcode in src/ours/custom-topk (or TOPK_BC_DIR).
set -euo pipefail

DEVICE="${1:-0}"
B=64
S=4096
K=512
# seg_len=4096 (one segment per row) fails to compile final_merge_unpack_kernel;
# 2048 is the smallest working segment for this shape.
SEG_LEN=2048
WARMUP=2
ACTIVE=5
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BC_DIR="${TOPK_BC_DIR:-${SCRIPT_DIR}/src/ours/custom-topk}"
export TOPK_BC_DIR="${BC_DIR}"
export TOPK_SMALLK_BC_DIR="${TOPK_SMALLK_BC_DIR:-${BC_DIR}}"
export TOPK_LAYERED_BC_DIR="${TOPK_LAYERED_BC_DIR:-${BC_DIR}}"

echo "=== baseline: torch-npu torch.topk ==="
BASE_US="$(python3 "${SCRIPT_DIR}/src/baseline/torch_bench.py" "${DEVICE}" "${WARMUP}" "${ACTIVE}" \
    | tee /dev/stderr | awk -F= '/^BENCH torch.topk=/{print $2}' | awk '{print $1}')"
echo "baseline_us: ${BASE_US}"

echo "=== ours: Triton-Ascend topk ==="
OURS_US="$(python3 "${SCRIPT_DIR}/src/ours/topk-triton.py" --device "${DEVICE}" \
    --m "${B}" --n "${S}" --k "${K}" --seg_len "${SEG_LEN}" \
    --warmup "${WARMUP}" --active "${ACTIVE}" --bench \
    | tee /dev/stderr | awk '/^BENCH  torch/{print $4}' | sed 's/triton_topk=//; s/us//')"
echo "ours_us: ${OURS_US}"

echo "================ SUMMARY ================"
echo "shape: B=${B} S=${S} K=${K} fp32, seg_len: ${SEG_LEN}, device: ${DEVICE}"
echo "timing: torch_npu profiler, 2 warm-up + 5 active calls, sum of kernel Duration(us)"
echo "baseline_us: ${BASE_US}"
echo "ours_us: ${OURS_US}"
awk -v base="${BASE_US}" -v ours="${OURS_US}" 'BEGIN {printf "speedup: %.3fx\n", base / ours}'

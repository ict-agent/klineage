#!/bin/bash
# One-click benchmark for GEMM M=N=K=4096 BF16 on Ascend 910B1.
#
# Usage: bash run_benchmark.sh [device_id]
#
# Both sides run under msprof op with the same window (--warm-up=3,
# --launch-count=10) and are timed by the kernel's device-side Task Duration
# (median of the 10 profiled launches):
#   baseline : torch-npu torch.matmul
#   ours     : our_matmul, a hand-written AscendC kernel
# Requirements: CANN env sourced (set_env.sh), NPU device, python3 with
#   torch + torch_npu, msprof, cmake.
set -euo pipefail

DEVICE="${1:-0}"
M=4096
N=4096
K=4096
WARMUP=3
LAUNCHES=10
BATCH=$((WARMUP + LAUNCHES))
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${SCRIPT_DIR}/build_artifact"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "[ERROR] ASCEND_HOME_PATH not set. Source CANN set_env.sh first." >&2
    exit 1
fi
for tool in msprof cmake python3; do
    command -v "${tool}" >/dev/null 2>&1 || { echo "[ERROR] ${tool} not found." >&2; exit 1; }
done

median_us() {
    sort -n | awk '{v[NR]=$1} END {m=int(NR/2)+1; print (NR%2)?v[m]:(v[m-1]+v[m])/2}'
}

# Median Task Duration (us) of the launches msprof profiled for one kernel.
# sample_median <label> <kernel-name> <application>
sample_median() {
    local label="$1" kernel="$2" app="$3"
    local out="${WORKDIR}/prof_${label}"
    rm -rf "${out}"
    msprof op --output="${out}" --kernel-name="${kernel}" \
        --warm-up="${WARMUP}" --launch-count="${LAUNCHES}" --application="${app}" > /dev/null

    local csv count
    csv="$(find "${out}" -name 'OpBasicInfo_*.csv' | sort)"
    count="$(printf '%s\n' "${csv}" | grep -c .)"
    if [[ "${count}" -ne "${LAUNCHES}" ]]; then
        echo "[ERROR] ${label}: ${count} profiled launches, expected ${LAUNCHES}" >&2
        exit 1
    fi

    echo "${label}_samples_us: $(printf '%s\n' "${csv}" | xargs awk -F, 'FNR==2 {print $3}')" >&2
    printf '%s\n' "${csv}" | xargs awk -F, 'FNR==2 {print $3}' | median_us
}

mkdir -p "${WORKDIR}"

# 1. Build our kernel (no external dependency).
cmake -S "${SCRIPT_DIR}/src/ours" -B "${WORKDIR}/build" > /dev/null
cmake --build "${WORKDIR}/build" -j4
BIN="${WORKDIR}/build/our_matmul"

# 2. Baseline: torch-npu.
echo "=== baseline: torch-npu torch.matmul ==="
BASE_MEDIAN="$(sample_median baseline MatMulV3 \
    "python3 ${SCRIPT_DIR}/src/baseline/torch_bench.py ${DEVICE} ${BATCH}")"
echo "baseline_median_us: ${BASE_MEDIAN}"

# 3. Ours: same msprof window; the CPU golden would only delay the process.
echo "=== ours: our_matmul ==="
export SKIP_VERIFY=1
OURS_MEDIAN="$(sample_median ours our_matmul "${BIN} ${M} ${N} ${K} ${DEVICE} ${BATCH}")"
echo "ours_median_us: ${OURS_MEDIAN}"

# 4. Summary.
echo "================ SUMMARY ================"
echo "shape: ${M}x${N}x${K} bf16, device: ${DEVICE}"
echo "timing: msprof op --warm-up=${WARMUP} --launch-count=${LAUNCHES}, kernel Task Duration median"
echo "baseline_median_us: ${BASE_MEDIAN}"
echo "ours_median_us: ${OURS_MEDIAN}"
awk -v base="${BASE_MEDIAN}" -v ours="${OURS_MEDIAN}" \
    'BEGIN {printf "speedup: %.3fx\n", base / ours}'

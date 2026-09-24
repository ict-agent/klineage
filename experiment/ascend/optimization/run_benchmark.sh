#!/usr/bin/env bash
# Reproduce the GDN Stage 6/7 comparison on one Ascend device.
#
#   bash run_benchmark.sh [device_id]      # default 0
#
# Needs torch_npu, triton-ascend (FlagTree), vLLM and vllm-ascend importable;
# source the CANN set_env.sh first.
set -euo pipefail

DEVICE=${1:-0}
cd "$(dirname "$0")"
ASCEND_RT_VISIBLE_DEVICES="$DEVICE" python bench_gdn.py --device npu:0 --json results/gdn.json

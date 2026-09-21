#!/usr/bin/env bash
# Build and validate this fixed-shape artifact from any working directory.
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.1.0}"
if [[ -f "$ASCEND_HOME_PATH/set_env.sh" ]]; then
  source "$ASCEND_HOME_PATH/set_env.sh"
fi
set -u
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python3}"
cd "$ROOT"
command -v asc_opc >/dev/null || { echo 'CANN 9.1.0 development tools (asc_opc) are required.' >&2; exit 1; }
"$PYTHON" -c 'import torch, torch_npu; print("torch:", torch.__version__, "torch_npu:", torch_npu.__version__)'
mkdir -p results/latest
"$PYTHON" build.py 2>&1 | tee results/latest/build.log
"$PYTHON" benchmark.py --oracle "$@" 2>&1 | tee results/latest/benchmark.log

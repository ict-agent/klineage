#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
exec "${PYTHON:-$REPO/.venv-ascend/bin/python}" "$HERE/benchmark.py" --kernel fused_add_rmsnorm "$@"

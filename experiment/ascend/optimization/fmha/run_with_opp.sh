#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:?Usage: run_with_opp.sh baseline|candidate python script.py}"
shift
case "$MODE" in
  baseline|candidate) ;;
  *) echo 'Mode must be baseline or candidate' >&2; exit 2 ;;
esac
export ASCEND_OPP_PATH="$ROOT/opp_$MODE"
export ASCEND_CUSTOM_OPP_PATH=
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
exec "$@"

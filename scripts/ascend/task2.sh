#!/usr/bin/env bash
# Local Task 2 entry point; only `start` launches model sessions.
set -euo pipefail
TASK_REPO=$(cd "$(dirname "$0")/../.." && pwd)
export KLINEAGE_HOST=${KLINEAGE_HOST:-910b}
export KLINEAGE_CONTAINER=${KLINEAGE_CONTAINER:-pjj-fmha-opt}
export KLINEAGE_REMOTE_ROOT=${KLINEAGE_REMOTE_ROOT:-/mnt/nvme0n1/home/pjj/klineage-generation-harness}
export KLINEAGE_CONTAINER_ROOT=${KLINEAGE_CONTAINER_ROOT:-/mnt/pjj/klineage-generation-harness}
export KLINEAGE_RUN_ROOT=${KLINEAGE_RUN_ROOT:-$HOME/klineage-task2-runs}
export ASCEND_ARCH=${ASCEND_ARCH:-Ascend910B3}
cd "$TASK_REPO"
case "${1:-help}" in
  plan)
    .venv-ascend/bin/python scripts/ascend/batch.py --kernels sparse_attention,fused_add_rmsnorm --settings without_memory,with_memory --devices 2,3 --timeout 7200 --dry-run
    ;;
  start)
    source "$TASK_REPO/scripts/ascend/deepseek_env.sh"
    shift
    scripts/ascend/run.sh start --kernels sparse_attention,fused_add_rmsnorm --settings without_memory,with_memory --devices 2,3 --timeout 7200 --model deepseek-flash "$@"
    ;;
  probe|smoke|sync|bootstrap|status|logs)
    scripts/ascend/run.sh "$@"
    ;;
  *) printf '%s\n' 'Usage: scripts/ascend/task2.sh {plan|probe|smoke|sync|bootstrap|start|status|logs}' ;;
esac

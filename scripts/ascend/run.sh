#!/usr/bin/env bash
# Drive the Ascend generation experiment from this Mac.
#
# Codex runs locally; evaluation runs on 910b1 inside the vllm0.23.0-zcj container.
#
#   scripts/ascend/run.sh sync           copy the repo to 910b1 (for remote eval)
#   scripts/ascend/run.sh bootstrap      install klineage in the container
#   scripts/ascend/run.sh probe          print the remote platform string
#   scripts/ascend/run.sh smoke [npu]    remote evaluate smoke test on one NPU
#   scripts/ascend/run.sh start [args]   launch the local batch detached
#   scripts/ascend/run.sh status         batch progress
#   scripts/ascend/run.sh logs           tail the batch log
#   scripts/ascend/run.sh plot           latency curves from events.jsonl
set -euo pipefail

HOST=${HOST:-910b1}
CONTAINER=${CONTAINER:-vllm0.23.0-zcj}
# Harness root on the host: resolved there because the container's HOME is /root.
ROOT=${ROOT:-$(ssh "$HOST" 'echo $HOME/ascend-harness')}
REPO="$ROOT/repo"
LOCAL=$(cd "$(dirname "$0")/../.." && pwd)
PY="$LOCAL/.venv-ascend/bin/python"
LOG="$LOCAL/experiment/ascend/generation/batch.log"
# Unit workspaces live outside the checkout so Codex cannot walk up into
# the repo (expert packs, transfer runs). Override with RUN_ROOT=...
RUN_ROOT=${RUN_ROOT:-$HOME/klineage-runs}

container() {
  local encoded
  encoded=$(printf '%s' "$*" | base64 | tr -d '\n')
  ssh "$HOST" "docker exec -i $CONTAINER bash -lc \"\$(echo $encoded | base64 -d)\" </dev/null"
}

sync_repo() {
  ssh "$HOST" "mkdir -p $REPO"
  # The container runs as root and leaves root-owned bytecode behind, which
  # blocks the next --delete; hand the tree back to the host user first.
  local uid gid
  uid=$(ssh "$HOST" 'id -u')
  gid=$(ssh "$HOST" 'id -g')
  container "chown -R $uid:$gid $REPO 2>/dev/null || true"
  # --delete-excluded: the host copy must mirror this sanitized subset, and
  # plain --delete keeps the excluded expert packs and transfer runs there,
  # where an agent with ssh can read them.
  rsync -az --delete --delete-excluded \
    --exclude .git --exclude '*.pdf' --exclude __pycache__ \
    --exclude 'scripts/ascend/expert' --exclude 'scripts/apply_transfer.py' \
    --exclude 'experiment/_transfer' --exclude 'experiment/transfer_*' \
    --exclude 'scripts/ascend/batch.py' --exclude 'scripts/ascend/collect.py' \
    --exclude 'scripts/ascend/gen_inputs.py' --exclude 'scripts/ascend/plot.py' \
    --exclude 'scripts/ascend/smoke_eval.py' --exclude 'scripts/ascend/README.md' \
    --exclude 'scripts/ascend/*.tmpl' \
    --include 'experiment/' \
    --include 'experiment/*/' \
    --include 'experiment/*/problems/' \
    --include 'experiment/*/problems/**' \
    --exclude 'experiment/***' \
    "$LOCAL/" "$HOST:$REPO/"
}

bootstrap() {
  sync_repo
  container "cd $REPO && pip install -q -e . && python -c 'import torch, torch_npu, klineage; print(\"torch\", torch.__version__, \"npu\", torch_npu.__version__)'"
}

probe() { container "cd $REPO && KLINEAGE_BACKEND=ascend ASCEND_ARCH=dav-c220 python scripts/ascend/remote_eval.py --probe"; }

smoke() { container "cd $REPO && python scripts/ascend/smoke_eval.py --repo $REPO --work $ROOT/smoke --device ${1:-2}"; }

start() {
  mkdir -p "$(dirname "$LOG")"
  # Detached: a plain nohup child dies with the shell's process group when the
  # launching session ends.
  "$PY" "$LOCAL/scripts/ascend/spawn.py" "$LOG" \
    "$PY" "$LOCAL/scripts/ascend/batch.py" "$@"
  echo "started; scripts/ascend/run.sh status"
}

status() {
  test -f "$RUN_ROOT/status.json" \
    && cat "$RUN_ROOT/status.json" \
    || tail -n 30 "$LOG"
}

logs() { tail -n 60 "$LOG"; }

plot() { "$PY" "$LOCAL/scripts/ascend/plot.py" --root "$RUN_ROOT" --out "$RUN_ROOT/plots"; }

case "${1:-}" in
  sync) shift; sync_repo "$@" ;;
  bootstrap) shift; bootstrap "$@" ;;
  probe) shift; probe "$@" ;;
  smoke) shift; smoke "$@" ;;
  start) shift; start "$@" ;;
  status) shift; status "$@" ;;
  logs) shift; logs "$@" ;;
  plot) shift; plot "$@" ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac

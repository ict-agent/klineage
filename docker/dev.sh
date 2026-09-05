#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}"

export KLINEAGE_UID="${KLINEAGE_UID:-$(id -u)}"
export KLINEAGE_GID="${KLINEAGE_GID:-$(id -g)}"

action="${1:-shell}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "${action}" in
    build)
        exec docker compose build dev "$@"
        ;;
    shell)
        exec docker compose run --rm dev bash "$@"
        ;;
    run)
        if [[ $# -eq 0 ]]; then
            echo "usage: ./docker/dev.sh run COMMAND [ARG ...]" >&2
            exit 2
        fi
        exec docker compose run --rm dev "$@"
        ;;
    check)
        exec docker compose run --rm dev python docker/smoke_test.py "$@"
        ;;
    up)
        exec docker compose up --detach dev "$@"
        ;;
    down)
        exec docker compose down "$@"
        ;;
    *)
        echo "unknown action: ${action}" >&2
        echo \
            "usage: ./docker/dev.sh {build|shell|run|check|up|down}" \
            >&2
        exit 2
        ;;
esac

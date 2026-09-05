#!/usr/bin/env bash
set -euo pipefail

workspace="${KLINEAGE_WORKSPACE:-/workspace/klineage}"
project_file="${workspace}/pyproject.toml"

if [[ -f "${project_file}" ]]; then
    uv pip install \
        --python "${VIRTUAL_ENV}/bin/python" \
        --editable "${workspace}"
fi

exec "$@"

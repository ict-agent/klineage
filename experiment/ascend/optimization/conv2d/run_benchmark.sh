#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

device_args=()
output_dir=""
while (($#)); do
    case "$1" in
        --device|--physical-device)
            device_args+=("$1" "${2:?Missing device index}")
            shift 2 ;;
        --output-dir)
            output_dir="${2:?Missing output directory}"
            shift 2 ;;
        *) echo "Usage: $0 [--device N] [--physical-device N] [--output-dir DIR]" >&2; exit 2 ;;
    esac
done

if [[ -n "${CANN_ENV:-}" ]]; then
    set +u
    source "$CANN_ENV"
    set -u
fi
if [[ -z "$output_dir" ]]; then
    mkdir -p results
    output_dir=$(mktemp -d results/run.XXXXXX)
else
    mkdir -p "$output_dir"
fi

run_stage() {
    local script="$1" report_name="$2"
    if ! "${PYTHON:-python3}" "$script" "${device_args[@]}" \
        --output "$output_dir/$report_name.json" >"$output_dir/$report_name.log" 2>&1; then
        echo "FAIL: see $output_dir/$report_name.log" >&2
        return 1
    fi
}

run_stage benchmark.py full
run_stage benchmark_kernels.py kernels
cat "$output_dir/kernels.txt"
echo "PASS: $output_dir"

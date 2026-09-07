"""NCU command construction and CSV decoding."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from klineage.profiling import ProfileOptions


_NCU = "/usr/local/cuda/bin/ncu"
_RANGE = "klineage_profile"
_REGEX = "regex:"
_ID = "ID"
_KERNEL = "Kernel Name"
_METRIC = "Metric Name"
_FIELDS = {
    "kernel": _KERNEL, "launch_id": _ID, "section": "Section Name",
    "metric": _METRIC, "unit": "Metric Unit", "value": "Metric Value",
}
_RAW_META = frozenset((
    _ID, "Process ID", "Process Name", "Host Name", _KERNEL, "Context", "Stream",
    "Block Size", "Grid Size", "Device", "CC",
))
_ERRORS = {
    "ERR_NVGPUCTRPERM": "NCU ERR_NVGPUCTRPERM: GPU counter permission denied",
    "No kernels were profiled": "NCU: No kernels were profiled",
}
_LOG_PREFIXES = ("==PROF==", "==WARNING==", "==ERROR==")


def _ncu_command(
    options: ProfileOptions, raw_path: Path, report_path: Path,
) -> tuple[str, ...]:
    # The worker brackets only target launches; nested NVTX ranges remain included.
    command = [
        _NCU, "--target-processes", "application-only", "--replay-mode", "kernel",
        "--nvtx", "--nvtx-include", _RANGE + "/", "--set", options.set,
        "--page", "raw", "--csv", "--print-units", "base",
        "--log-file", str(raw_path), "--export", str(report_path),
        "--force-overwrite", "--kernel-name-base", "demangled",
    ]
    for section in options.sections:
        command.extend(("--section", section))

    if options.kernel_filter:
        pattern = options.kernel_filter
        command.extend(("--kernel-name", pattern if pattern.startswith(_REGEX) else _REGEX + pattern))

    return tuple(command)


def _parse_metrics(text: str) -> list[dict[str, str]]:
    for marker, message in _ERRORS.items():
        if marker in text:
            raise ValueError(message)

    metrics = []
    header = []
    units = {}

    # Parse complete CSV records so quoted newlines survive intervening profiler logs.
    for row in csv.reader(io.StringIO(text, newline="")):
        if not row or row[0].startswith(_LOG_PREFIXES):
            continue
        if _ID in row and _KERNEL in row:
            header, units = row, {}
            continue
        if not header or len(row) != len(header):
            continue

        values = dict(zip(header, row))
        if _METRIC in values:
            if not all(name in values for name in _FIELDS.values()):
                continue
            metric = {name: values[column] for name, column in _FIELDS.items()}
            if all(metric[name].strip() for name in ("kernel", "launch_id", "metric", "value")):
                metrics.append(metric)
            continue

        # Raw CSV stores units above launch rows and provides no section mapping.
        if not values[_ID]:
            units = values
            continue
        if not values[_KERNEL].strip():
            continue
        for name, value in values.items():
            if name in _RAW_META or not name.strip() or not value.strip():
                continue
            metrics.append({
                "kernel": values[_KERNEL], "launch_id": values[_ID], "section": "",
                "metric": name, "unit": units.get(name, ""), "value": value,
            })

    if not metrics:
        raise ValueError("NCU report contains no metrics")
    return metrics

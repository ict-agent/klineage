"""Plot performance per evaluate call against elapsed time or spent tokens.

Reads the unit artifacts collected under the run root (~/klineage-runs).
Writes one PNG per kernel, a CSV per unit, and table.md: the summary the task
asks for (correct, latency, speedup over the torch-npu baseline). Falls back to
CSV when matplotlib is unavailable.
"""

from __future__ import annotations

import argparse
import os
import csv
import json
from datetime import datetime
from pathlib import Path

SETTINGS = ("without_memory", "with_memory")
#: The task's labels for the two settings; the directory names stay the run ids.
SETTING_LABELS = {
    "without_memory": "Without Expert Knowledge",
    "with_memory": "With Expert Knowledge",
}


def read_unit(unit: Path) -> list[dict]:
    events = unit / "events.jsonl"
    if not events.is_file():
        return []
    rows, tokens, started = [], 0, None
    for line in events.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        when = datetime.fromisoformat(event["at"])
        started = started or when
        if event.get("event") == "api_response":
            tokens += event.get("total_tokens") or 0
            continue
        if event.get("event") != "evaluate":
            continue
        rows.append({
            "at": event["at"],
            "seconds": round((when - started).total_seconds(), 1),
            "tokens": tokens,
            "latency_ms": event.get("latency_ms"),
            "baseline_ms": event.get("baseline_ms"),
            "speedup": event.get("speedup"),
            "correctness_passed": event.get("correctness_passed"),
        })
    return rows


def summarize(kernel: str, setting: str, rows: list[dict]) -> str:
    """One table row: the best passing measurement of a unit, or its status."""

    label = SETTING_LABELS.get(setting, setting)
    passing = [row for row in rows
               if row["correctness_passed"] and row["latency_ms"]]
    correct = "✓" if passing else "✗"
    if not passing:
        return f"| {kernel.upper()} | {label} | {correct} | - | - |"
    best = min(passing, key=lambda row: row["latency_ms"])
    latency = f"{best['latency_ms'] * 1000:.1f} us"
    speedup = f"{best['speedup']:.2f}×" if best.get("speedup") else "-"
    return f"| {kernel.upper()} | {label} | {correct} | {latency} | {speedup} |"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                    default=Path(os.environ.get("KLINEAGE_RUN_ROOT", "~/klineage-runs")).expanduser())
    parser.add_argument("--out", type=Path, default=Path("experiment/ascend/plots"))
    parser.add_argument("--x", choices=("seconds", "tokens"), default="tokens")
    parser.add_argument("--y", choices=("latency", "speedup"), default="latency")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    series: dict[str, dict[str, list[dict]]] = {}
    for kernel_dir in sorted(args.root.iterdir()):
        if not kernel_dir.is_dir():
            continue
        for setting in SETTINGS:
            rows = read_unit(kernel_dir / setting)
            if rows:
                series.setdefault(kernel_dir.name, {})[setting] = rows

    for kernel, groups in series.items():
        for setting, rows in groups.items():
            path = args.out / f"{kernel}-{setting}.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    if not series:
        raise SystemExit(f"no events.jsonl under {args.root}")

    # table.md: the Kernel/Setting/Correct/Latency/vs-Baseline summary.
    lines = ["| Kernel | Setting | Correct | Latency (us) | vs. Baseline |",
             "| --- | --- | --- | --- | --- |"]
    for kernel, groups in sorted(series.items()):
        for setting, rows in sorted(groups.items()):
            lines.append(summarize(kernel, setting, rows))
    (args.out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing; CSV written to", args.out)
        return

    field = "latency_ms" if args.y == "latency" else "speedup"
    label = "latency (ms)" if args.y == "latency" else "speedup vs baseline (x)"
    figure, axes = plt.subplots(1, len(series), figsize=(4 * len(series), 3.2), squeeze=False)
    for column, (kernel, groups) in enumerate(sorted(series.items())):
        axis = axes[0][column]
        for setting, rows in groups.items():
            points = [row for row in rows if row.get(field)]
            axis.plot(
                [row[args.x] for row in points],
                [row[field] for row in points],
                marker="o",
                label=setting,
            )
        axis.set_title(kernel)
        axis.set_xlabel(args.x)
        axis.set_ylabel(label)
        axis.legend()
    figure.tight_layout()
    target = args.out / f"{args.y}-vs-{args.x}.png"
    figure.savefig(target, dpi=150)
    print("wrote", target)


if __name__ == "__main__":
    main()

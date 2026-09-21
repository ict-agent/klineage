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

#: A gated call within this factor of the best one ends the useful search: the
#: curve is flat afterwards, so that call is the run's peak.
PLATEAU = 1.05


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
        when = datetime.fromisoformat(event.get("finished_at") or event["at"])
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
            "compile_passed": event.get("compile_passed", True),
            "profile_passed": event.get("profile_passed", True),
            "version": event.get("version"),
        })
    return rows


def passed(row: dict) -> bool:
    return all(row.get(key, True) for key in
               ("correctness_passed", "compile_passed", "profile_passed"))


def running_best(rows: list[dict], field: str, x: str) -> list[dict]:
    best, points = None, []
    for row in sorted(rows, key=lambda item: item[x]):
        value = row.get(field)
        if not passed(row) or value is None or value <= 0:
            continue
        best = value if best is None else (max(best, value) if field == 'speedup' else min(best, value))
        points.append(dict(row, **{field: best}))
    return points


def observed_end(unit: Path, x: str) -> float:
    events = []
    path = unit/'events.jsonl'
    if not path.exists():
        return 0
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    if x == 'tokens':
        return sum(e.get('total_tokens') or 0 for e in events if e.get('event') == 'api_response')
    times = [datetime.fromisoformat(e.get('finished_at') or e['at']) for e in events if e.get('at')]
    return (max(times)-min(times)).total_seconds() if times else 0


def milestones(rows: list[dict]) -> dict:
    """First passing call, plateau, and best call of one unit.

    ``rows`` come from :func:`read_unit`, so each one already carries its
    position on the time and token axes.
    """

    passing = [row for row in rows
               if passed(row) and row.get("latency_ms")]
    if not passing:
        return {"gate_calls": len(rows), "rejected": len(rows)}
    best = min(passing, key=lambda row: row["latency_ms"])
    threshold = best["latency_ms"] * PLATEAU
    peak = next((row for row in passing if row["latency_ms"] <= threshold), None)
    return {"first": passing[0], "peak": peak, "best": best,
            "gate_calls": len(rows), "rejected": len(rows) - len(passing)}


def summarize(kernel: str, setting: str, rows: list[dict]) -> str:
    """One table row: the best passing measurement of a unit, or its status."""

    label = SETTING_LABELS.get(setting, setting)
    passing = [row for row in rows
               if passed(row) and row["latency_ms"]]
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
    parser.add_argument("--y", choices=("latency", "speedup"), default="speedup")
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
    label = "Running-best valid latency (ms, log)" if args.y == "latency" else "Running-best valid speedup vs torch (log)"
    figure, axes = plt.subplots(1, len(SETTINGS), figsize=(12, 4), sharey=True, squeeze=False)
    colors = {kernel: f'C{index}' for index, kernel in enumerate(sorted(series))}
    for column, setting in enumerate(SETTINGS):
        axis = axes[0][column]
        for kernel, groups in sorted(series.items()):
            points = running_best(groups.get(setting, []), field, args.x)
            if not points:
                continue
            x = [row[args.x] for row in points]
            y = [row[field] for row in points]
            end = observed_end(args.root/kernel/setting, args.x)
            if end > x[-1]:
                x.append(end)
                y.append(y[-1])
            axis.step(x, y, where='post', linewidth=2, color=colors[kernel], label=kernel)
            axis.scatter([row[args.x] for row in points], [row[field] for row in points],
                         s=10, color=colors[kernel])
        axis.set_title(SETTING_LABELS[setting])
        axis.set_xlabel('Elapsed time (seconds)' if args.x == 'seconds' else 'Cumulative API tokens')
        axis.set_yscale('log')
        extent = max((observed_end(args.root/kernel/group, args.x)
                      for kernel in series for group in SETTINGS), default=1)
        axis.set_xlim(0, max(1, extent))
        axis.grid(True, which='both', alpha=0.25)
        if args.y == 'speedup':
            axis.axhline(1, color='gray', linestyle=':', linewidth=1)
            axis.text(0.98, 1, 'torch parity', transform=axis.get_yaxis_transform(),
                      ha='right', va='bottom', color='gray')
        if axis.get_legend_handles_labels()[0]:
            axis.legend()
        else:
            axis.text(0.5, 0.5, 'No valid measurement', transform=axis.transAxes, ha='center')
    axes[0][0].set_ylabel(label)
    figure.tight_layout()
    target = args.out / f"{args.y}-vs-{args.x}.png"
    figure.savefig(target, dpi=180)
    figure.savefig(target.with_suffix('.pdf'))
    plt.close(figure)
    print("wrote", target)


if __name__ == "__main__":
    main()

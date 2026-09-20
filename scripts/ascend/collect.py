#!/usr/bin/env python3
"""Assemble the submitted artifact tree from a finished run root.

The task asks for one directory per unit::

    kernel_name/
    |- events.jsonl     # api_response + evaluate records
    |- trace.jsonl      # Codex session trace
    `- versions/        # kernel artifact per gate call
        `- versionN/

Reads ``~/klineage-runs/<kernel>/<setting>`` and writes the same layout under
``experiment/ascend/generation/<kernel>/``, expanding every version's frozen
``kernel.json`` into a readable ``submission/`` bundle and adding README.md
with the measured protocol and the summary table (see plot.py).

    scripts/ascend/collect.py --kernel kda [--root ~/klineage-runs]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

SETTINGS = ("without_memory", "with_memory")
#: Copied verbatim from the run root into the submission directory.
UNIT_FILES = ("events.jsonl", "trace.jsonl", "result.json", "final_message.txt",
              "baseline.json")


def expand(version: Path) -> None:
    """Write a version's frozen sources back out as a submission bundle."""

    manifest = version / "kernel.json"
    if not manifest.is_file():
        return
    sources = json.loads(manifest.read_text(encoding="utf-8")).get("source_files") or {}
    for name, text in sources.items():
        target = version / "submission" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


def copy_unit(unit: Path, target: Path) -> None:
    if target.exists():     # stale artifacts would otherwise mix with this run
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name in UNIT_FILES:
        if (unit / name).is_file():
            shutil.copy2(unit / name, target / name)
    for version in sorted((unit / "versions").glob("version*")):
        target_version = target / "versions" / version.name
        shutil.copytree(version, target_version, dirs_exist_ok=True)
        expand(target_version)
    if (unit / "submission").is_dir():
        shutil.copytree(unit / "submission", target / "submission", dirs_exist_ok=True)


def readme(kernel: str, collected: dict[str, list[dict]], notes: str = "") -> str:
    """Protocol, baseline, and the summary table the task asks for."""

    from plot import summarize

    lines = [
        f"# {kernel} — AI-generated Ascend implementation",
        "",
        "Both settings run the same prompt, the same gate, and the same budget;",
        "the only difference is the expert material in `work/expert/`.",
        "",
        "## Measurement protocol",
        "",
        "- Gate: `scripts/ascend/eval.py`, one call per version, evaluated on 910B1",
        "  inside the `vllm0.23.0-zcj` container.",
        "- Timer: NPU stream events (`npu-events`) around one operator call,",
        "  10 warmup + 50 timed iterations, 3 trials, median of trial medians.",
        "- Baseline: the problem's torch reference on the NPU, same timer and",
        "  policy, measured once per device and cached in `baseline.json`.",
        "- Speedup: `baseline_ms / latency_ms`.",
        "",
        "## Results",
        "",
        "| Kernel | Setting | Correct | Latency (us) | vs. Baseline |",
        "| --- | --- | --- | --- | --- |",
    ]
    for setting, rows in collected.items():
        lines.append(summarize(kernel, setting, rows))
    if notes:
        lines += ["", notes.strip(), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="kda")
    parser.add_argument("--root", type=Path,
                        default=Path(os.environ.get("KLINEAGE_RUN_ROOT",
                                                    "~/klineage-runs")).expanduser())
    parser.add_argument("--out", type=Path, default=Path("experiment/ascend/generation"))
    parser.add_argument("--notes", type=Path,
                        help="markdown file appended to README.md (deviations, baselines)")
    args = parser.parse_args(argv)

    from plot import read_unit

    source = args.root / args.kernel
    if not source.is_dir():
        raise SystemExit(f"no run root at {source}")
    target = args.out / args.kernel
    target.mkdir(parents=True, exist_ok=True)

    collected = {}
    for setting in SETTINGS:
        unit = source / setting
        if not unit.is_dir():
            continue
        copy_unit(unit, target / setting)
        collected[setting] = read_unit(unit)
        print(f"{setting}: {len(collected[setting])} evaluate(s), "
              f"{len(list((target / setting / 'versions').glob('version*')))} version(s)")

    notes = args.notes.read_text(encoding="utf-8") if args.notes else ""
    (target / "README.md").write_text(readme(args.kernel, collected, notes), encoding="utf-8")
    print("wrote", target)


if __name__ == "__main__":
    main()

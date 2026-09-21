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
import textwrap
from pathlib import Path

SETTINGS = ("without_memory", "with_memory")
#: The gate's numerical tolerance (``src/klineage/constants.py``); a definition
#: cannot ask for a different one, so both settings are graded the same way.
NUMERICAL_TOLERANCE = 1e-2
#: Where a run root keeps the problem this kernel solves.
DEFINITION_DIR = "work/problems/definitions"
#: The expert pack is this study's input, not run output: keep it across collects.
EXPERT_DIR = "expert"
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
        for item in target.iterdir():
            if item.name == EXPERT_DIR:
                continue
            shutil.rmtree(item) if item.is_dir() else item.unlink()
    target.mkdir(parents=True, exist_ok=True)
    for name in UNIT_FILES:
        if (unit / name).is_file():
            shutil.copy2(unit / name, target / name)
    for version in sorted((unit / "versions").glob("version*")):
        target_version = target / "versions" / version.name
        shutil.copytree(version, target_version, dirs_exist_ok=True)
        expand(target_version)
    if (unit / "submission").is_dir():
        shutil.copytree(unit / "submission", target / "submission", dirs_exist_ok=True)


def shape_of(spec: dict, axes: dict[str, int]) -> str:
    """Render a tensor's shape with the problem's constant axes resolved."""

    names = spec.get("shape") or []
    if not names:
        return "scalar"
    return "[" + ", ".join(str(axes.get(name, name)) for name in names) + "]"


def tensor_rows(specs: dict, axes: dict[str, int], kind: str) -> list[str]:
    rows = []
    for name, spec in specs.items():
        rows.append(f"| `{name}` | {kind} | {shape_of(spec, axes)} | {spec['dtype']} |")
    return rows


def oracle_line(definition: dict) -> str:
    """What the gate grades: the torch reference, its bound, its own checker."""

    common = (f"Both settings are graded against the definition's torch reference "
              f"running on the NPU: output shape, dtype and device, then "
              f"`torch.allclose(rtol=atol={NUMERICAL_TOLERANCE:g})` "
              f"(`NUMERICAL_TOLERANCE`, `src/klineage/constants.py`).")
    if "def check_outputs" in (definition.get("reference") or ""):
        return (common + " This definition also defines `check_outputs`, which the "
                "gate runs on top of that; its assertions are in the definition's "
                "reference field.")
    return common


def problem_block(definition: dict) -> str:
    """The operator: axes, tensors, and the oracle, straight from the definition."""

    axes = {name: axis["value"] for name, axis in definition.get("axes", {}).items()}
    op_type = definition.get("op_type", "op")
    headline = definition.get("description") or ""
    title = f"`{definition['name']}` ({op_type})"
    if headline and headline != op_type:
        title += f" — {headline}"
    lines = [
        "## Problem",
        "",
        title,
        "",
        "Axes: " + ", ".join(f"{name}={value}" for name, value in axes.items()) + ".",
        "",
        "| Tensor | Direction | Shape | Dtype |",
        "| --- | --- | --- | --- |",
    ]
    lines += tensor_rows(definition.get("inputs", {}), axes, "input")
    lines += tensor_rows(definition.get("outputs", {}), axes, "output")
    lines += ["", *textwrap.wrap(oracle_line(definition), width=72), ""]
    return "\n".join(lines[:-1])


def unit_device(source: Path, setting: str) -> str | None:
    """The NPU a setting ran on, as recorded by its unit result."""

    result = source / setting / "result.json"
    if not result.is_file():
        return None
    return json.loads(result.read_text(encoding="utf-8")).get("device")


def unit_language(source: Path, setting: str) -> str | None:
    """The `language` key of a setting's submitted bundle."""

    config = source / setting / "submission" / "config.toml"
    if not config.is_file():
        return None
    for line in config.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip() == "language":
            return value.strip().strip('"')
    return None


def setup_block(source: Path) -> list[str]:
    """Language pin and per-setting devices, read back from the collected tree."""

    devices = [f"{setting} on NPU {device}"
               for setting in SETTINGS
               if (device := unit_device(source, setting))]
    languages = sorted({language for setting in SETTINGS
                        if (language := unit_language(source, setting))})
    if not devices and not languages:
        return []
    lines = ["## Setup", ""]
    if languages:
        lines += textwrap.wrap(
            f"- Language: `{' / '.join(languages)}` (`submission/config.toml`), shared "
            f"by both settings; the task leaves the choice open.",
            width=72, subsequent_indent="  ")
    if devices:
        lines += textwrap.wrap(
            f"- Devices: {'; '.join(devices)}; each setting scores against the "
            f"baseline measured on its own device.",
            width=72, subsequent_indent="  ")
    lines.append("")
    return lines


def baseline_line(collected: dict[str, list[dict]]) -> list[str]:
    """The flat reference lines of the curves: one baseline per setting."""

    from plot import SETTING_LABELS

    parts = []
    for setting, rows in collected.items():
        values = {row["baseline_ms"] for row in rows if row.get("baseline_ms")}
        if values:
            parts.append(f"{SETTING_LABELS.get(setting, setting)} {min(values):.4g} ms")
    if not parts:
        return []
    return textwrap.wrap(
        f"- Baselines, the flat lines the curves are read against: {'; '.join(parts)}.",
        width=72, subsequent_indent="  ")


def cell(row: dict) -> str:
    """One milestone as time · tokens · gated latency (speedup vs baseline)."""

    speedup = f"{row['speedup']:.2f}×" if row.get("speedup") else "-"
    return (f"{row['seconds'] / 60:.1f} min · {row['tokens'] / 1e6:.2f} M · "
            f"{row['latency_ms']:.4g} ms ({speedup})")


def trajectory_row(setting: str, rows: list[dict]) -> str:
    """Where a session left the baseline, where it peaked, and where it ended."""

    from plot import SETTING_LABELS, milestones

    marks = milestones(rows)
    calls = f"{marks['gate_calls']}"
    if marks["rejected"]:
        calls += f" ({marks['rejected']} rejected)"
    label = SETTING_LABELS.get(setting, setting)
    if "first" not in marks:
        return f"| {label} | - | - | - | {calls} |"
    return (f"| {label} | {cell(marks['first'])} | {cell(marks['peak'])} | "
            f"{cell(marks['best'])} | {calls} |")


def readme(kernel: str, collected: dict[str, list[dict]], notes: str = "",
           problem: str = "", setup: list[str] | None = None) -> str:
    """Study setup, protocol, results, and where each setting peaked."""

    from plot import PLATEAU, summarize

    lines = [
        f"# {kernel} — AI-generated Ascend implementation",
        "",
        "One Codex session per setting, same prompt, same gate, same budget; the",
        "only difference is the expert material the agent may read, under",
        "`with_memory/work/expert/` versus nothing.",
        "",
        "The deliverable is a curve, not a single number: every accepted version",
        "has a gated latency, so a session traces its performance against",
        "wall-clock time and spent tokens. The baseline is a flat line — the",
        "torch reference never improves. Read the curves for two things: how far",
        "above that line a setting ended up, and how early in time/tokens it got",
        "there.",
        "",
    ]
    if problem:
        lines += [problem, ""]
    if setup:
        lines += setup
    lines += [
        "## Measurement protocol",
        "",
        "- Gate: `scripts/ascend/eval.py`, one call per candidate version,",
        "  evaluated on 910B1 inside the `vllm0.23.0-zcj` container (torch-npu,",
        "  CANN 9.1.0).",
        "- Timer: NPU stream events (`npu-events`) around one operator call:",
        "  10 warmup + 50 timed iterations, 3 trials, median of trial medians.",
        "  Compilation and input preparation sit outside the timed interval.",
        "- Baseline: the problem definition's torch reference on the NPU, same",
        "  timer and policy, measured once per device and cached in",
        "  `baseline.json`; both settings of a kernel score against the baseline",
        "  of the device they ran on.",
        "- Speedup: `baseline_ms / latency_ms`.",
        "- A version counts only when the gate passes it: output shape, dtype and",
        "  device as the definition asks, plus its correctness oracle.",
        "  Every evaluated version is frozen under `versions/version<N>/`.",
        "",
        "## Results",
        "",
        "| Kernel | Setting | Correct | Latency (us) | vs. Baseline |",
        "| --- | --- | --- | --- | --- |",
    ]
    for setting, rows in collected.items():
        lines.append(summarize(kernel, setting, rows))
    if baselines := baseline_line(collected):
        lines += ["", *baselines]

    lines += [
        "",
        "## Trajectory",
        "",
        *textwrap.wrap(
            "Milestones of each session, clocked from its first API response, so both "
            "axes of the curve are readable. `first` is the gate call that first passed: "
            "the curve leaves the flat baseline line there. `peak` is the first gated "
            f"call within {PLATEAU - 1:.0%} of that setting's best; from there on, session "
            "time stopped buying performance. `best` is the final artifact's own "
            "measurement.",
            width=72,
        ),
        "",
        "| Setting | First passing gate | Peak (≤1.05× best) | Best gate | Gate calls |",
        "| --- | --- | --- | --- | --- |",
    ]
    for setting, rows in collected.items():
        lines.append(trajectory_row(setting, rows))

    lines += [
        "",
        "## Artifacts",
        "",
        "```text",
        f"{kernel}/",
        "|- README.md               this file",
        "|- plots/                  csv per setting, table.md, latency curve",
        "|- without_memory/         bare prompt",
        "`- with_memory/            prompt plus the expert pack (see expert/source.json)",
        "```",
        "",
        "Each setting directory holds `events.jsonl` (one `api_response` record",
        "per model call, one `evaluate` record per gate call, both timestamped),",
        "`trace.jsonl` (the Codex session), `baseline.json`, the frozen",
        "`versions/version<N>/` bundles and the submitted `submission/`.",
        "",
        "## Regenerating",
        "",
        "```sh",
        f"scripts/ascend/collect.py --kernel {kernel}   # rebuild this tree from ~/klineage-runs",
        "scripts/ascend/plot.py --root <run root> --x tokens --y latency",
        "```",
    ]
    if notes:
        lines += ["", notes.strip(), ""]
    return "\n".join(lines) + "\n"


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
    problem = ""
    for setting in SETTINGS:
        definition = source / setting / DEFINITION_DIR / f"{args.kernel}.json"
        if definition.is_file():
            problem = problem_block(json.loads(definition.read_text(encoding="utf-8")))
            break
    (target / "README.md").write_text(
        readme(args.kernel, collected, notes, problem, setup_block(source)),
        encoding="utf-8")
    print("wrote", target)


if __name__ == "__main__":
    main()

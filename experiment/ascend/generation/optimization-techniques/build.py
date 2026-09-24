"""Build the two source-audited AscendC technique matrices."""

import csv
import json
import re
import shutil
import subprocess
import xml.sax.saxutils
from pathlib import Path


HERE = Path(__file__).resolve().parent
GENERATION = HERE.parent
DATA = json.loads((HERE / "matrix.json").read_text(encoding="utf-8"))
OPS = DATA["operators"]
ROWS = DATA["rows"]
SETTINGS = DATA["settings"]

WIDTH = 1810
LEFT = 790
CELL = 180
TABLE_X = 40
TABLE_RIGHT = TABLE_X + LEFT + CELL * len(OPS)
TITLE_Y = 42
SUBTITLE_Y = 72
HEADER_TOP = 98
HEADER_BOTTOM = 390
DATA_TOP = 406
SECTION_H = 36
ROW_H = 58
BOTTOM_PAD = 100


def esc(text):
    return xml.sax.saxutils.escape(str(text), {'"': "&quot;"})


def check_configs():
    pattern = re.compile(r'language\s*=\s*["\']ascendc["\']')
    for operator in OPS:
        for setting in SETTINGS:
            config = GENERATION / operator["key"] / setting / "submission" / "config.toml"
            if not config.is_file() or not pattern.search(config.read_text(encoding="utf-8")):
                raise SystemExit(f"Expected AscendC config: {config}")


def check_matrix():
    if not OPS or not ROWS:
        raise SystemExit("The matrix needs operators and rows")
    sections = [row["section"] for row in ROWS]
    if sections != sorted(sections, key=sections.index):
        raise SystemExit("Sections must remain grouped and ordered")
    for row in ROWS:
        for setting in SETTINGS.values():
            values = row[setting["mark"]]
            if len(values) != len(OPS):
                raise SystemExit(f"Wrong number of cells in: {row['label']}")


def check_mark(cx, cy, present):
    if present:
        return (
            f'<path d="M {cx - 10} {cy} l 7 8 l 15 -18" '
            'fill="none" stroke="#111315" stroke-width="4.2" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
        )
    return (
        f'<path d="M {cx - 9} {cy} h 18" '
        'fill="none" stroke="#626a70" stroke-width="2.2" '
        'stroke-linecap="round"/>'
    )


def build_svg(setting_key):
    config = SETTINGS[setting_key]
    sections = []
    for row in ROWS:
        if not sections or sections[-1] != row["section"]:
            sections.append(row["section"])

    height = (
        DATA_TOP
        + len(sections) * SECTION_H
        + len(ROWS) * ROW_H
        + BOTTOM_PAD
    )
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">AscendC optimization techniques by operator</title>',
        f'<desc id="desc">{esc(config["title"])}. Check marks indicate source-level evidence in the selected final submission.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<g font-family="Arial, Helvetica, sans-serif" fill="#111315">',
        f'<text x="{TABLE_X}" y="{TITLE_Y}" font-size="29" font-weight="700">{esc(config["title"])}</text>',
        f'<text x="{TABLE_X}" y="{SUBTITLE_Y}" font-size="16" fill="#4b5359">Selected final submission source audit | all ten submitted bundles pin AscendC</text>',
        f'<line x1="{TABLE_X}" y1="{HEADER_TOP}" x2="{TABLE_RIGHT}" y2="{HEADER_TOP}" stroke="#111315" stroke-width="2"/>',
    ]

    for index, operator in enumerate(OPS):
        cx = TABLE_X + LEFT + (index + 0.5) * CELL
        cy = 218
        out.append(
            f'<text x="{cx}" y="{cy}" transform="rotate(-90 {cx} {cy})" '
            'text-anchor="middle" font-size="21" font-weight="600">'
            f'{esc(operator["label"])}</text>'
        )
    out.append(
        f'<line x1="{TABLE_X}" y1="{HEADER_BOTTOM}" x2="{TABLE_RIGHT}" '
        f'y2="{HEADER_BOTTOM}" stroke="#111315" stroke-width="1.4"/>'
    )

    y = DATA_TOP
    section_seen = set()
    for row in ROWS:
        if row["section"] not in section_seen:
            section_seen.add(row["section"])
            out.append(
                f'<rect x="{TABLE_X}" y="{y}" width="{TABLE_RIGHT - TABLE_X}" '
                f'height="{SECTION_H}" fill="#e9ecee"/>'
            )
            out.append(
                f'<text x="{TABLE_X + 18}" y="{y + 24}" font-size="18" '
                f'font-weight="700">{esc(row["section"])}</text>'
            )
            y += SECTION_H

        cy = y + ROW_H / 2
        out.append(
            f'<text x="{TABLE_X + 22}" y="{cy + 6}" font-size="17.5">'
            f'{esc(row["label"])}</text>'
        )
        values = row[config["mark"]]
        for index, present in enumerate(values):
            cx = TABLE_X + LEFT + (index + 0.5) * CELL
            out.append(check_mark(cx, cy, present))
        out.append(
            f'<line x1="{TABLE_X}" y1="{y + ROW_H}" x2="{TABLE_RIGHT}" '
            f'y2="{y + ROW_H}" stroke="#e0e3e5" stroke-width="0.8"/>'
        )
        y += ROW_H

    out.extend([
        f'<line x1="{TABLE_X}" y1="{y + 10}" x2="{TABLE_RIGHT}" y2="{y + 10}" '
        'stroke="#111315" stroke-width="1.5"/>',
        check_mark(TABLE_X + 18, y + 39, True),
        f'<text x="{TABLE_X + 40}" y="{y + 45}" font-size="15">Present in selected source</text>',
        check_mark(TABLE_X + 311, y + 39, False),
        f'<text x="{TABLE_X + 333}" y="{y + 45}" font-size="15">Not observed or not applicable</text>',
        f'<text x="{TABLE_X}" y="{y + 75}" font-size="14" fill="#4b5359">'
        'Source-level presence is not an isolated performance attribution. '
        'A/B names refer to expert-knowledge settings, not implementation language.</text>',
        '</g>',
        '</svg>',
    ])
    return "\n".join(out) + "\n"


def write_csv():
    path = HERE / "matrix.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["setting", "section", "technique"] + [op["label"] for op in OPS])
        for setting_key, config in SETTINGS.items():
            for row in ROWS:
                writer.writerow([
                    setting_key,
                    row["section"],
                    row["label"],
                    *("present" if value else "not_observed" for value in row[config["mark"]]),
                ])


def main():
    check_configs()
    check_matrix()
    write_csv()
    svg_paths = []
    for setting_key, config in SETTINGS.items():
        path = HERE / config["filename"]
        path.write_text(build_svg(setting_key), encoding="utf-8")
        svg_paths.append(path)
        print(path)

    quicklook = shutil.which("qlmanage")
    if quicklook:
        subprocess.run(
            [quicklook, "-t", "-s", str(WIDTH), "-o", str(HERE), *map(str, svg_paths)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        for source, target in zip(svg_paths, (HERE / "without-expert.png", HERE / "with-expert.png")):
            source.with_name(source.name + ".png").replace(target)
            print(target)


if __name__ == "__main__":
    main()

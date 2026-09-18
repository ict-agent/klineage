"""Run the Ascend generation experiment locally: Codex here, evaluation on 910b1.

For every (kernel, setting) unit this script builds an isolated workspace under
``experiment/ascend/generation/<kernel>/<setting>`` and runs one local Codex
session against the model provider configured in ``~/.codex/config.toml``. The
session is capped by ``--timeout``. A local usage proxy sits between Codex and
the provider, so ``events.jsonl`` carries one record per model round trip next
to the evaluation records written by ``scripts/ascend/eval.py``.

Artifacts per unit::

    <kernel>/<setting>/
    |- events.jsonl        # api_response (proxy) + evaluate (eval.py) events
    |- trace.jsonl         # codex exec --json session trace
    |- versions/versionN   # kernel snapshot per evaluation
    |- work/               # agent workspace (AGENTS.md, problems, submission)
    |- submission/         # final bundle copy
    `- result.json         # status, counters, timings

Setting names map to the task: without_memory = A, with_memory = B.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KERNELS = ("kda", "sparse_attention", "top_p", "fused_add_rmsnorm", "gqa")
SETTINGS = ("without_memory", "with_memory")
SKILLS = ("bench", "ascendc")
RUN_ROOT = REPO / "experiment" / "ascend" / "generation"
CODEX_DEFAULT = "codex"
EVAL = REPO / "scripts" / "ascend" / "eval.py"


def stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def local_python() -> Path:
    """The interpreter that owns klineage in this checkout."""

    for name in (".venv-ascend/bin/python", ".venv/bin/python"):
        candidate = REPO / name
        if candidate.is_file():
            return candidate
    return Path(sys.executable)


def rooted(path: Path) -> str:
    """Path as written from the project root; absolute if it lives outside."""

    absolute = Path(os.path.abspath(path))
    try:
        return str(absolute.relative_to(REPO))
    except ValueError:
        return str(path)


def eval_cmd(work: Path, device: str = "") -> str:
    """The evaluation gate, with every path relative to the project root."""

    command = f"{rooted(local_python())} {rooted(EVAL)} --work {rooted(work)}"
    return f"{command} --device {device}" if device else command


def path_fields(work: Path) -> dict[str, str]:
    """Path tokens for the workspace templates, all relative to the root."""

    return {
        "{work_rel}": rooted(work),
        "{root_rel}": os.path.relpath(REPO, work),
    }


def render(text: str, fields: dict[str, str]) -> str:
    """Replace every `{token}` with its rendered value."""

    for token, value in fields.items():
        text = text.replace(token, value)
    return text


def template_fields(work: Path, kernel: str, setting: str, device: str = "") -> dict[str, str]:
    """Tokens shared by AGENTS.md and the prompt; paths sit under the root."""

    expert = setting == "with_memory"
    return {
        "{kernel}": kernel,
        "{setting}": setting,
        "{eval_cmd}": eval_cmd(work, device),
        "{expert_line}": f"Expert knowledge: read `{rooted(work)}/expert/` first.\n" if expert else "",
        **path_fields(work),
    }


def provider_settings() -> tuple[str, str]:
    """(provider name, base_url) from the user's Codex config."""

    import tomllib

    data = tomllib.loads(Path("~/.codex/config.toml").expanduser().read_text(encoding="utf-8"))
    name = data.get("model_provider", "custom")
    return name, data["model_providers"][name]["base_url"]


def prepare(work: Path, kernel: str, setting: str, template: str) -> bool:
    """Reset the workspace: problems, skills, AGENTS.md, expert knowledge."""

    for name in ("submission", "work"):
        target = work.parent / name
        if target.exists():
            shutil.rmtree(target)
    (work.parent / "versions").mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True)
    shutil.copytree(REPO / "experiment" / kernel / "problems", work / "problems")

    skills = work / ".agents" / "skills"
    skills.mkdir(parents=True, exist_ok=True)
    for name in SKILLS:
        (skills / name).symlink_to(REPO / "src" / "klineage" / "skills" / name)

    expert = setting == "with_memory"
    source = REPO / "scripts" / "ascend" / "expert" / kernel
    files = [path for path in source.rglob("*") if path.is_file() and path.name != ".keep"]
    if expert:
        if not files:
            raise FileNotFoundError(
                f"expert knowledge missing for {kernel}; fill scripts/ascend/expert/{kernel}/"
            )
        shutil.copytree(source, work / "expert")

    (work / "AGENTS.md").write_text(render(template, template_fields(work, kernel, setting)),
                                    encoding="utf-8")
    return expert


def events_counts(events: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not events.is_file():
        return counts
    for line in events.read_text(encoding="utf-8").splitlines():
        try:
            name = json.loads(line).get("event")
        except ValueError:
            continue
        counts[name] = counts.get(name, 0) + 1
    return counts


def package(out: Path, work: Path) -> None:
    events = work / ".klineage" / "stats" / "events.jsonl"
    if events.is_file():
        shutil.copy2(events, out / "events.jsonl")
    versions = work / "versions"
    if versions.is_dir():
        shutil.copytree(versions, out / "versions", dirs_exist_ok=True)
    submission = work / "submission"
    if submission.is_dir():
        shutil.copytree(submission, out / "submission", dirs_exist_ok=True)
    kernel_json = work / "kernel.json"
    if kernel_json.is_file():
        shutil.copy2(kernel_json, out / "kernel.json")


def run_unit(unit: tuple[str, str], device: str, options: argparse.Namespace) -> dict:
    from klineage.logging import UsageProxy

    kernel, setting = unit
    out = Path(options.run_root) / kernel / setting
    work = out / "work"
    record = {"kernel": kernel, "setting": setting, "device": device,
              "started_at": datetime.now(UTC).isoformat()}
    started = time.monotonic()
    error = None
    try:
        template = (REPO / "scripts" / "ascend" / "agents.md.tmpl").read_text(encoding="utf-8")
        prepare(work, kernel, setting, template)
        fields = template_fields(work, kernel, setting, device)
        fields["{hours}"] = str(options.timeout // 3600)
        prompt = render((REPO / "scripts" / "ascend" / "prompt.md.tmpl").read_text(encoding="utf-8"),
                        fields)

        provider, upstream = provider_settings()
        stats = work / ".klineage" / "stats"
        with UsageProxy(upstream, directory=stats) as proxy:
            command = [
                options.codex_bin, "-a", "never", "exec",
                "-s", "danger-full-access",
                "--json", "--skip-git-repo-check",
                "--output-last-message", str(out / "final_message.txt"),
                "-C", str(work),
                "-c", f"model_providers.{provider}.base_url={proxy.url}",
                "-",
            ]
            with (out / "trace.jsonl").open("w", encoding="utf-8") as trace, \
                 (out / "stderr.log").open("w", encoding="utf-8") as stderr:
                process = subprocess.Popen(
                    command, cwd=work, stdin=subprocess.PIPE, stdout=trace, stderr=stderr,
                    text=True, start_new_session=True,
                )
                try:
                    process.communicate(prompt, timeout=options.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.communicate()
                    raise TimeoutError(f"Codex timed out after {options.timeout}s")
            if process.returncode:
                raise RuntimeError(f"Codex exited with {process.returncode}")
    except Exception as failure:  # partial evidence is still packaged below
        error = f"{type(failure).__name__}: {failure}"
        record["error"] = error

    package(out, work)
    record.update(events_counts(out / "events.jsonl"))
    record["versions"] = len(list((out / "versions").glob("version*")))
    record["duration_s"] = round(time.monotonic() - started, 1)
    record["finished_at"] = datetime.now(UTC).isoformat()
    record["status"] = "error" if error else "ok"
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", help="comma list; default all five")
    parser.add_argument("--settings", help="comma list of without_memory,with_memory")
    parser.add_argument("--devices", default="2,3,4,5", help="NPU ids, one unit at a time each")
    parser.add_argument("--timeout", type=int, default=7200, help="seconds per Codex session")
    parser.add_argument("--codex-bin", default=CODEX_DEFAULT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    options = parse_args(argv)
    options.run_root = Path(options.run_root).expanduser().resolve()
    kernels = tuple(options.kernels.split(",")) if options.kernels else KERNELS
    settings = tuple(options.settings.split(",")) if options.settings else SETTINGS
    devices = [item.strip() for item in options.devices.split(",") if item.strip()]

    units = []
    for kernel in kernels:
        if kernel not in KERNELS:
            raise SystemExit(f"unknown kernel: {kernel}")
        for setting in settings:
            if setting not in SETTINGS:
                raise SystemExit(f"unknown setting: {setting}")
            marker = options.run_root / kernel / setting / "result.json"
            if marker.is_file() and not options.retry_failed:
                print(f"skip finished: {kernel}/{setting}")
                continue
            units.append((kernel, setting))
    print(f"{len(units)} unit(s), devices {devices}")
    if options.dry_run:
        return

    from multiprocessing import Pool

    tasks = [(unit, devices[index % len(devices)], options) for index, unit in enumerate(units)]
    pool = Pool(len(devices))
    try:
        results = []
        for record in pool.starmap(run_unit, tasks):
            results.append(record)
            brief = {key: record.get(key) for key in
                     ("kernel", "setting", "device", "status", "duration_s", "versions")}
            print(json.dumps(brief), flush=True)
            (options.run_root / "status.json").write_text(
                json.dumps(results, indent=2) + "\n", encoding="utf-8")
    finally:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()

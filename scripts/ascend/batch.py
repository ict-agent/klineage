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
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KERNELS = ("kda", "sparse_attention", "top_p", "fused_add_rmsnorm", "gqa")
SETTINGS = ("without_memory", "with_memory")
#: The evaluator documentation plus the artifact spec both settings get: the
#: task's own sample for the AscendC bundle format.
SKILLS = ("bench", "ascendc")
#: Dropped from the unit copies of ``scripts/`` and ``src/``: the gate plus the
#: package it imports is all the agent needs. Everything listed here is
#: experiment metadata (settings, expert packs, CUDA transfer tooling, memory
#: prompts) that would bias the without_memory setting or leak the design.
PRUNE = (
    "scripts/apply_transfer.py",
    "scripts/ascend/README.md",
    "scripts/ascend/agents.md.tmpl",
    "scripts/ascend/batch.py",
    "scripts/ascend/task2.sh",
    "scripts/ascend/deepseek_env.sh",
    "scripts/ascend/deepseek.example.toml",
    "scripts/ascend/resume.py",
    "scripts/ascend/trace_proxy.py",
    "scripts/ascend/session_loop.py",
    "scripts/ascend/device_guard.py",
    "scripts/ascend/collect.py",
    "scripts/ascend/notes",
    "scripts/ascend/run.sh",
    "scripts/ascend/spawn.py",
    "scripts/ascend/community_baseline.py",
    "scripts/ascend/expert",
    "scripts/ascend/gen_inputs.py",
    "scripts/ascend/plot.py",
    "scripts/ascend/prompt.md.tmpl",
    "scripts/ascend/smoke_eval.py",
    "src/klineage/harness",
    "src/klineage/memory",
    "src/klineage/prompts",
    "src/klineage/skills/cuda",
    "src/klineage/skills/hip",
)
VENV_LINK = ".venv"
#: Unit workspaces live outside the checkout: Codex walks up from its cwd for
#: AGENTS.md, so a root inside the repo would expose the expert packs and the
#: CUDA transfer runs to the without_memory setting.
RUN_ROOT = Path(os.environ.get("KLINEAGE_RUN_ROOT", Path.home() / "klineage-runs"))
#: The packaged expert source lives with the with_memory artifact. The external
#: root remains a fallback for older experiment layouts.
EXPERT_ROOT = Path(os.environ.get("KLINEAGE_EXPERT_ROOT", Path.home() / "klineage-expert"))
CODEX_DEFAULT = "codex"
EVAL = REPO / "scripts" / "ascend" / "eval.py"


def stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def venv_source() -> Path:
    """The checkout's virtualenv, the interpreter that owns klineage."""

    for name in (".venv-ascend", ".venv"):
        candidate = REPO / name
        if candidate.is_dir():
            return candidate
    return Path(sys.executable).parents[1]


def scrub(path: Path, root: Path) -> None:
    """Rewrite checkout paths in the unit copy to the unit's own paths.

    Codex finds the checkout through absolute paths it reads, and the checkout
    holds the expert packs and the transferred solutions the without_memory
    setting must not see.
    """

    source = str(REPO).encode()
    for item in path.rglob("*"):
        if not item.is_file() or item.is_symlink():
            continue
        try:
            data = item.read_bytes()
        except OSError:
            continue
        if source not in data:
            continue
        try:
            text = data.decode()
        except UnicodeDecodeError:
            item.unlink()          # bytecode embeds its build path
            continue
        item.write_text(text.replace(str(REPO), str(root)), encoding="utf-8")


def install_unit(unit: Path) -> None:
    """Give the unit root its own runner, source and interpreter copies.

    Copies, never links: a symlink to the checkout is the fastest way for the
    agent to walk into the settings it is not supposed to read.
    """

    unit.mkdir(parents=True, exist_ok=True)
    skip = shutil.ignore_patterns("__pycache__", "*.pyc")
    for name in ("scripts", "src"):
        target = unit / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(REPO / name, target, symlinks=True, ignore=skip)

    for name in PRUNE:
        victim = unit / name
        if victim.is_dir():
            shutil.rmtree(victim)
        elif victim.exists():
            victim.unlink()

    venv = unit / VENV_LINK
    if venv.is_symlink() or venv.exists():
        venv.unlink() if venv.is_symlink() else shutil.rmtree(venv)
    shutil.copytree(venv_source(), venv, symlinks=True, ignore=skip)

    for name in sorted(site for site in (venv / "lib").glob("python*/site-packages")):
        editable = name / "klineage.pth"
        if editable.is_file():
            editable.write_text(str(unit / "src") + "\n", encoding="utf-8")

    scrub(unit / "scripts", unit)
    scrub(venv, unit)

    # The gate is the only measurement: keep the copies from being rewritten.
    for name in ("scripts", "src"):
        for item in (unit / name).rglob("*"):
            if item.is_file():
                item.chmod(item.stat().st_mode & ~0o222)


def rooted(path: Path, root: Path) -> str:
    """Path as written from the unit root; absolute if it lives outside."""

    absolute = Path(os.path.abspath(path))
    try:
        return str(absolute.relative_to(root))
    except ValueError:
        return str(path)


def eval_cmd(work: Path, unit: Path, device: str = "") -> str:
    """The evaluation gate, with every path relative to the unit root."""

    python = f"{VENV_LINK}/bin/python"
    gate = str(EVAL.relative_to(REPO))         # the unit copy, relative to its root
    command = f"{python} {gate} --work {rooted(work, unit)}"
    if os.environ.get("KLINEAGE_LANGUAGE") in ("triton", "ascendc"):
        command += " --language " + os.environ["KLINEAGE_LANGUAGE"]
    return f"{command} --device {device}" if device else command


def path_fields(work: Path, unit: Path) -> dict[str, str]:
    """Path tokens for the workspace templates, all relative to the unit root."""

    return {
        "{work_rel}": rooted(work, unit),
        "{root_rel}": os.path.relpath(unit, work),
    }


def render(text: str, fields: dict[str, str]) -> str:
    """Replace every `{token}` with its rendered value."""

    for token, value in fields.items():
        text = text.replace(token, value)
    return text


def expert_source(kernel: str) -> Path:
    """Find the packaged expert source for a kernel."""

    packaged = REPO / "experiment" / "ascend" / "generation" / kernel / "with_memory" / "expert"
    for source in (packaged, EXPERT_ROOT / kernel, REPO / "scripts" / "ascend" / "expert" / kernel):
        if source.is_dir():
            return source
    return packaged


def template_fields(work: Path, kernel: str, setting: str, unit: Path,
                    device: str = "", hours: int = 0) -> dict[str, str]:
    """Tokens shared by AGENTS.md and the prompt; paths sit under the unit root."""

    expert = setting == "with_memory"
    return {
        "{kernel}": kernel,
        "{target_arch}": os.environ.get("ASCEND_ARCH", "Ascend910B1"),
        "{setting}": setting,
        "{language_contract}": (
            'You must use Triton (triton-ascend) for all device computation. '
            'Set build language = "python", entry_point = "kernel.py::kernel". '
            'Python may only wrap/launch Triton kernels and allocate buffers; '
            'do not implement computation in torch/torch-npu, AscendC, C++, or call prebuilt operators.'
            if os.environ.get('KLINEAGE_LANGUAGE') == 'triton' else
            'You must use AscendC for all device computation. Set build language = "ascendc", '
            'entry_point = "kernel.asc::kernel". Implement a native AscendC kernel and the required '
            'C++/pybind wrapper. Do not use Triton, torch/torch-npu computation, or prebuilt operators. '
            'Read work/.agents/skills/ascendc/SKILL.md for the ABI, compiler and current-stream requirements.'
            if os.environ.get('KLINEAGE_LANGUAGE') == 'ascendc' else
            'Implementation language is your choice: AscendC or a Python wrapper launching a device kernel.'
        ),
        "{hours}": str(hours),
                "{eval_cmd}": eval_cmd(work, unit, device),
        "{expert_line}": f"Expert knowledge: read `{rooted(work, unit)}/expert/` first.\n" if expert else "",
        **path_fields(work, unit),
    }


def provider_config(options):
    import tomllib

    path = options.provider_config.expanduser().resolve()
    data = tomllib.loads(path.read_text())
    model = options.model or data.get('model', '')
    if 'deepseek' not in model.lower():
        raise ValueError('Provide the DeepSeek-V4.1-Flash model configuration; refusing to use another model.')
    name = data.get('model_provider')
    provider = dict(data.get('model_providers', {}).get(name, {}))
    if not provider.get('base_url'):
        raise ValueError('The provider must define a Responses-compatible base_url.')
    if provider.get('wire_api', 'responses') != 'responses':
        raise ValueError('Use a Responses-compatible endpoint or adapter.')
    key = provider.get('env_key')
    if key and not os.environ.get(key):
        raise ValueError(f'Missing provider credential environment variable: {key}')
    return model, provider, path


def write_config(home, model, provider, source):
    import tomli_w

    home.mkdir(parents=True, exist_ok=True)
    config = dict(model=model, model_provider='experiment', model_context_window=1000000,
                  model_auto_compact_token_limit=900000, approval_policy='never',
                  sandbox_mode='danger-full-access', web_search='disabled',
                  features={'multi_agent': False}, model_providers={'experiment': provider})
    target = home/'config.toml'
    target.write_text(tomli_w.dumps(config))
    target.chmod(0o600)
    auth = source.parent/'auth.json'
    if provider.get('requires_openai_auth') and auth.is_file():
        shutil.copy2(auth, home/'auth.json')
        (home/'auth.json').chmod(0o600)
    public = dict(config)
    public['model_providers'] = {'experiment': {k: v for k, v in provider.items()
                                if k in ('name', 'base_url', 'wire_api', 'env_key', 'requires_openai_auth')}}
    (home.parent/'config.public.toml').write_text(tomli_w.dumps(public))


def provider_settings():
    # Legacy resume entry; new runs require an explicit DeepSeek configuration.
    import tomllib
    data = tomllib.loads(Path('~/.codex/config.toml').expanduser().read_text())
    name = data.get('model_provider', 'custom')
    return name, data['model_providers'][name]['base_url']


def prepare(work: Path, kernel: str, setting: str, template: str, unit: Path,
            hours: int, device: str = "") -> bool:
    """Reset the workspace: problems, skills, AGENTS.md, expert knowledge."""

    # baseline.json survives: it is a property of the device, not of the run.
    for name in ("submission", "work", "versions"):
        target = work.parent / name
        if target.exists():
            shutil.rmtree(target)
    for name in ("events.jsonl", "kernel.json", "result.json", "final_message.txt",
                 "status.json"):
        stale = work.parent / name
        if stale.is_file():
            stale.unlink()
    (work.parent / "versions").mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True)
    shutil.copytree(REPO / "experiment" / kernel / "problems", work / "problems")

    skills = work / ".agents" / "skills"
    skills.mkdir(parents=True, exist_ok=True)
    for name in SKILLS:
        # The unit's own copy, not the checkout: a symlink into the checkout is
        # a path the agent can follow up into the expert packs.
        (skills / name).symlink_to(unit / "src" / "klineage" / "skills" / name)

    expert = setting == "with_memory"
    source = expert_source(kernel)
    files = [path for path in source.rglob("*") if path.is_file() and path.name != ".keep"]
    if expert:
        if not files:
            raise FileNotFoundError(
                f"expert knowledge missing for {kernel}; fill "
                f"experiment/ascend/generation/{kernel}/with_memory/expert/"
            )
        shutil.copytree(source, work / "expert")

    fields = template_fields(work, kernel, setting, unit, device, hours=hours)
    (work / "AGENTS.md").write_text(render(template, fields), encoding="utf-8")
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
    from trace_proxy import AuditProxy

    kernel, setting = unit
    out = Path(options.run_root) / kernel / setting
    work = out / "work"
    record = {"kernel": kernel, "setting": setting, "device": device,
              "started_at": datetime.now(UTC).isoformat()}
    started = time.monotonic()
    hours = options.timeout // 3600
    error = None
    try:
        template = (REPO / "scripts" / "ascend" / "agents.md.tmpl").read_text(encoding="utf-8")
        if out.exists() and any((out/name).exists() for name in ('trace.jsonl', 'result.json', 'status.json')):
            archive = out.parent/'archive'
            archive.mkdir(exist_ok=True)
            out.rename(archive/f"{setting}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}")
        install_unit(out)
        prepare(work, kernel, setting, template, out, hours, device)
        fields = template_fields(work, kernel, setting, out, device, hours)
        prompt = render((REPO / "scripts" / "ascend" / "prompt.md.tmpl").read_text(encoding="utf-8"),
                        fields)

        model, provider, source_config = provider_config(options)
        upstream = provider['base_url']
        record.update(ascend_arch=os.environ.get('ASCEND_ARCH'), model=model, model_label='DeepSeek-V4.1-Flash',
                      model_context_window=1000000, timeout_s=options.timeout,
                      required_language=os.environ.get("KLINEAGE_LANGUAGE", "unrestricted"),
                      host=os.environ.get('KLINEAGE_HOST', '910b1'),
                      container=os.environ.get('KLINEAGE_CONTAINER', 'vllm0.23.0-zcj'))
        (out/'prompt.txt').write_text(prompt)
        shutil.copy2(work/'AGENTS.md', out/'AGENTS.session.md')
        stats = work / ".klineage" / "stats"
        with AuditProxy(upstream, stats) as proxy:
            provider['base_url'] = proxy.url
            home = out/'codex-home'
            write_config(home, model, provider, source_config)
            environment = dict(os.environ, CODEX_HOME=str(home))
            from session_loop import run_session
            record.update(run_session(out, work, environment, options.codex_bin, prompt, options.timeout))
    except Exception as failure:  # partial evidence is still packaged below
        error = f"{type(failure).__name__}: {failure}"
        record["error"] = error

    package(out, work)
    from session_loop import audit_span
    record.update(audit_span(out, options.timeout))
    sessions = sorted((out/'codex-home/sessions').rglob('*.jsonl'))
    if len(sessions) == 1:
        shutil.copy2(sessions[0], out/'session.jsonl')
    elif sessions:
        shutil.copytree(out/'codex-home/sessions', out/'sessions', dirs_exist_ok=True)
    api = work/'.klineage/stats/api'
    if api.is_dir():
        shutil.copytree(api, out/'api', dirs_exist_ok=True)
    candidates = []
    for version in (out/'versions').glob('version*'):
        result_path = version/'result.json'
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        if all(result.get(k) for k in ('compile_passed', 'correctness_passed', 'profile_passed')) and result.get('latency_ms'):
            candidates.append((result['latency_ms'], version, result))
    if candidates:
        latency, version, measured = min(candidates, key=lambda item: item[0])
        if (out/'submission').exists():
            (out/'submission').rename(out/'agent_submission')
        shutil.copytree(version/'submission', out/'submission')
        shutil.copy2(version/'kernel.json', out/'kernel.json')
        record.update(selected_version=version.name, latency_ms=latency,
                      speedup=measured.get('speedup'), correctness_passed=True)
    else:
        record['correctness_passed'] = False
    record.update(events_counts(out / "events.jsonl"))
    record["versions"] = len(list((out / "versions").glob("version*")))
    record["duration_s"] = round(time.monotonic() - started, 1)
    record["finished_at"] = datetime.now(UTC).isoformat()
    record["status"] = "error" if error else ("ok" if record.get("response_span_passed") and record.get("iteration_evidence") else "incomplete")
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
    parser.add_argument('--provider-config', type=Path,
                        default=Path(os.environ.get('KLINEAGE_PROVIDER_CONFIG', '~/.codex/config.toml')))
    parser.add_argument('--model', help='Exact provider model ID for DeepSeek-V4.1-Flash')
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

    provider_config(options)  # Fail before starting any unit if model/auth is missing.
    if not devices or not 0 < options.timeout <= 7200:
        raise SystemExit('Supply devices and a timeout in (0, 7200].')
    from multiprocessing import Pool

    results = []
    # Finish both settings before the next kernel. Round-robin assignment to a
    # generic pool can otherwise start two units on the same NPU.
    for kernel in kernels:
        group = [unit for unit in units if unit[0] == kernel]
        for start in range(0, len(group), len(devices)):
            wave = group[start:start+len(devices)]
            tasks = [(unit, devices[index], options) for index, unit in enumerate(wave)]
            from device_guard import check_devices
            check_devices(os.environ.get('KLINEAGE_HOST', '910b1'), devices[:len(wave)])
            with Pool(len(tasks)) as pool:
                records = pool.starmap(run_unit, tasks)
            results.extend(records)
            for record in records:
                print(json.dumps(record), flush=True)
            (options.run_root/'status.json').write_text(json.dumps(results, indent=2)+'\n')
        if group:
            subprocess.run([sys.executable, str(REPO/'scripts/ascend/collect.py'),
                            '--kernel', kernel, '--root', str(options.run_root),
                            '--out', str(REPO/'experiment/ascend/generation')], check=True)
            for axis in ('seconds', 'tokens'):
                subprocess.run([sys.executable, str(REPO/'scripts/ascend/plot.py'),
                                '--root', str(options.run_root), '--out',
                                str(REPO/'experiment/ascend/generation/next-plots'),
                                '--x', axis, '--y', 'speedup'], check=True)



if __name__ == '__main__':
    main()

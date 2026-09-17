"""Run the memory-transfer Apply comparison across every experiment problem.

Every problem starts from its torch-reference reproduction: the `apply/baseline`
naive kernel for held-out problems, or `init_memory/init` for the extraction
sources. Each runs one Apply round twice, without and with memory, on the same
model and identical inputs, so the only difference is the mounted cards.

fmha, kda and fused_add_rmsnorm are fully uploaded, and sparse_attention's
without_memory group with them: those units are listed in COMPLETED and run no
round here. Their cards still feed the pool that every other problem mounts, and
sparse_attention's with_memory group is still pending.

    python scripts/apply_transfer.py --compare

Each run writes to its own `experiment/transfer_<UTC timestamp>` directory, so
runs never overwrite one another. Pass --resume-from an earlier run to keep the
units it finished and reuse its card pool; pass --output to place the run
somewhere else entirely.

Card sources mount their own extracted memory; every other problem mounts the
pooled pool. Progress is a tqdm bar; results and evidence stay under --output.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from tqdm import tqdm

from klineage.artifact.kernel import Kernel, load_kernel, save_kernel
from klineage.artifact.problem import load_trace
from klineage.cli.optimize import optimize
from klineage.constants import STATS_DIRECTORY, RunKind

# Extraction sources: they hold memory and start from their own Init kernel.
CARD_SOURCES = ("fmha", "fused_add_rmsnorm", "kda", "sparse_attention", "topk")
# Held-out problems: no extraction, so they start from a frozen naive baseline.
HELD_OUT = ("block_sparse", "gdn", "gqa", "residual_layernorm", "top_p")
TARGETS = (*CARD_SOURCES, *HELD_OUT)
GROUPS = ("without_memory", "with_memory")

#: Every unit the comparison covers: one per (problem, group), two per problem.
UNITS = tuple((problem, group) for problem in TARGETS for group in GROUPS)
#: Units already measured and uploaded under experiment/_transfer. A listed unit
#: runs no round here. Its problem's cards still feed the pool like any other, and
#: a partner group may still be pending; only this one unit is skipped.
COMPLETED = frozenset((
    "fmha/without_memory",
    "fmha/with_memory",
    "fused_add_rmsnorm/without_memory",
    "fused_add_rmsnorm/with_memory",
    "kda/without_memory",
    "kda/with_memory",
    "sparse_attention/without_memory",
))
#: Units this run performs, in problem order with each problem's units adjacent.
PENDING = tuple(unit for unit in UNITS if f"{unit[0]}/{unit[1]}" not in COMPLETED)
#: Problems with at least one pending unit, in target order.
RUN_PROBLEMS = tuple(dict.fromkeys(problem for problem, _ in PENDING))
#: Problems where a completed group shares a problem with a pending one. Comparing
#: such a pair needs the completed group's kernel, which lives only in the uploaded
#: tree; place it at <output>/<problem>/<group>/ before running with --compare.
MIXED = tuple(
    problem
    for problem in RUN_PROBLEMS
    if any(f"{problem}/{group}" in COMPLETED for group in GROUPS)
)

#: Cards taken per source into the pool; 0 or less takes every card.
CARDS_PER_SOURCE = 34
SKILL_FILE = "SKILL.md"
KERNEL_FILE = "kernel.json"
RESULT_FILE = "result.json"
PROBLEMS_DIR = "problems"
DEFINITIONS_DIR = "definitions"
MEMORY_DIR = "memory"
INIT_DIR = Path("init_memory") / "init"
BASELINE_DIR = Path("apply") / "baseline"
MISSING = "-"

DEFAULT_MODEL = "DeepSeek-V4.1-Flash"
DEFAULT_BASE_URL = "https://aiping.cn/api/v1"
DEFAULT_KEY_ENV = "AIPING_API_KEY"
#: Two hours per Apply round unless --timeout says otherwise.
DEFAULT_TIMEOUT = 7200
#: Environment variable selecting the visible accelerator.
CUDA_DEVICES = "CUDA_VISIBLE_DEVICES"
#: Environment variable selecting how CUDA numbers the visible accelerators.
CUDA_ORDER = "CUDA_DEVICE_ORDER"
#: Number cards by PCI bus id, so an ordinal names the card nvidia-smi lists.
#: Without it CUDA's own FASTEST_FIRST order puts L20s at 0 and 1 on this host,
#: and a run pinned to a "H100" ordinal silently lands on an L20.
CUDA_ORDER_PCI = "PCI_BUS_ID"
#: Default device ordinals when --devices is omitted.
DEFAULT_DEVICES = ("0", "1", "2", "3")
#: Directory prefix for a run's own output, below the experiment directory.
RUN_PREFIX = "transfer_"
#: Context the model accepts. The endpoint advertises up to 1M.
CONTEXT_WINDOW = 1000000
#: Compact only near the window's end, well after one round's working set.
AUTO_COMPACT_TOKENS = 900000

# Bashrc assignment: export ANTHROPIC_AUTH_TOKEN="..." with optional quotes.
BASH_ASSIGNMENT = r"^\s*(?:export\s+)?{name}=[\"']?([^\"'\s]+)"
BASHRC = Path("~/.bashrc")


@dataclass(frozen=True, slots=True)
class Provider:
    """One OpenAI-compatible endpoint reachable through Codex.

    `base_url` is where Codex sends its requests, so pointing it at a usage proxy
    is what puts the proxy in path. `key_env` names the variable Codex resolves
    the key from; the caller publishes that variable before the run.
    """

    model: str
    base_url: str
    key_env: str
    key: str

    def config_flags(self) -> list[str]:
        # A distinct provider keeps the host's settings out of the request.
        name = "klineage"
        return [
            "-c", "model_provider=" + name,
            "-c", f"model_providers.{name}.name={name}",
            "-c", f"model_providers.{name}.base_url={self.base_url}",
            "-c", f"model_providers.{name}.env_key={self.key_env}",
            "-c", f"model_providers.{name}.wire_api=responses",
            "-c", f"model_providers.{name}.requires_openai_auth=false",
            # This model ships no metadata, so Codex would fall back to a small
            # window and compact repeatedly. Set both the window and the compact
            # threshold, which otherwise follows the fallback.
            "-c", f"model_context_window={CONTEXT_WINDOW}",
            "-c", f"model_auto_compact_token_limit={AUTO_COMPACT_TOKENS}",
        ]


def bashrc_value(name: str) -> str | None:
    """Read one assignment from the shell profile; the key never hits the repo."""

    path = BASHRC.expanduser()
    if not path.is_file():
        return None
    match = re.search(BASH_ASSIGNMENT.format(name=re.escape(name)), path.read_text(), re.M)
    return match.group(1) if match else None


def resolve_provider(args: argparse.Namespace) -> Provider:
    key = args.api_key or os.environ.get(args.key_env) or bashrc_value("ANTHROPIC_AUTH_TOKEN")
    if not key:
        raise SystemExit(
            f"No API key: set {args.key_env}, pass --api-key, or define "
            "ANTHROPIC_AUTH_TOKEN in ~/.bashrc"
        )
    return Provider(args.model, args.base_url, args.key_env, key)


def runner_class(provider: Provider, effort: str):
    """Bind the endpoint into the runner Action instantiates internally.

    The runner resolves the provider itself instead of having a subclass append
    flags, so a proxy URL reaches Codex by the one path that sets config.
    """

    from klineage.harness.codex_runner import CodexRunner

    class ProviderRunner(CodexRunner):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("reasoning_effort", effort)
            kwargs.setdefault("api_base_url", provider.base_url)
            kwargs.setdefault("api_key_env", provider.key_env)
            super().__init__(*args, **kwargs)

        def command(self, final_message_path: Path) -> list[str]:
            command = super().command(final_message_path)
            # Global flags must precede the subcommand; --model follows it.
            at = command.index("exec") + 1
            command[at:at] = ["--model", provider.model]
            return command

    return ProviderRunner


def usage_proxy(provider: Provider, stats: Path, enabled: bool):
    """Serve the provider through a usage-recording proxy, or pass through.

    One proxy per unit keeps model spend in that unit's own stats file; a shared
    proxy could still record the same events, but under a directory no single
    unit owns.
    """

    if not enabled:
        return contextlib.nullcontext(provider)

    from klineage.logging import UsageProxy

    proxy = UsageProxy(provider.base_url, directory=stats)
    return _ProxiedProvider(provider, proxy)


class _ProxiedProvider:
    """Point one provider at a running proxy for the body of a `with` block."""

    def __init__(self, provider: Provider, proxy):
        self.provider = provider
        self.proxy = proxy

    def __enter__(self) -> Provider:
        self.proxy.__enter__()
        os.environ[self.provider.key_env] = self.provider.key
        return replace(self.provider, base_url=self.proxy.url)

    def __exit__(self, *exc) -> None:
        self.proxy.__exit__(*exc)


def source_memory(experiment: Path, problem: str, limit: int) -> list[Path]:
    """The card directories of one extraction problem, in id order.

    A non-positive limit takes every card.
    """

    cards = sorted((experiment / problem / INIT_DIR.parent / MEMORY_DIR).glob(f"*/{SKILL_FILE}"))
    selected = cards if limit <= 0 else cards[:limit]
    return [card.parent for card in selected]


def pool_memory(experiment: Path, destination: Path, limit: int) -> list[dict]:
    """Copy the first card directories of each source into one shared pool."""

    destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    for source in CARD_SOURCES:
        for card in source_memory(experiment, source, limit):
            target = destination / source / card.name / SKILL_FILE
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(card / SKILL_FILE, target)
            manifest.append({
                "source": source,
                "card": card.name,
                "path": str(target),
                "pooled": str(card / SKILL_FILE),
            })
    return manifest


def materialize_start(experiment: Path, problem: str, source: Path) -> Path:
    """Give one problem directory a kernel.json, so optimize() can start there.

    `experiment/<p>/problems/` holds the definition, workload and inputs but no
    kernel, and the extraction workdir that has one also carries the expert
    implementation Apply must not read. Build the kernel here instead, resolving
    the problem through the local definition so the workload inputs point at the
    local copies. Idempotent: an existing kernel.json is kept.
    """

    problems = experiment / problem / PROBLEMS_DIR
    if (problems / KERNEL_FILE).is_file():
        return problems

    kernel = load_kernel(source)
    trace = load_trace(problems / DEFINITIONS_DIR / f"{problem}.json")

    # The local definition is authoritative: it carries the current name and the
    # workload whose input paths resolve under this directory.
    save_kernel(
        replace(
            kernel,
            problem=replace(
                kernel.problem,
                name=trace.definition["name"],
                definition=trace.definition,
                workload=trace.workload,
            ),
            source_files={},
            compile_flags=(),
            validation=None,
        ),
        problems,
    )
    return problems


def source_kernel(experiment: Path, problem: str) -> Path:
    """The torch-reference reproduction every problem optimizes away from."""

    for candidate in (experiment / problem / BASELINE_DIR, experiment / problem / INIT_DIR):
        if (candidate / KERNEL_FILE).is_file():
            return candidate
    raise SystemExit(f"No torch-reference start kernel for {problem}")


def memory_for(problem: str, pool: Path) -> Path:
    """Card sources mount their own slice of the pool; others mount all of it.

    The pool stores each source under its own name, so a source's slice holds
    exactly the same cards the pool contributed for it, at the same byte count.
    """

    own = pool / problem
    return own if problem in CARD_SOURCES and own.is_dir() else pool


def apply_round(
    start: Path,
    memory: Path | None,
    run_dir: Path,
    *,
    timeout: int,
    device: str,
) -> Kernel:
    """Run exactly one Apply round on one device; the workdir must not exist yet.

    Pinning CUDA_VISIBLE_DEVICES reaches every descendant: the codex process, the
    evaluation worker it spawns, and the compiler and profiler under that worker.
    """

    previous = os.environ.get(CUDA_DEVICES)
    os.environ[CUDA_DEVICES] = device
    os.environ[CUDA_ORDER] = CUDA_ORDER_PCI
    try:
        return optimize(
            start,
            memory_dir=memory,
            workdir=run_dir,
            max_apply_step=1,
            enable_verifier=False,
            timeout=timeout,
            max_retries=0,
        )
    finally:
        if previous is None:
            os.environ.pop(CUDA_DEVICES, None)
        else:
            os.environ[CUDA_DEVICES] = previous


def summarize(before: Kernel, after: Kernel) -> dict:
    return {
        "name_before": before.name,
        "name_after": after.name,
        "fingerprint_before": before.fingerprint,
        "fingerprint_after": after.fingerprint,
        "changed": before.fingerprint != after.fingerprint,
        "validation": after.validation.to_dict() if after.validation else None,
    }


def compare(reference: Path, candidate: Path, work: Path) -> dict:
    """Measure one kernel against a shared reference on identical inputs."""

    from klineage.harness import evaluate

    if work.exists():  # a run per output directory; never reuse stale evidence
        shutil.rmtree(work)
    work.mkdir(parents=True)
    result = evaluate(load_kernel(candidate), work, reference=load_kernel(reference))
    return result.to_dict()


def write_report(output: Path, runs: dict, comparison: dict, provider: Provider) -> None:
    lines = [
        "# Memory transfer: Apply across experiment problems",
        "",
        f"Model: `{provider.model}` via `{provider.base_url}`. "
        "One Apply round per group, no verifier; paired CUPTI timing on identical inputs.",
        "",
        "Source problems mount their own extracted cards; held-out problems mount the "
        "shared pool. Source rows therefore reuse cards derived from their own kernel.",
        "",
        "Units already measured and uploaded to the repository are marked `done` below "
        "and were not rerun; `pending` units are the ones this run covers.",
        "",
        "| Problem | Unit | State | Without memory (us) | With memory (us) | Speedup |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    # compare() measures candidate (with_memory) against reference (without_memory),
    # so latency_ms is the guided group and reference_latency_ms the baseline.
    latency, speedup = {}, {}
    for problem, _ in UNITS:
        row = comparison.get(f"{problem}/with_memory", {})
        guided = None if "error" in row else row.get("latency_ms")
        baseline = None if "error" in row else row.get("reference_latency_ms")
        latency[(problem, "with_memory")] = f"{guided * 1000:.3f}" if guided else "—"
        latency[(problem, "without_memory")] = (
            f"{baseline * 1000:.3f}" if baseline else "—"
        )
        speedup[problem] = f"{baseline / guided:.3f}x" if guided and baseline else "—"
    for problem, group in UNITS:
        key = f"{problem}/{group}"
        state = "done" if key in COMPLETED else "pending"
        lines.append(
            f"| {problem} | {group} | {state} "
            f"| {latency[(problem, 'without_memory')]} "
            f"| {latency[(problem, 'with_memory')]} | {speedup[problem]} |"
        )
    lines.extend(["", "## Round outcomes", "",
                  "| Problem | Origin | Group | State | Changed | Kernel |",
                  "| --- | --- | --- | --- | --- | --- |"])
    for problem, group in UNITS:
        origin = "source" if problem in CARD_SOURCES else "held-out"
        key = f"{problem}/{group}"
        state = "done" if key in COMPLETED else "pending"
        record = runs.get(key, {})
        changed = "yes" if record.get("changed") else "no"
        name = record.get("name_after", record.get("error", "—"))
        lines.append(
            f"| {problem} | {origin} | {group} | {state} | {changed} | {name} |"
        )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")


def tasks_for(experiment: Path, pool: Path) -> list[tuple[str, str, Path, Path | None]]:
    """Every pending (problem, group, start, memory) unit, pairs adjacent."""

    units = []
    for problem, group in PENDING:
        start = experiment / problem / PROBLEMS_DIR
        memory = memory_for(problem, pool)
        units.append((problem, group, start, memory if group == "with_memory" else None))
    return units


def applied_step(output: Path, problem: str, group: str) -> Path:
    """The round-0 step directory of one unit.

    `apply_steps` names it `<workdir>/apply/<step>`, and each round runs with that
    directory as its workspace, so the kernel and the measurement timeline both
    live beneath it.
    """

    return output / problem / group / RunKind.APPLY / "0"


def completed_record(run_dir: Path) -> dict | None:
    """The record of a unit that already ran, or None when it must run.

    `result.json` is written once, when an attempt ends, so its presence is the
    completion marker. A kernel on disk is not: a round cut off mid-flight can
    leave one behind, and taking that as done would skip the round that never
    finished.
    """

    path = run_dir / RESULT_FILE
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def one_unit(
    unit: tuple[str, str, Path, Path | None],
    *,
    output: Path,
    timeout: int,
    device: str,
    provider: Provider,
    effort: str,
    proxy: bool,
) -> tuple[str, dict]:
    """Run one problem/group pair on one pinned device and return its record.

    Runs in its own process: CUDA_VISIBLE_DEVICES is process state, so threads
    would race on it. The pin reaches every descendant of this process.

    `provider` is rebuilt here around this unit's own stats directory and usage
    proxy, because the runner is constructed inside this process.
    """

    os.environ[CUDA_DEVICES] = device
    os.environ[CUDA_ORDER] = CUDA_ORDER_PCI
    problem, group, start, memory_dir = unit
    run_dir = output / problem / group

    # A unit that already recorded a result is left untouched.
    done = completed_record(run_dir)
    if done is not None:
        done.setdefault("device", device)
        return f"{problem}/{group}", done

    before = load_kernel(start)
    record = {
        "problem": problem,
        "group": group,
        "start": str(start),
        "memory": str(memory_dir) if memory_dir else None,
        "own_memory": problem in CARD_SOURCES,
        "device": device,
        "card": card_name(device),
        "stats": str(unit_stats(output, problem, group)),
    }

    stats = unit_stats(output, problem, group)
    try:
        with usage_proxy(provider, stats, proxy) as bound:
            bind_runner(bound, effort)
            after = apply_round(start, memory_dir, run_dir, timeout=timeout, device=device)
        record.update(summarize(before, after))
    except Exception as error:  # keep the pair running; record the failure
        record.update({"error": f"{type(error).__name__}: {error}"})
    (run_dir / RESULT_FILE).parent.mkdir(parents=True, exist_ok=True)
    (run_dir / RESULT_FILE).write_text(json.dumps(record, indent=2) + "\n")
    return f"{problem}/{group}", record


def unit_stats(output: Path, problem: str, group: str) -> Path:
    """Where one unit's measurement timeline lands.

    The runner derives it from its workspace as `<workdir>/.klineage/stats`, and
    the workspace is the step directory `completed_run` reads the kernel from, so
    both are built from the same path. Round 0 is the only step this script runs.
    """

    return applied_step(output, problem, group) / STATS_DIRECTORY


def bind_runner(provider: Provider, effort: str) -> None:
    """Point Action's internally constructed runner at this unit's endpoint."""

    import klineage.action.action as action_module

    action_module.CodexRunner = runner_class(provider, effort)


def device_worker(
    queue,
    results,
    *,
    output: Path,
    timeout: int,
    provider: Provider,
    effort: str,
    proxy: bool,
) -> None:
    """Consume units until the queue drains, one at a time on this device."""

    while True:
        item = queue.get()
        if item is None:
            return
        index, unit, device = item
        try:
            key, record = one_unit(
                unit,
                output=output,
                timeout=timeout,
                device=device,
                provider=provider,
                effort=effort,
                proxy=proxy,
            )
        except Exception as error:  # a crashed unit must not kill the worker
            key = f"{unit[0]}/{unit[1]}"
            record = {"problem": unit[0], "group": unit[1],
                      "error": f"{type(error).__name__}: {error}", "device": device}
        results.put((index, key, record))


def timestamped_output(parent: Path) -> Path:
    """A fresh run directory under `parent`, named for the moment it starts.

    Every run gets its own directory, so a later run cannot overwrite an earlier
    one's evidence and two runs are never confused for each other. The stamp is
    UTC to the second; a run takes hours, so a collision would mean two started
    in the same second.
    """

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return parent / f"{RUN_PREFIX}{stamp}"


def run(args: argparse.Namespace) -> None:
    experiment = Path(args.experiment).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else timestamped_output(experiment)
    )
    output.mkdir(parents=True, exist_ok=True)

    # A resumed run takes its finished units from an earlier run's directory. The
    # work still happens here, so this directory stays self-contained.
    resumed = (
        Path(args.resume_from).expanduser().resolve()
        if args.resume_from is not None
        else None
    )

    provider = resolve_provider(args)
    os.environ[provider.key_env] = provider.key

    memory = output / MEMORY_DIR
    if resumed is not None and (resumed / MEMORY_DIR).is_dir():
        shutil.copytree(resumed / MEMORY_DIR, memory, dirs_exist_ok=True)
        manifest = json.loads((resumed / "memory-manifest.json").read_text())
    else:
        manifest = pool_memory(experiment, memory, args.cards_per_source)
    (output / "memory-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def carry_over(problem: str, group: str) -> None:
        """Copy a resumed unit's result and measured kernel into this run.

        The kernel is what a paired comparison reads, so a skipped unit that
        left it behind can still be reported against its partner.
        """

        run_dir = output / problem / group
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resumed / problem / group / RESULT_FILE, run_dir / RESULT_FILE)
        earlier = applied_step(resumed, problem, group) / KERNEL_FILE
        if earlier.is_file():
            worked = applied_step(output, problem, group)
            worked.mkdir(parents=True, exist_ok=True)
            shutil.copy2(earlier, worked / KERNEL_FILE)

    # Each problem directory needs its own kernel.json before Apply can start.
    units = []
    for problem, group, start, memory_dir in tasks_for(experiment, memory):
        if resumed is not None and completed_record(resumed / problem / group):
            carry_over(problem, group)
            continue
        units.append((problem, group, materialize_start(experiment, problem, source_kernel(experiment, problem)), memory_dir))

    # One worker process per device; each takes the next free unit, so a slow
    # problem never blocks a device that could start another. The ordinals are
    # resolved before any round starts, so a wrong or mixed set stops the run
    # instead of quietly pairing two groups across different hardware.
    device_list = args.devices
    print(f"Devices ({CUDA_ORDER}={CUDA_ORDER_PCI}):")
    check_devices(device_list)
    queue: multiprocessing.Queue = multiprocessing.Queue()
    results_queue: multiprocessing.Queue = multiprocessing.Queue()
    for index, unit in enumerate(units):
        queue.put((index, unit, device_list[index % len(device_list)]))
    for _ in device_list:
        queue.put(None)

    workers = [
        multiprocessing.Process(
            target=device_worker,
            args=(queue, results_queue),
            kwargs={
                "output": output,
                "timeout": args.timeout,
                "provider": provider,
                "effort": args.reasoning_effort,
                "proxy": not args.no_usage_proxy,
            },
        )
        for _ in device_list
    ]
    for worker in workers:
        worker.start()

    ordered: list[dict | None] = [None] * len(units)
    with tqdm(total=len(units), desc=provider.model, unit="run") as bar:
        for _ in range(len(units)):
            index, key, record = results_queue.get()
            ordered[index] = record
            state = "changed" if record.get("changed") else (
                "error" if "error" in record else MISSING)
            bar.set_postfix_str(f"{key} [{record.get('device')}] {state}")
            bar.update(1)
    for worker in workers:
        worker.join()

    results = {
        f"{record['problem']}/{record['group']}": record
        for record in ordered
        if record is not None
    }

    (output / "runs.json").write_text(json.dumps(results, indent=2) + "\n")

    # Compare every problem whose pair this run produced, including a pending
    # group whose partner came from a completed unit.
    comparison = {}
    if args.compare:
        missing = [
            f"{problem}/{group}"
            for problem in MIXED
            for group in GROUPS
            if f"{problem}/{group}" in COMPLETED
            and not (output / problem / group / RunKind.APPLY / "0" / KERNEL_FILE).is_file()
        ]
        if missing:
            print(
                f"Compare skipped for {', '.join(sorted(MIXED))}: staged kernel missing for "
                f"{', '.join(missing)}. Copy those groups from experiment/_transfer first."
            )
        with tqdm(RUN_PROBLEMS, desc="compare", unit="problem") as bar:
            for problem in bar:
                reference = output / problem / "without_memory"
                candidate = output / problem / "with_memory"
                bar.set_postfix_str(problem)
                if not (reference / RunKind.APPLY / "0" / KERNEL_FILE).is_file() or not (
                    candidate / RunKind.APPLY / "0" / KERNEL_FILE
                ).is_file():
                    failure = {"error": "missing staged kernel for one group"}
                    for group in GROUPS:
                        comparison[f"{problem}/{group}"] = failure
                    continue
                try:
                    measured = compare(
                        reference, candidate, output / problem / "comparison"
                    )
                    for group in GROUPS:
                        comparison[f"{problem}/{group}"] = measured
                except Exception as error:
                    failure = {"error": f"{type(error).__name__}: {error}"}
                    for group in GROUPS:
                        comparison[f"{problem}/{group}"] = failure
        (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")

    write_report(output, results, comparison, provider)
    failed = [k for k, v in results.items() if "error" in v]
    print(f"Apply runs: {len(results) - len(failed)}/{len(results)} completed; failures: {failed or 'none'}")


def card_name(device: str) -> str | None:
    """The card a pinned ordinal names, or None when it cannot be identified.

    Runs a child process, because CUDA_VISIBLE_DEVICES is read once at runtime
    start and cannot be probed in-process for several ordinals at a time.
    """

    program = (
        "import torch;"
        "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
    )
    environment = {
        **os.environ,
        CUDA_DEVICES: device,
        CUDA_ORDER: CUDA_ORDER_PCI,
    }
    try:
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=120,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    name = result.stdout.strip()
    return name or None


def check_devices(device_list: tuple[str, ...]) -> dict[str, str]:
    """Report each ordinal's card, refusing a set that spans more than one model.

    Pairing a with-memory group against a without-memory one only means something
    when both run on the same hardware, so a mixed set is rejected before any
    round starts.
    """

    found = {}
    for device in device_list:
        name = card_name(device)
        if name is None:
            raise SystemExit(
                f"cannot identify the card behind --devices {device}; "
                "set CUDA_VISIBLE_DEVICES yourself to check, or pass a valid ordinal"
            )
        found[device] = name
        print(f"  device {device}: {name}")
    models = set(found.values())
    if len(models) > 1:
        listing = ", ".join(f"{d}={n}" for d, n in found.items())
        raise SystemExit(
            f"--devices spans more than one card model ({listing}); a paired "
            "comparison across models is not comparable, so pick one model"
        )
    return found


def devices(value: str) -> tuple[str, ...]:
    """Parse a comma-separated device list, rejecting blanks and duplicates."""

    items = tuple(part.strip() for part in value.split(",") if part.strip())
    if not items:
        raise argparse.ArgumentTypeError("at least one device is required")
    if len(set(items)) != len(items):
        raise argparse.ArgumentTypeError(f"duplicate devices: {value!r}")
    return items


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default="experiment")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Where this run writes. Omit for a fresh <experiment>/transfer_<UTC "
            "timestamp> directory, so runs never overwrite each other"
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "An earlier run's directory. Units it recorded a result for are "
            "carried over instead of run again, along with its card pool"
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--key-env", default=DEFAULT_KEY_ENV)
    parser.add_argument("--api-key", help="Override the environment/profile key")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument(
        "--cards-per-source",
        type=int,
        default=CARDS_PER_SOURCE,
        help="Cards taken per source; 0 or less takes every card",
    )
    parser.add_argument(
        "--devices",
        type=devices,
        default=DEFAULT_DEVICES,
        help=(
            "Comma-separated CUDA device ordinals; one worker process each. "
            f"Numbered by PCI bus id ({CUDA_ORDER}={CUDA_ORDER_PCI}), so an "
            "ordinal names the card nvidia-smi lists. Every ordinal must be the "
            "same card model."
        ),
    )
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--no-usage-proxy",
        action="store_true",
        help="Send Codex straight to --base-url instead of through a usage recorder",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="After the rounds, measure each with/without pair on identical inputs",
    )
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()

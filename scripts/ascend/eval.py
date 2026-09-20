"""Fixed evaluation entry for the Ascend generation experiment.

The agent calls this script after every meaningful kernel change::

    <venv>/python scripts/ascend/eval.py --work <workspace> [--device N]

It serializes the workspace's ``submission/`` bundle into ``kernel.json``,
synchronizes it to 910b1, runs the real evaluation inside the
``vllm0.23.0-zcj`` container, appends one ``evaluate`` record to
``<workspace>/.klineage/stats/events.jsonl``, stores the measured kernel under
``<workspace>/versions/version<N>/``, and prints the result as JSON.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from datetime import UTC, datetime
from hashlib import sha1
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
#: Harness root on the host: `<root>/repo` is this project's checkout there,
#: `<root>/runs` holds one evaluation copy per unit.
HARNESS_ROOT_DEFAULT = "~/ascend-harness"
REPO_NAME = "repo"
RUNS_NAME = "runs"
ARCH_DEFAULT = "dav-c220"
#: The torch-npu baseline is 50-500x slower than a device kernel; one harness
#: protocol for it costs minutes, so it is measured once per device and cached.
BASELINE_FILE = "baseline.json"
BASELINE_TIMEOUT = 3600.0
#: Container-local: the container runs as root, and a root-owned directory under
#: the run tree blocks the next rsync --delete.
BASELINE_SCRATCH = "/tmp/klineage-baseline"


def now() -> str:
    return datetime.now(UTC).isoformat()


def host_path(host: str, path: str) -> str:
    """Expand a leading `~` on the host: the container's HOME is /root."""

    if not path.startswith("~"):
        return path
    result = subprocess.run(["ssh", host, "echo $HOME"], capture_output=True, text=True)
    home = result.stdout.strip()
    if result.returncode or not home:
        detail = (result.stderr or "no $HOME").strip()[-200:]
        raise SystemExit(f"cannot resolve {path} on {host}: {detail}")
    return home + path[1:]


def run_ssh(host: str, container: str, script: str, *,
           timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run a shell snippet in the container through ssh, stdin from /dev/null.

    Codex refuses a piped stdin, so the redirect lives on the remote side; the
    snippet travels base64-encoded to keep quoting intact.
    """

    encoded = base64.b64encode(script.encode()).decode()
    command = f'docker exec -i {container} bash -lc "$(echo {encoded} | base64 -d)" </dev/null'
    return subprocess.run(["ssh", host, command], capture_output=True, text=True, timeout=timeout)


def remote_script(work: str, repository: str, timeout: float, device: str | None) -> str:
    environment = f"KLINEAGE_BACKEND=ascend ASCEND_ARCH={ARCH_DEFAULT}"
    if device:
        environment += f" ASCEND_RT_VISIBLE_DEVICES={device}"
    # umask 022: the container runs as root, and its default 027 hides snapshots
    # from the host user that rsync runs as.
    # The worker resolves workload inputs against its working directory and the
    # trace convention stores them relative to `problems/`, so it runs there.
    return (
        f"umask 022 && cd {work}/problems && {environment} "
        f"python {repository}/scripts/ascend/remote_eval.py --repo {repository} "
        f"--work {work} --timeout {timeout:g}"
    )


def baseline_script(work: str, repository: str, timeout: float, device: str | None) -> str:
    """Time the problem's torch reference: the Baseline column of the report."""

    environment = f"KLINEAGE_BACKEND=ascend ASCEND_ARCH={ARCH_DEFAULT}"
    if device:
        environment += f" ASCEND_RT_VISIBLE_DEVICES={device}"
    scratch = BASELINE_SCRATCH
    return (
        f"umask 022 && mkdir -p {scratch} && cd {work}/problems && {environment} "
        f"python {repository}/scripts/ascend/remote_eval.py --repo {repository} "
        f"--baseline --problems {work}/problems --scratch {scratch} "
        f"--timeout {timeout:g}"
    )


def push_work(work: Path, host: str, remote: str) -> None:
    """Mirror the workspace to the host; the worker reads it from there."""

    subprocess.run(["ssh", host, f"mkdir -p {remote}"], check=True)
    subprocess.run(
        ["rsync", "-az", "--delete", "--exclude", ".klineage", "--exclude", "versions",
         "--exclude", "build", "--exclude", "evaluations", f"{work}/", f"{host}:{remote}/"],
        check=True,
    )


def cached_baseline(work: Path, device: str) -> float | None:
    path = work.parent / BASELINE_FILE
    if not path.is_file():
        return None
    entry = json.loads(path.read_text(encoding="utf-8")).get(device) or {}
    return entry.get("baseline_ms")


def measure_baseline(work: Path, args: argparse.Namespace, remote: str,
                     repository: str) -> float:
    """Measure the torch-npu baseline once, then reuse it for every version."""

    device = args.device or "default"
    push_work(work, args.host, remote)
    result = run_ssh(
        args.host,
        args.container,
        baseline_script(remote, repository, BASELINE_TIMEOUT, args.device),
        timeout=BASELINE_TIMEOUT + 600,
    )
    payload = None
    for line in reversed(result.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict) and candidate.get("baseline_ms"):
            payload = candidate
            break
    if payload is None:
        detail = (result.stderr or result.stdout or "no baseline timing").strip()[-400:]
        raise RuntimeError(f"baseline measurement failed: {detail}")

    path = work.parent / BASELINE_FILE
    cache = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    cache[device] = {"baseline_ms": payload["baseline_ms"], "measured_at": now()}
    path.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
    return payload["baseline_ms"]


def bundle_sources(submission: Path) -> dict[str, str]:
    sources = {}
    for path in sorted(submission.rglob("*")):
        if path.is_file():
            sources[str(path.relative_to(submission))] = path.read_text(encoding="utf-8")
    if "config.toml" not in sources:
        raise SystemExit(f"missing {submission}/config.toml")
    return sources


def slug(work: Path) -> str:
    """Remote run directory name: `<kernel>-<setting>-work` when recognizable."""

    parts = work.resolve().parts
    for index, name in enumerate(parts):
        if name in ("without_memory", "with_memory"):
            return "-".join(parts[max(index - 1, 0) :])
    if "generation" in parts:
        return "-".join(parts[parts.index("generation") + 1 :])
    return sha1(str(work).encode()).hexdigest()[:12]


def platform_of(work: Path, host: str, container: str) -> str:
    cache = work / ".klineage" / "platform"
    if cache.is_file():
        return cache.read_text(encoding="utf-8").strip()
    result = run_ssh(
        host,
        container,
        f"cd {REPO_REMOTE} && KLINEAGE_BACKEND=ascend ASCEND_ARCH={ARCH} "
        "python scripts/ascend/remote_eval.py --probe",
    )
    platform = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if result.returncode or not platform.startswith("ascend"):
        raise SystemExit(f"platform probe failed: {result.stderr[-500:]}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(platform + "\n", encoding="utf-8")
    return platform


def append_event(work: Path, fields: dict) -> None:
    from klineage.logging import LOG_FILE

    stats = work / ".klineage" / "stats"
    stats.mkdir(parents=True, exist_ok=True)
    with (stats / LOG_FILE).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": "evaluate", **fields}, default=str) + "\n")


def main(argv: list[str] | None = None) -> None:
    global REPO_REMOTE, RUNS, ARCH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="2", help="NPU id on the host")
    parser.add_argument("--timeout", type=float, default=1800, help="evaluation worker seconds")
    parser.add_argument("--host", default="910b1")
    parser.add_argument("--container", default="vllm0.23.0-zcj")
    parser.add_argument("--remote-root", default=HARNESS_ROOT_DEFAULT,
                        help="harness root on the host: repo/ + runs/")
    parser.add_argument("--arch", default=ARCH_DEFAULT)
    parser.add_argument("--baseline", action="store_true",
                        help="measure and cache the torch-npu baseline, then exit")
    args = parser.parse_args(argv)

    root = host_path(args.host, args.remote_root)
    REPO_REMOTE, RUNS, ARCH = f"{root}/{REPO_NAME}", f"{root}/{RUNS_NAME}", args.arch
    work = args.work.expanduser().resolve()
    remote = f"{RUNS}/{slug(work)}"
    if args.baseline:
        print(json.dumps({"work": str(work),
                          "baseline_ms": measure_baseline(work, args, remote, REPO_REMOTE)}))
        return
    submission = work / "submission"
    if not submission.is_dir():
        raise SystemExit(f"missing kernel bundle: {submission}")

    definitions = sorted((work / "problems" / "definitions").glob("*.json"))
    if len(definitions) != 1:
        raise SystemExit("workspace must hold exactly one problem definition")
    definition_path = definitions[0]
    workload_path = work / "problems" / "workloads" / definition_path.with_suffix(".jsonl").name
    if not workload_path.is_file():
        raise SystemExit(f"missing workload: {workload_path}")

    from klineage.artifact.kernel import Kernel, save_kernel
    from klineage.artifact.problem import ProblemSpec

    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    workload = json.loads(workload_path.read_text(encoding="utf-8").splitlines()[0])["workload"]
    problem = ProblemSpec(
        name=definition["name"],
        definition=definition,
        workload=workload,
        language="ascendc",
        platform=platform_of(work, args.host, args.container),
    )
    kernel = Kernel(definition["name"], problem, bundle_sources(submission))
    save_kernel(kernel, work)

    push_work(work, args.host, remote)

    started = now()
    result = run_ssh(
        args.host,
        args.container,
        remote_script(remote, REPO_REMOTE, args.timeout, args.device),
        timeout=args.timeout + 600,
    )
    finished = now()

    payload = None
    for line in reversed(result.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict) and "correctness_passed" in candidate:
            payload = candidate
            break

    fields = {
        "at": started,
        "finished_at": finished,
        "kernel_name": kernel.name,
        "fingerprint": kernel.fingerprint,
        "platform": problem.platform,
    }
    if payload is None:
        fields["error"] = (result.stderr or result.stdout or "evaluation failed").strip()[-800:]
        append_event(work, fields)
        raise SystemExit(json.dumps(fields, indent=2))
    fields.update({name: payload.get(name) for name in
                   ("compile_passed", "correctness_passed", "profile_passed")})
    if payload.get("latency_ms") is not None:
        fields["latency_ms"] = payload["latency_ms"]

    # The study pins the artifact to triton-ascend: a timed call that launched
    # no triton kernel measured torch, and its latency is not comparable.
    launches = None
    for line in reversed(result.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict) and "gate" in candidate:
            launches = candidate["gate"].get("triton_launches")
            break
    if launches is not None:
        fields["triton_launches"] = launches
        fields["device_kernel"] = launches > 0

    if payload.get("latency_ms") is not None:
        try:
            baseline = cached_baseline(work, args.device) \
                or measure_baseline(work, args, remote, REPO_REMOTE)
            fields["baseline_ms"] = baseline
            fields["speedup"] = round(baseline / payload["latency_ms"], 3)
        except Exception as failure:  # a missing baseline must not void the run
            fields["baseline_error"] = f"{type(failure).__name__}: {failure}"[:300]
    append_event(work, fields)

    # Fetch the snapshot the remote evaluation just wrote, as version<N-1>.
    listing = run_ssh(args.host, args.container, f"ls {remote}/.klineage/versions 2>/dev/null || true")
    numbers = sorted(int(line) for line in listing.stdout.split() if line.isdigit())
    if numbers:
        # Local numbering counts this workspace's own versions, so it stays 0..N
        # even when the remote run directory already holds earlier snapshots.
        versions = work / "versions"
        index = sum(1 for path in versions.glob("version*") if path.is_dir()) if versions.is_dir() else 0
        folder = versions / f"version{index}"
        folder.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["rsync", "-az", f"{args.host}:{remote}/.klineage/versions/{numbers[-1]}/", f"{folder}/"],
                check=True,
            )
        except subprocess.CalledProcessError as failure:  # keep the measurement
            print(f"warning: could not fetch snapshot {numbers[-1]}: {failure}", file=sys.stderr)

    report = dict(payload)
    report.update({name: fields[name] for name in
                   ("baseline_ms", "speedup", "triton_launches", "device_kernel")
                   if name in fields})
    print(json.dumps(report, indent=2))
    if fields.get("device_kernel") is False:
        raise SystemExit(
            "rejected: the timed path launched no triton kernel; this study "
            "measures triton-ascend implementations only")


if __name__ == "__main__":
    main()

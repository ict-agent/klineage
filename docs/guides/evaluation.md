# Evaluation guide

The evaluation service builds a kernel, checks it against the problem's reference
implementation and measures each workload. Actions use this feedback to decide
what to try next and which result to return.

Generation starts a local evaluator by default. Start a separate service when
you want to reuse a device across runs or evaluate on another host.

<figure class="diagram" tabindex="0" aria-label="The agent submits kernels to an evaluation service, which runs them in an isolated worker.">
  <img src="../assets/diagrams/evaluation-hosts.svg"
       alt="The agent submits kernels to an evaluation service, which runs them in an isolated worker."
       width="800" height="420" loading="lazy">
</figure>

## Start a CUDA evaluation server

Install KLineage with the CUDA extra using [Getting started](../getting-started.md).
The evaluation host needs the matching driver, toolkit and Linux bubblewrap.
The server and its workers use the installed Python environment by default.

Set `KLINEAGE_PROBLEM` to one problem directory or a directory containing several
problems. From the repository root:

```bash
export KLINEAGE_PROBLEM="$PWD/problems"

klineage-server \
  --backend cuda \
  --root server-work/evaluator \
  --server http://127.0.0.1:9100 \
  --device-node /dev/nvidia0 \
  --device-node /dev/nvidiactl \
  --device-node /dev/nvidia-uvm \
  --env CUDA_VISIBLE_DEVICES=0
```

Replace device paths for your host. Each problem needs `definition.json` and
`workloads.jsonl`; evaluation requires at least one workload. Fixed-input paths
are relative to the problem directory, and those files must remain available.
Problem IDs come from `Problem.name`, normally the name in the definition.

Use these options to adjust the service:

| Option | Purpose |
| --- | --- |
| `--problem PATH` | Override `KLINEAGE_PROBLEM`; repeat for several problems or collections |
| `--server ENDPOINT` | Choose an HTTP or Unix-socket listener; `--listen` is an alias |
| `--timeout SECONDS` | Limit each evaluation job, including queue time |
| `--device-node PATH` | Expose a device to workers; repeat for each node |
| `--read-only PATH` | Mount additional runtime dependencies |
| `--include PATH` | Add evaluator-host compiler headers |
| `--env NAME=VALUE` | Set worker runtime variables |
| `--python PATH` | Use a separate worker interpreter |

A collection includes its direct child problem directories. Duplicate IDs and
incomplete problems are rejected. Existing `target.json` metadata takes precedence
when loading a problem; otherwise the service queries its backend for the target.

Run one service per device configuration, with separate environments for CUDA,
HIP and Ascend. A visibility variable selects devices within the runtime;
`--device-node` supplies access to them. Headers mounted in an agent's repository
must also be available to the evaluator if compilation needs them.

For a Unix listener, use `--server unix:///absolute/path/evaluation.sock`.
HTTP listeners accept a hostname and port; URL path prefixes, credentials,
queries and fragments are rejected. Put any HTTP path prefix in an external proxy.
The service has no authentication or built-in TLS: restrict it through socket
permissions or network controls, and use a TLS proxy for remote HTTPS.

### Use a separate worker interpreter

A CPU-only service can delegate execution with `--python /opt/device-env/bin/python`.
That environment must contain KLineage, the backend dependencies and vendor runtime.
Problem registration still runs in the service process. Prepare target metadata
with the device environment first so registration does not need to probe hardware:

```bash
/opt/device-env/bin/python - <<'PY'
import json
import shutil
from pathlib import Path

from klineage.artifact import load_problem
from klineage.backend import get_backend

root = Path("server-work/tasks/gemm")
shutil.copytree(Path("problems/gemm"), root)
problem = load_problem(root, backend=get_backend("cuda"))
(root / "target.json").write_text(
    json.dumps(problem.target(), indent=2) + "\n", encoding="utf-8"
)
PY
```

Start the server with that fresh problem copy, the worker interpreter and the
same device options:

```bash
klineage-server \
  --backend cuda --root server-work/evaluator \
  --problem server-work/tasks/gemm \
  --server http://127.0.0.1:9100 \
  --python /opt/device-env/bin/python \
  --device-node /dev/nvidia0 --device-node /dev/nvidiactl \
  --device-node /dev/nvidia-uvm --env CUDA_VISIBLE_DEVICES=0
```

## Submit and recover a result

On the controller, select the service:

```bash
export KLINEAGE_SERVER=http://127.0.0.1:9100
```

The generation, extraction and optimization commands also accept `--server`,
which overrides this variable. For Python scripts, install the library in your
own environment or run `uv run --no-project --with-editable . python script.py`
from the repository.

Submit an authored Kernel directory containing `metadata.json` and `submission/`:

```python
from pathlib import Path

import klineage

problem = klineage.problem("gemm")
work = Path("agent-workspace/evaluation")
reply = klineage.evaluate(Path("candidate"), work=work, timeout=300)

print(reply.job_id, reply.accepted)
if not reply.accepted:
    raise RuntimeError(reply.error or "Inspect the evaluation diagnostics")
best_reply = reply
```

Use `klineage.info()["problems"]` to list available IDs. Evaluation selects the
task from the Kernel's problem reference. Ambiguous references raise an error.

With `KLINEAGE_SERVER` configured, source files are submitted without local
compilation. Otherwise evaluation runs directly on the local backend, without
starting a server listener. Both paths return the same Evaluation and save the
same version layout. A remote failure does not trigger local evaluation.
An optional `reference`
provides a timing baseline; the problem's reference implementation remains the
correctness oracle.

Evaluation saves each returned Kernel, including failed evaluations:

| Path under `work` | Contents |
| --- | --- |
| `versions/<N>/` | Complete Kernel versions, starting at 0 |
| `versions/<N>/metadata.json` | Validation, workload timings, job ID and diagnostics |
| `problems/<problem_id>` | Symlink to the original problem directory |

Actions select the last numbered version. To return to an earlier accepted job,
fetch it as a new final version:

```python
restored = klineage.result(best_reply, work=work)
if not restored.accepted:
    raise RuntimeError("The recorded evaluation was not accepted")
```

The saved version contains the matching sources and measurements. Scratch files
can remain unchanged. Actions check the final Kernel against its service record;
Python controllers using Client can do the same with `client.verify(kernel)` and inspect
`accepted`.

A remote client can submit and retrieve results without local device libraries.
Generation and local Kernel loading also need the registered problem directory
at the same path, typically through a shared mount. Stored jobs remain usable
only while their registered problem is unchanged.

## Per-workload results

Each workload runs in its own worker. The candidate and optional baseline use
the same inputs, backend and measurement policy for that case. Read the results
by workload UUID:

```python
for workload_id, result in reply.validation.workloads.items():
    print(workload_id, result.accepted, result.latency_ms, result.reference_latency_ms)
```

Every workload must pass. For multiple workloads, top-level latency fields are
`None`; compare candidate and reference within each case. Paired performance
gates use a 0.99 ratio tolerance for each trial and the overall trial medians.

Check `reply.accepted` for the complete job status. A timeout can reject a job
even when some measurements passed. Completed results and diagnostics remain
available in `reply.diagnostics` and the saved version's `metadata.json`.

## Runtime settings { #deadlines-runtime-information-and-credentials }

### Time limits

One evaluation budget covers preparation, queue time and all workload workers,
capped by the service's `--timeout`. Starting another workload does not reset
that budget. A client disconnect does not cancel an active job.

Agent attempts and evaluation jobs have separate time limits. Controllers with
a total budget must track the remaining time across retries and requests.
`klineage.info()` reports the configured worker's device information; restart the
service after changing its runtime configuration.

### Model access

Model and provider settings come from `$CODEX_HOME/config.toml`, or
`~/.codex/config.toml` by default. Explicit Python arguments override
`KLINEAGE_*` variables, which override that configuration. See
[Configure model access](../getting-started.md#configure-model-access) for the
variables.

The provider bridge reads the configured API-key variable or Codex login in
`auth.json`. Host API-key variables are not forwarded to the agent. The agent
reuses `CODEX_HOME`, so files there, including saved credentials, remain visible
inside its sandbox. Renew expired login before rerunning. Named profiles,
keyring-only credentials and credential commands are unsupported; use root-level
settings and a host key variable when needed.

## Optional device diagnostics

Local generation supplies device and toolchain access automatically and adds the
selected compiler directory to `PATH`. With an
existing evaluator or manually composed actions, configure the agent separately
when it needs local builds or tools such as `ncu`. Direct `AgentRuntime` instances
have no device nodes by default.

For a host with these CUDA and Nsight Compute installations:

```python
from pathlib import Path

from klineage.harness import AgentRuntime, CodexRunner
from klineage.server import Client

agent = AgentRuntime(
    python=Path("/opt/cuda-env/bin/python"),
    device_nodes=tuple(Path(name) for name in (
        "/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm",
    )),
    read_only=(Path("/opt/nvidia/nsight-compute/current"),),
    environment={
        "PATH": "/opt/nvidia/nsight-compute/current:/usr/local/cuda/bin:/usr/bin:/bin",
        "CUDA_VISIBLE_DEVICES": "0",
    },
)
runner = CodexRunner(
    Path("agent-workspace/diagnostics"), evaluation=Client(), agent=agent
)
```

Use the same fields for HIP or Ascend with their own interpreter, device nodes
and runtime paths. Command-line equivalents are listed under
[Shared service and agent options](cli.md#shared-service-and-agent-options).
Performance counters may require host permissions beyond device access.

Diagnostic reports help the agent improve a kernel; evaluation still goes through
`klineage.evaluate()`. Run diagnostics and service benchmarks at different times
on a shared device so they do not affect each other's measurements.

## Isolation and embedding

The agent can write its run directory and access its existing Codex home, declared
inputs and runtimes. Trial sources belong in `current_kernel/`; helper scripts,
diagnostics and logs belong in `workspace/`. A task gateway provides evaluation
access, and a separate bridge handles model requests.
Workers receive the configured devices and problem inputs; service records stay
outside their writable directories. Missing sandbox support fails startup.
Candidate code and its checker share a worker process, so this isolation does not
protect the checker from deliberately malicious candidate code.

For a custom backend, `--plugin package.module` imports registration code in the
service and every worker. The controller and agent also need that package and
registration import. Follow the [Extension guide](extending.md#register-the-backend).

Python applications can embed `Service` directly. A custom `runtime_factory`
receives the remaining timeout; its runtime must support per-call timeouts and
return results for every workload. See the
[Evaluation service API](../reference/server.md) and
[Runtime API](../reference/runtime.md) for these interfaces.

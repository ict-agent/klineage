"""Start a backend evaluation service."""

import argparse
import importlib
import json
import os
import signal
import sys
import threading
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path

from klineage.backend import get_backend
from klineage.backend.registry import BACKEND_ENV

from .protocol import DEFAULT_TIMEOUT
from .service import PROBLEM_ENV, Service, load_problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--backend", required=True, help="Backend language, e.g. cuda, hip or ascendc"
    )
    parser.add_argument(
        "--problem",
        type=Path,
        action="append",
        help="Problem directory or collection; defaults to KLINEAGE_PROBLEM",
    )
    parser.add_argument(
        "--server", metavar="ENDPOINT",
        default="http://127.0.0.1:0", help="Evaluation service listener",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--include", type=Path, action="append", default=[])
    parser.add_argument("--plugin", action="append", default=[])
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Backend worker interpreter",
    )
    parser.add_argument("--device-node", type=Path, action="append", default=[])
    parser.add_argument("--read-only", type=Path, action="append", default=[])
    parser.add_argument(
        "--env", action="append", default=[], help="Worker environment NAME=VALUE"
    )
    args = parser.parse_args()
    for name in args.plugin:
        importlib.import_module(name)
    backend = get_backend(args.backend)
    paths = args.problem or ([Path(os.environ[PROBLEM_ENV])] if os.environ.get(PROBLEM_ENV) else [])
    if not paths:
        parser.error("--problem or KLINEAGE_PROBLEM must identify a problem directory")
    with redirect_stdout(sys.stderr):
        problems = load_problems(paths, backend=backend)
    from .worker import WorkerRuntime

    environment = {}
    for assignment in args.env:
        name, separator, value = assignment.partition("=")
        if not separator or not name.isidentifier():
            parser.error("--env must contain NAME=VALUE")
        environment[name] = value
    target = environment.get(BACKEND_ENV)
    if target is not None and target not in (backend.language, str(backend.kind)):
        try:
            selected = get_backend("python", target)
        except ValueError:
            parser.error(f"{BACKEND_ENV} is not a supported backend")
        if selected.kind != backend.kind:
            parser.error(f"{BACKEND_ENV} conflicts with --backend")
    environment[BACKEND_ENV] = str(backend.kind)
    factory = partial(
        WorkerRuntime,
        python=args.python,
        worker_plugins=args.plugin,
        device_nodes=args.device_node,
        read_only=args.read_only,
        environment=environment,
    )
    service = Service(
        args.root,
        backend,
        problems,
        timeout=args.timeout,
        include_paths=args.include,
        runtime_factory=factory,
    )
    stopped = threading.Event()
    for event in (signal.SIGTERM, signal.SIGINT):
        signal.signal(event, lambda *_: stopped.set())
    with service.start(args.server) as server:
        ready = {
            "endpoint": server.endpoint,
            "pid": os.getpid(),
            "problems": list(problems),
        }
        print(json.dumps(ready), flush=True)
        stopped.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

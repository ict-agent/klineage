"""Run workers and stop their descendant processes."""

from __future__ import annotations

import os
import signal
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path

#: Grace period before terminating an unresponsive process group.
STOP_GRACE_SECONDS = 5
PROCESS_SCOPE_ENV = "KLINEAGE_PROCESS_SCOPE"
STOP_POLL_SECONDS = 0.05


def scope_pids(scope: str) -> set[int]:
    """Find this run's descendants, including reparented tool processes."""
    marker = f"{PROCESS_SCOPE_ENV}={scope}".encode()
    pids = set()
    for path in Path("/proc").glob("[0-9]*/environ"):
        try:
            if path.stat().st_uid != os.getuid():
                continue
            if marker in path.read_bytes().split(b"\0"):
                pids.add(int(path.parent.name))
        except (OSError, ValueError):
            continue
    return pids


def signal_scope(scope: str, sig: signal.Signals):
    for pid in scope_pids(scope):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            continue


def stop_process(process: subprocess.Popen, *, scope: str | None = None):
    deadline = time.monotonic() + STOP_GRACE_SECONDS
    if scope is not None:
        signal_scope(scope, signal.SIGTERM)
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=max(0, deadline - time.monotonic()))

    # Tool shells create their own sessions, so a process group is insufficient.
    if scope is not None:
        while scope_pids(scope) and time.monotonic() < deadline:
            time.sleep(STOP_POLL_SECONDS)
        signal_scope(scope, signal.SIGKILL)
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def run_process(
    command: Sequence[str],
    *,
    payload: str,
    timeout: float,
    environment: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    scope = uuid.uuid4().hex
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**environment, PROCESS_SCOPE_ENV: scope},
        start_new_session=True,
    )
    timeout_error = None
    try:
        process.communicate(payload, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        timeout_error = error
    finally:
        # Reclaim detached descendants on success, failure and cancellation.
        stop_process(process, scope=scope)
        stdout, stderr = process.communicate()
    if timeout_error is not None:
        raise subprocess.TimeoutExpired(
            command, timeout, output=stdout, stderr=stderr
        ) from timeout_error
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

"""Run workers and stop their descendant processes."""

from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Mapping, Sequence

#: Grace period before terminating an unresponsive process group.
_GRACE_SECONDS = 5


def stop_process(process: subprocess.Popen):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_process(
    command: Sequence[str],
    *,
    payload: str,
    timeout: float,
    environment: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(payload, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        # Compilers and profilers may leave descendants after the worker exits.
        stop_process(process)
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            command, timeout, output=stdout, stderr=stderr
        ) from error
    except BaseException:
        stop_process(process)
        process.communicate()
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

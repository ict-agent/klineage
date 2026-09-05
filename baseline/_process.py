"""Process and artifact I/O for baseline experiments."""

from __future__ import annotations

import gzip
import json
import os
import shutil
import signal
import subprocess
from pathlib import Path

_TERM_GRACE_SECONDS = 5


def _execute(command, cwd, output, timeout, prompt=None) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stdout, output.with_suffix(".stderr").open("w") as stderr:
        process = subprocess.Popen(
            command, cwd=cwd, stdin=subprocess.PIPE, stdout=stdout,
            stderr=stderr, text=True, start_new_session=True,
        )
        output.with_suffix(".pid").write_text(str(process.pid) + "\n")
        try:
            process.communicate(prompt, timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate(process)
            return -signal.SIGTERM
        except BaseException:
            _terminate(process)
            raise
        finally:
            output.with_suffix(".pid").unlink(missing_ok=True)
    return process.returncode


def _terminate(process) -> None:
    # Kill descendants too; an orphaned compiler or kernel would pollute timing.
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=_TERM_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _git(root, *args) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _compress(path: Path) -> Path:
    destination = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as source, gzip.open(destination, "wb") as output:
        shutil.copyfileobj(source, output)
    path.unlink()
    return destination


def _thread_id(trace: Path) -> str | None:
    with trace.open() as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("type") == "thread.started":
                return event["thread_id"]
    return None


def _save_rollout(thread_id: str, destination: Path) -> None:
    home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    matches = list((home / "sessions").rglob(f"*{thread_id}.jsonl"))
    if len(matches) != 1:
        raise RuntimeError(f"cannot locate rollout for {thread_id}")
    with matches[0].open("rb") as source, gzip.open(destination, "wb") as output:
        shutil.copyfileobj(source, output)

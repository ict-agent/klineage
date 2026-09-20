#!/usr/bin/env python3
"""Start a command in its own session so it outlives the launching shell.

    scripts/ascend/spawn.py <log> <command> [args...]

macOS has no setsid(1); the double fork plus os.setsid() is the portable way to
escape the caller's process group, which the caller's terminal tears down.
"""

from __future__ import annotations

import os
import sys


def spawn(log: str, argv: list[str]) -> int:
    pid = os.fork()
    if pid:
        os.waitpid(pid, 0)          # reap the intermediate child
        return 0

    os.setsid()                     # new session: immune to the caller's killpg
    if os.fork():
        os._exit(0)

    handle = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    os.dup2(handle, 1)
    os.dup2(handle, 2)
    os.execvp(argv[0], argv)        # bare names resolve through PATH
    os._exit(127)


if __name__ == "__main__":
    sys.exit(spawn(sys.argv[1], sys.argv[2:]))

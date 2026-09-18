"""Evaluate one kernel workspace on the host NPU (runs inside the container).

Called by ``scripts/ascend/eval.py`` on the Mac; not meant for direct use.
Prints one line of JSON: the ValidationResult, or the platform in --probe mode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

#: This file lives at `<project root>/scripts/ascend/`, so the checkout is two
#: levels up; the SSH caller always runs it from there.
REPO_DEFAULT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--probe", action="store_true", help="print the platform and exit")
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(args.repo / "src"))
    from klineage.backend import detect_backend

    backend = detect_backend()
    if args.probe:
        print(backend.platform())
        return
    if args.work is None:
        raise SystemExit("--work is required")

    from klineage.artifact.kernel import load_kernel
    from klineage.harness.eval import evaluate

    work = Path(args.work).resolve()
    kernel = load_kernel(work)
    result = evaluate(kernel, work, timeout=args.timeout)
    print(json.dumps(result.to_dict()))


if __name__ == "__main__":
    main()

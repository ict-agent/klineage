"""Write the safetensors inputs a competition workload still needs.

The packages ship ``definitions`` and ``workloads`` only: the sample tensors
under ``inputs/`` were never uploaded, and a problem cannot run without them.
A problem is 1:1 with a workload whose reference is data independent, so the
bulk tensors are marked ``random`` in the workload (seed 0, deterministic, the
same for both settings) and only the values that must not be noise stay here:
a random ``scale`` or positive ``lower_bound`` turns the decay into an overflow.
``probs`` must be a distribution: its rows are normalized over seed-0 logits,
since noise would break the sum-to-one the top-p reference relies on.

    .venv-ascend/bin/python scripts/ascend/gen_inputs.py --kernel kda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
SAFETENSORS = "safetensors"
STRIDES_METADATA = "klineage.strides"
#: Frozen by the definition: "stride [1]; frozen to float32(0.9)".
TOP_P_DEFAULT = 0.9

#: Inputs whose value carries semantics, so they cannot be random.  Anything
#: else listed as safetensors falls back to seed-0 Gaussian noise.
SEMANTIC = {
    "scale": lambda sizes: sizes["HEAD_DIM"] ** -0.5,
    "lower_bound": lambda sizes: -5.0,
    "top_p": lambda sizes: TOP_P_DEFAULT,
}
#: Inputs that hold a distribution, not an arbitrary tensor: normalized per row.
PROBABILITY = ("probs",)


def axis_sizes(definition: dict, workload: dict) -> dict[str, int]:
    """Resolve every axis to an integer; workload axes override definition ones."""

    sizes = {
        name: (spec["value"] if spec["type"] == "const" else None)
        for name, spec in definition["axes"].items()
    }
    sizes.update(workload["axes"])
    missing = [name for name, size in sizes.items() if size is None]
    if missing:
        raise SystemExit(f"unresolved axes: {', '.join(missing)}")
    return sizes


def wanted(workload: dict) -> list[str]:
    return [name for name, spec in workload["inputs"].items() if spec["type"] == SAFETENSORS]


def build(definition: dict, workload: dict, names: list[str], seed: int) -> dict:
    sizes = axis_sizes(definition, workload)
    generator = torch.Generator().manual_seed(seed)
    inputs = {}
    for name in names:
        spec = definition["inputs"][name]
        shape = tuple(sizes[axis] for axis in spec["shape"])
        dtype = getattr(torch, spec["dtype"])
        if name in SEMANTIC:
            inputs[name] = torch.full(shape, SEMANTIC[name](sizes), dtype=dtype)
            continue
        if name in PROBABILITY:
            logits = torch.randn(shape, generator=generator, dtype=torch.float32)
            inputs[name] = torch.softmax(logits, dim=-1).to(dtype)
            continue
        inputs[name] = torch.randn(shape, generator=generator).to(dtype)
    return inputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="kda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = parser.parse_args(argv)

    from safetensors.torch import save_file

    root = REPO / "experiment" / args.kernel / "problems"
    definition = json.loads((root / "definitions" / f"{args.kernel}.json").read_text())
    workload = json.loads(
        (root / "workloads" / f"{args.kernel}.jsonl").read_text().splitlines()[0]
    )["workload"]
    names = wanted(workload)
    if not names:
        raise SystemExit(f"{args.kernel}: workload needs no safetensors input")

    target = root / "inputs" / f"{args.kernel}.safetensors"
    if target.exists() and not args.force:
        raise SystemExit(f"{target} exists; pass --force to overwrite")

    inputs = build(definition, workload, names, args.seed)
    for name, tensor in inputs.items():
        print(f"{name:14s} {str(tuple(tensor.shape)):18s} {tensor.dtype}")
    random_names = [name for name in definition["inputs"] if name not in names]
    print(f"random in workload: {', '.join(random_names)}")

    target.parent.mkdir(parents=True, exist_ok=True)
    save_file(inputs, str(target), metadata={STRIDES_METADATA: "{}"})
    print(f"wrote {target.relative_to(REPO)} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

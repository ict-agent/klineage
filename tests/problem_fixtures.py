"""Small FlashInfer Trace contracts for tests."""

from klineage.contract import ProblemSpec


def problem_spec(
    name="gemm",
    language="cuda",
    platform="sm120",
    *,
    description="Compute GEMM.",
    operator=None,
    size=128,
    dtype="float16",
):
    return ProblemSpec(
        name=name,
        language=language,
        platform=platform,
        definition={
            "name": name,
            "op_type": operator or name,
            "description": description,
            "axes": {axis: {"type": "var"} for axis in ("M", "N", "K")},
            "inputs": {
                "a": {"shape": ["M", "K"], "dtype": dtype},
                "b": {"shape": ["K", "N"], "dtype": dtype},
            },
            "outputs": {"output": {"shape": ["M", "N"], "dtype": dtype}},
            "reference": "def run(a, b):\n    return a @ b\n",
        },
        workload={
            "uuid": "3ec7cc7a-bdd8-44f4-8a68-bda263c44fd0",
            "axes": {"M": size, "N": size, "K": size},
            "inputs": {"a": {"type": "random"}, "b": {"type": "random"}},
        },
    )


def io_problem(
    inputs=(), outputs=(), *, name="gemm", language="cuda", platform="sm120"
):
    axes = {}
    specs = {}
    for role, values in (("inputs", inputs), ("outputs", outputs)):
        specs[role] = {}
        for value in values:
            shape = []
            for index, size in enumerate(value.shape):
                axis = f"{role}_{value.name}_{index}"
                axes[axis] = {"type": "const", "value": size}
                shape.append(axis)
            specs[role][value.name] = {
                "dtype": value.dtype,
                "shape": shape,
                "description": value.description,
            }
    return ProblemSpec(
        name=name,
        language=language,
        platform=platform,
        definition={
            "name": name,
            "op_type": name,
            "axes": axes,
            **specs,
            "reference": "def run(*args): pass\n",
        },
        workload={
            "uuid": "3ec7cc7a-bdd8-44f4-8a68-bda263c44fd0",
            "axes": {},
            "inputs": {value.name: {"type": "random"} for value in inputs},
        },
    )

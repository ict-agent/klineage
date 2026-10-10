"""Small FlashInfer Trace contracts for tests."""

from klineage.artifact.problem import Problem


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
    return Problem(
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
        workloads=({
            "uuid": "3ec7cc7a-bdd8-44f4-8a68-bda263c44fd0",
            "axes": {"M": size, "N": size, "K": size},
            "inputs": {"a": {"type": "random"}, "b": {"type": "random"}},
        },),
    )


def io_problem(
    inputs=None, outputs=None, *, name="gemm", language="cuda", platform="sm120"
):
    inputs, outputs = inputs or {}, outputs or {}
    axes = {}
    specs = {}
    for role, values in (("inputs", inputs), ("outputs", outputs)):
        specs[role] = {}
        for tensor_name, value in values.items():
            dimensions = value.get("shape", [])
            shape = None if dimensions is None else []
            for index, size in enumerate(dimensions or ()):
                axis = f"{role}_{tensor_name}_{index}"
                axes[axis] = {"type": "const", "value": size}
                shape.append(axis)
            specs[role][tensor_name] = {
                **value,
                "dtype": value.get("dtype"),
                "shape": shape,
            }
    return Problem(
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
        workloads=({
            "uuid": "3ec7cc7a-bdd8-44f4-8a68-bda263c44fd0",
            "axes": {},
            "inputs": {name: {"type": "random"} for name in inputs},
        },),
    )

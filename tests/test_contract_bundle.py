import json
import unittest
from types import MappingProxyType

from kernel_fixtures import edit_kernel, make_kernel
from problem_fixtures import problem_spec

from klineage.action import same_implementation
from klineage.artifact.problem import Problem
from klineage.serialization import restore, to_json
from klineage.utils import json_value

CUDA_SOURCE = "__global__ void kernel() {}"


def problem() -> Problem:
    return Problem(
        name="vector-add",
        definition={
            "name": "vector-add",
            "op_type": "elementwise",
            "description": "Increment every element of x and return the result.",
            "axes": {"N": {"type": "var"}},
            "inputs": {"x": {"shape": ["N"], "dtype": "float32"}},
            "outputs": {"result": {"shape": ["N"], "dtype": "float32"}},
            "reference": "def run(x):\n    return x + 1\n",
        },
        workloads=({
            "uuid": "c03709e8-79f7-4d50-b75f-5c82884b13b2",
            "axes": {"N": 1024},
            "inputs": {"x": {"type": "random"}},
        },),
        language="cuda",
        platform="sm90",
    )


class ContractTests(unittest.TestCase):
    def test_json_normalizes_collections(self):
        child = {"values": range(3)}
        source = MappingProxyType({"children": (child,)})
        result = json_value(source, "contract")
        self.assertEqual(result, {"children": [{"values": [0, 1, 2]}]})
        child["values"] = range(5)
        self.assertEqual(result["children"][0]["values"], [0, 1, 2])

    def test_trace_data_is_detached(self):
        raw = json.loads(to_json(problem()))
        value = restore(raw, Problem)
        raw["tree"]["children"].clear()
        self.assertEqual(value, problem())
        exported = value.workloads[0]
        exported["inputs"]["x"]["type"] = "scalar"
        self.assertEqual(value, problem())

    def test_identity_binds_io_order(self):
        spec = problem_spec()
        current = make_kernel("gemm", spec, source_files={"kernel.cu": CUDA_SOURCE})
        reordered = {
            **spec.definition,
            "inputs": dict(reversed(spec.definition["inputs"].items())),
        }
        changed = edit_kernel(current, problem=spec.replace(definition=reordered))
        self.assertFalse(same_implementation(current, changed))

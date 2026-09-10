import unittest
from dataclasses import replace

from problem_fixtures import problem_spec

from klineage.artifact.kernel import Kernel
from klineage.contract import ProblemSpec

CUDA_SOURCE = "__global__ void kernel() {}"


def problem() -> ProblemSpec:
    return ProblemSpec(
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
        workload={
            "uuid": "c03709e8-79f7-4d50-b75f-5c82884b13b2",
            "axes": {"N": 1024},
            "inputs": {"x": {"type": "random"}},
        },
        language="cuda",
        platform="sm90",
    )


class ContractTests(unittest.TestCase):
    def test_problem_json_schema(self):
        value = problem()
        expected = {"name", "definition", "workload", "language", "platform"}
        self.assertEqual(set(value.to_dict()), expected)
        self.assertEqual(ProblemSpec.from_dict(value.to_dict()), value)

    def test_trace_data_is_detached(self):
        raw = problem().to_dict()
        value = ProblemSpec.from_dict(raw)
        raw["definition"]["inputs"]["x"]["shape"].append("N")
        raw["workload"]["axes"]["N"] = 2048
        self.assertEqual(value, problem())
        exported = value.to_dict()
        exported["workload"]["inputs"]["x"]["type"] = "scalar"
        self.assertEqual(value, problem())

    def test_fingerprint_binds_io_order(self):
        spec = problem_spec()
        current = Kernel("gemm", spec, source_files={"kernel.cu": CUDA_SOURCE})
        reordered = {
            **spec.definition,
            "inputs": dict(reversed(spec.definition["inputs"].items())),
        }
        changed = replace(current, problem=replace(spec, definition=reordered))
        self.assertNotEqual(current.fingerprint, changed.fingerprint)

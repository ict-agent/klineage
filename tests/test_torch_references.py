from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
PROBLEMS = Path(__file__).resolve().parents[1] / "problems"
GEMM = PROBLEMS / "gemm"
SMALL_AXES = {"M": 2, "N": 3, "K": 4}

if TORCH_AVAILABLE:
    import torch
    from safetensors.torch import save_file

    from klineage.artifact.problem import load_trace, trace_inputs


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable")
class TorchReferenceTests(unittest.TestCase):

    def test_gemm_weight_layout(self):
        module = load_trace(GEMM)
        module.workloads[0]["axes"] = {"M": 3, "N": 5, "K": 4}
        inputs = trace_inputs(module.definition, module.workloads[0], device="cpu", seed=17)
        actual = module.run(**inputs)
        expected = inputs["x"] @ inputs["weight"].T
        torch.testing.assert_close(actual, expected)
        self.assertEqual(tuple(actual.shape), (3, 5))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable")
class TraceTests(unittest.TestCase):

    def fixed_input(self):
        # Keep loader coverage independent of bundled input datasets.
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        module = load_trace(GEMM)
        workload = module.workloads[0]
        workload["axes"] = dict(SMALL_AXES)
        shape = (SMALL_AXES["M"], SMALL_AXES["K"])
        value = torch.full(shape, 3.0, dtype=torch.bfloat16)
        path = root / "input.safetensors"
        save_file({"x": value}, str(path))
        workload["inputs"]["x"] = {
            "type": "safetensors", "path": str(path), "tensor_key": "x",
        }
        return module, value

    def test_dataset_references(self):
        paths = sorted(path.parent for path in PROBLEMS.glob("*/definition.json"))
        self.assertEqual(paths, [GEMM])
        module = load_trace(GEMM)
        definition = module.definition
        self.assertFalse(set(definition["inputs"]) & set(definition["outputs"]))

    def test_input_order_and_seed(self):
        module = load_trace(GEMM)
        workload = module.workloads[0]
        workload["axes"] = dict(SMALL_AXES)
        workload["inputs"] = dict(
            reversed(list(workload["inputs"].items()))
        )
        first = trace_inputs(module.definition, workload, device="cpu", seed=17)
        repeat = trace_inputs(module.definition, workload, device="cpu", seed=17)
        changed = trace_inputs(
            module.definition, workload, device="cpu", seed=42
        )
        self.assertEqual(list(first), ["x", "weight"])
        for name in first:
            torch.testing.assert_close(first[name], repeat[name], atol=0, rtol=0)
            self.assertFalse(torch.equal(first[name], changed[name]))

    def test_structured_inputs(self):
        module, expected = self.fixed_input()
        inputs = trace_inputs(module.definition, module.workloads[0], device="cpu")
        torch.testing.assert_close(inputs["x"], expected)

    def test_rejects_input_shape(self):
        module, _ = self.fixed_input()
        module.workloads[0]["axes"]["M"] += 1
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            trace_inputs(module.definition, module.workloads[0], device="cpu")

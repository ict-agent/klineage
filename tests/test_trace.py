import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
ROOT = Path(__file__).resolve().parents[1] / "problems"

if TORCH_AVAILABLE:
    import torch
    from safetensors import safe_open

    from klineage.artifact.problem import (
        load_trace,
        trace_definition,
        trace_inputs,
        trace_module,
        trace_workload,
    )
    from klineage.backend import get_backend
    from klineage.harness import eval as worker


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable")
class TraceTests(unittest.TestCase):
    def test_captured_layout_survives(self):
        values = {"x": torch.arange(6, dtype=torch.float32).reshape(2, 3).T}
        expected = values["x"] + 1
        definition = trace_definition(
            "increment",
            "elementwise",
            "def torch_ref(x): return x + 1",
            values,
            {"output": expected},
        )
        with tempfile.TemporaryDirectory() as temporary:
            workload = trace_workload(values, Path(temporary) / "inputs.safetensors")
            restored = trace_inputs(definition, workload, device="cpu")
        torch.testing.assert_close(restored["x"], values["x"], check_stride=True)

    def test_reference_needs_no_file(self):
        original = load_trace(ROOT / "gemm")
        original.workloads[0]["axes"] = {"M": 2, "N": 3, "K": 4}
        with patch.object(
            Path, "read_text", side_effect=AssertionError("read problem file")
        ):
            module = trace_module(original.definition, original.workloads)
            inputs = trace_inputs(module.definition, module.workloads[0], device="cpu")
            actual = module.torch_ref(*inputs.values())
        torch.testing.assert_close(actual, original.torch_ref(*inputs.values()))

    def test_dataset_references(self):
        paths = sorted(path.parent for path in ROOT.glob("*/definition.json"))
        self.assertEqual(len(paths), 5)
        for path in paths:
            with self.subTest(problem=path.stem):
                module = load_trace(path)
                definition = module.definition
                self.assertFalse(set(definition["inputs"]) & set(definition["outputs"]))
                for workload in module.workloads:
                    for descriptor in workload["inputs"].values():
                        if descriptor["type"] == "random":
                            continue
                        self.assertEqual(descriptor["type"], "safetensors")
                        self.assertTrue(Path(descriptor["path"]).is_absolute())
                        with safe_open(descriptor["path"], framework="pt") as tensors:
                            self.assertIn(descriptor["tensor_key"], tensors.keys())

    def test_input_order_and_seed(self):
        module = load_trace(ROOT / "gemm")
        workload = module.workloads[0]
        workload["axes"] = {"M": 2, "N": 3, "K": 4}
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
        module = load_trace(ROOT / "fmha")
        descriptor = module.workloads[0]["inputs"]["cu_seqlens_q"]
        with safe_open(descriptor["path"], framework="pt") as tensors:
            torch.testing.assert_close(
                tensors.get_tensor("cu_seqlens_q"),
                torch.arange(0, 16385, 2048, dtype=torch.int32),
            )
            torch.testing.assert_close(
                tensors.get_tensor("cu_seqlens_q"), tensors.get_tensor("cu_seqlens_k")
            )
        module = load_trace(ROOT / "gdn")
        descriptor = module.workloads[0]["inputs"]["q"]
        with safe_open(descriptor["path"], framework="pt") as tensors:
            for name in ("q", "k"):
                norms = tensors.get_tensor(name).float().norm(dim=-1)
                torch.testing.assert_close(
                    norms, torch.ones_like(norms), atol=0.02, rtol=0
                )
            gates = tensors.get_tensor("g")
            self.assertTrue((gates <= 0).all())
            self.assertEqual((gates == 0).all(dim=1).sum().item(), 12)
            beta = tensors.get_tensor("beta")
            self.assertTrue(((beta > 0) & (beta < 1)).all())

    def test_inspection_keeps_trace(self):
        path = ROOT / "gemm"
        original = load_trace(path)
        module = load_trace(path)
        module.workloads[0]["axes"] = {"M": 2, "N": 3, "K": 4}
        inputs = trace_inputs(module.definition, module.workloads[0], device="cpu")
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(worker, "load_problem_module", return_value=(module, path / "definition.json")),
            patch.object(worker, "load_inputs", return_value=inputs),
            patch.object(
                worker, "require_tensor", side_effect=lambda value, *args: value
            ),
            patch.object(worker, "detect_backend", return_value=get_backend("cuda")),
            patch.object(type(get_backend("cuda")), "info", return_value={"platform": "sm90"}),
        ):
            destination = Path(temporary) / "inputs.safetensors"
            result = worker.inspect_reference(path, workload_path=destination)
            self.assertEqual(result["definition"], original.definition)
            self.assertEqual(tuple(result["workloads"]), module.workloads)
            self.assertFalse(destination.exists())

    def test_rejects_duplicate_workload(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gemm"
            path.mkdir()
            source = ROOT / "gemm/definition.json"
            (path / "definition.json").write_text(source.read_text())
            records = path / "workloads.jsonl"
            text = (ROOT / "gemm/workloads.jsonl").read_text()
            records.write_text(text * 2)
            with self.assertRaisesRegex(ValueError, "duplicate workload uuid"):
                load_trace(path)
            record = json.loads(text)
            record["definition"] = "other"
            records.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "different definition"):
                load_trace(path)

    def test_rejects_wrong_tensor_shape(self):
        module = load_trace(ROOT / "fmha")
        definition = copy.deepcopy(module.definition)
        definition["inputs"] = {"cu_seqlens_q": definition["inputs"]["cu_seqlens_q"]}
        module.workloads[0]["axes"]["OFFSETS"] = 3
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            trace_inputs(definition, module.workloads[0], device="cpu")


if __name__ == "__main__":
    unittest.main()

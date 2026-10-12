from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
PROBLEMS = Path(__file__).resolve().parents[1] / "problems"

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F

    from klineage.artifact.problem import load_trace, trace_inputs


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable")
class TorchReferenceTests(unittest.TestCase):

    def test_gemm_weight_layout(self):
        module = load_trace(PROBLEMS / "gemm")
        module.workloads[0]["axes"] = {"M": 3, "N": 5, "K": 4}
        inputs = trace_inputs(module.definition, module.workloads[0], device="cpu", seed=17)
        actual = module.run(**inputs)
        expected = inputs["x"] @ inputs["weight"].T
        torch.testing.assert_close(actual, expected)
        self.assertEqual(tuple(actual.shape), (3, 5))

    def test_conv2d_filter_layout(self):
        module = load_trace(PROBLEMS / "conv2d")
        module.workloads[0]["axes"] = {
            "N": 2,
            "H": 5,
            "W": 4,
            "C": 2,
            "F": 3,
            "FILTER_ROWS": 18,
        }
        inputs = trace_inputs(module.definition, module.workloads[0], device="cpu", seed=23)
        actual = module.run(**inputs)
        weight = (
            inputs["weight_hwcf"].T.reshape(3, 3, 3, 2).permute(0, 3, 1, 2).contiguous()
        )
        expected = F.conv2d(inputs["data_nhwc"].permute(0, 3, 1, 2), weight, padding=1)
        torch.testing.assert_close(actual, expected.permute(0, 2, 3, 1))
        self.assertEqual(tuple(actual.shape), (2, 5, 4, 3))

    def test_fmha_packed_output(self):
        module = load_trace(PROBLEMS / "fmha")
        generator = torch.Generator().manual_seed(42)
        q, k, v = (torch.randn(6, 2, 4, generator=generator) for _ in range(3))
        offsets = torch.tensor([0, 3, 6], dtype=torch.int32)
        actual = module.run(q, k, v, offsets, offsets)
        expected = (
            F.scaled_dot_product_attention(
                *(value.view(2, 3, 2, 4).transpose(1, 2) for value in (q, k, v))
            )
            .transpose(1, 2)
            .reshape(6, 2, 4)
        )
        torch.testing.assert_close(actual, expected)

    def test_topk_values_and_indices(self):
        module = load_trace(PROBLEMS / "topk")
        values = torch.tensor([[1.0, -2.0, 4.0, 3.0], [0.0, 8.0, 2.0, 7.0]])
        result = module.run(values, k=2)
        torch.testing.assert_close(result.values, values.gather(1, result.indices))
        torch.testing.assert_close(
            result.values.sort(dim=-1).values, torch.tensor([[3.0, 4.0], [7.0, 8.0]])
        )
        self.assertEqual(result.indices.dtype, torch.int64)
        self.assertEqual(module.definition["axes"]["K"]["value"], 512)

    def test_gdn_matches_recurrence(self):
        module = load_trace(PROBLEMS / "gdn")
        generator = torch.Generator().manual_seed(17)
        q, k = (
            F.normalize(torch.randn(1, 5, 1, 4, generator=generator), dim=-1)
            for _ in range(2)
        )
        v = torch.randn(1, 5, 2, 4, generator=generator)
        g = F.logsigmoid(torch.randn(1, 5, 2, generator=generator)) / 16
        beta = torch.randn(1, 5, 2, generator=generator).sigmoid()
        state = torch.randn(1, 2, 4, 4, generator=generator)
        actual, actual_state = module.run(q, k, v, g, beta, state, chunk_size=4)
        expected, expected_state = module.naive_recurrent_gated_delta_rule(
            q.repeat_interleave(2, dim=2),
            k.repeat_interleave(2, dim=2),
            v,
            beta,
            g,
            initial_state=state,
            output_final_state=True,
        )
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actual_state, expected_state, atol=2e-5, rtol=2e-5)
        for spec in module.definition["outputs"].values():
            self.assertEqual(spec["dtype"], "float32")


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
ROOT = Path(__file__).resolve().parents[1] / "problems"

if TORCH_AVAILABLE:
    import torch
    from safetensors import safe_open

    from klineage.artifact.problem import (
        load_trace,
        trace_inputs,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable")
class TraceTests(unittest.TestCase):

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

    def test_rejects_wrong_tensor_shape(self):
        module = load_trace(ROOT / "fmha")
        definition = copy.deepcopy(module.definition)
        definition["inputs"] = {"cu_seqlens_q": definition["inputs"]["cu_seqlens_q"]}
        module.workloads[0]["axes"]["OFFSETS"] = 3
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            trace_inputs(definition, module.workloads[0], device="cpu")

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
DEFINITIONS = Path(__file__).resolve().parents[1] / "problems/definitions"

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F

    from klineage.artifact.problem import load_trace, trace_inputs


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable")
class TorchReferenceTests(unittest.TestCase):
    def test_paper_workloads(self):
        expected = {
            "gemm": {"M": 4096, "N": 4096, "K": 4096},
            "conv2d": {"N": 8, "H": 56, "W": 56, "C": 64, "F": 128, "FILTER_ROWS": 576},
            "fmha": {"TOKENS": 16384, "HEADS": 64, "HEAD_DIM": 128, "OFFSETS": 9},
            "gdn": {"B": 1, "T": 4096, "HQ": 16, "HV": 48, "D": 128},
            "topk": {"B": 64, "S": 4096},
        }
        for name, axes in expected.items():
            with self.subTest(name=name):
                module = load_trace(DEFINITIONS / f"{name}.json")
                self.assertEqual(module.workload["axes"], axes)
                self.assertEqual(module.definition["op_type"], name)
                self.assertFalse(hasattr(module, "make_inputs"))

    def test_gemm_weight_layout(self):
        module = load_trace(DEFINITIONS / "gemm.json")
        module.workload["axes"] = {"M": 3, "N": 5, "K": 4}
        inputs = trace_inputs(module.definition, module.workload, device="cpu", seed=17)
        actual = module.run(**inputs)
        expected = inputs["x"] @ inputs["weight"].T
        torch.testing.assert_close(actual, expected)
        self.assertEqual(tuple(actual.shape), (3, 5))

    def test_conv2d_filter_layout(self):
        module = load_trace(DEFINITIONS / "conv2d.json")
        module.workload["axes"] = {
            "N": 2,
            "H": 5,
            "W": 4,
            "C": 2,
            "F": 3,
            "FILTER_ROWS": 18,
        }
        inputs = trace_inputs(module.definition, module.workload, device="cpu", seed=23)
        actual = module.run(**inputs)
        weight = (
            inputs["weight_hwcf"].T.reshape(3, 3, 3, 2).permute(0, 3, 1, 2).contiguous()
        )
        expected = F.conv2d(inputs["data_nhwc"].permute(0, 3, 1, 2), weight, padding=1)
        torch.testing.assert_close(actual, expected.permute(0, 2, 3, 1))
        self.assertEqual(tuple(actual.shape), (2, 5, 4, 3))

    def test_fmha_packed_output(self):
        module = load_trace(DEFINITIONS / "fmha.json")
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
        module = load_trace(DEFINITIONS / "topk.json")
        values = torch.tensor([[1.0, -2.0, 4.0, 3.0], [0.0, 8.0, 2.0, 7.0]])
        result = module.run(values, k=2)
        torch.testing.assert_close(result.values, values.gather(1, result.indices))
        torch.testing.assert_close(
            result.values.sort(dim=-1).values, torch.tensor([[3.0, 4.0], [7.0, 8.0]])
        )
        self.assertEqual(result.indices.dtype, torch.int64)
        self.assertEqual(module.definition["axes"]["K"]["value"], 512)

    def test_gdn_matches_recurrence(self):
        module = load_trace(DEFINITIONS / "gdn.json")
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


if __name__ == "__main__":
    unittest.main()

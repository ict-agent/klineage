from __future__ import annotations

import importlib.util
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F

    from problems.conv2d import reference as conv2d
    from problems.fmha import reference as fmha
    from problems.gdn import reference as gdn
    from problems.gemm import reference as gemm
    from problems.topk import reference as topk


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is provided by the CUDA image")
class TorchReferenceTests(unittest.TestCase):
    def test_paper_constants(self) -> None:
        self.assertEqual((gemm.M, gemm.N, gemm.K), (4096, 4096, 4096))
        self.assertEqual(
            (
                conv2d.BATCH,
                conv2d.IN_CHANNELS,
                conv2d.HEIGHT,
                conv2d.WIDTH,
                conv2d.OUT_CHANNELS,
                conv2d.KERNEL_SIZE,
            ),
            (8, 64, 56, 56, 128, 3),
        )
        self.assertEqual(
            (fmha.BATCH, fmha.HEADS, fmha.SEQUENCE_LENGTH, fmha.HEAD_DIM),
            (8, 64, 2048, 128),
        )
        self.assertEqual(
            (
                gdn.QK_HEADS,
                gdn.VALUE_HEADS,
                gdn.HEAD_DIM,
                gdn.CHUNK_SIZE,
                gdn.SEQUENCE_LENGTH,
            ),
            (16, 48, 128, 64, 4096),
        )
        self.assertEqual((topk.BATCH, topk.SEQUENCE_LENGTH, topk.K), (64, 4096, 512))

    def test_gemm_transposed_weight_contract(self) -> None:
        inputs = gemm.make_inputs(
            m=3, n=5, k=4, dtype=torch.float32, device="cpu", seed=17
        )
        actual = gemm.torch_ref(**inputs)
        expected = inputs["x"] @ inputs["weight"].T
        torch.testing.assert_close(actual, expected)
        self.assertEqual(tuple(actual.shape), (3, 5))

    def test_conv2d_nhwc_and_flattened_hwcf_contract(self) -> None:
        inputs = conv2d.make_inputs(
            batch=2,
            in_channels=2,
            height=5,
            width=4,
            out_channels=3,
            kernel_size=3,
            dtype=torch.float32,
            device="cpu",
            seed=23,
        )
        actual = conv2d.torch_ref(**inputs, kernel_size=3, stride=1, padding=1)
        weight = (
            inputs["weight_hwcf"].T.reshape(3, 3, 3, 2).permute(0, 3, 1, 2).contiguous()
        )
        expected = F.conv2d(
            inputs["data_nhwc"].permute(0, 3, 1, 2), weight, padding=1
        ).permute(0, 2, 3, 1)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(tuple(actual.shape), (2, 5, 4, 3))

    def test_fmha_returns_packed_nhd(self) -> None:
        inputs = fmha.make_inputs(
            batch=2,
            heads=2,
            sequence_length=3,
            head_dim=4,
            dtype=torch.float32,
            device="cpu",
            seed=42,
        )
        actual = fmha.torch_ref(**inputs)
        q = inputs["q"].view(2, 3, 2, 4).transpose(1, 2)
        k = inputs["k"].view(2, 3, 2, 4).transpose(1, 2)
        v = inputs["v"].view(2, 3, 2, 4).transpose(1, 2)
        expected = F.scaled_dot_product_attention(q, k, v)
        expected = expected.transpose(1, 2).reshape(6, 2, 4)
        torch.testing.assert_close(actual, expected)

    def test_topk_returns_values_and_indices(self) -> None:
        values = torch.tensor([[1.0, -2.0, 4.0, 3.0], [0.0, 8.0, 2.0, 7.0]])
        result = topk.torch_ref(values, k=2)
        selected = torch.gather(values, 1, result.indices)
        torch.testing.assert_close(result.values, selected)
        torch.testing.assert_close(
            result.values.sort(dim=-1).values,
            torch.tensor([[3.0, 4.0], [7.0, 8.0]]),
        )

    def test_chunked_gdn_matches_recurrent_definition(self) -> None:
        inputs = gdn.make_inputs(
            batch=1,
            sequence_length=5,
            qk_heads=1,
            value_heads=2,
            head_dim=4,
            dtype=torch.float32,
            device="cpu",
            seed=17,
            swa_ratio=1.0,
        )
        actual, actual_state = gdn.torch_ref(**inputs, chunk_size=4)
        repeats = inputs["v"].shape[2] // inputs["q"].shape[2]
        expected, expected_state = gdn.naive_recurrent_gated_delta_rule(
            inputs["q"].repeat_interleave(repeats, dim=2),
            inputs["k"].repeat_interleave(repeats, dim=2),
            inputs["v"],
            inputs["beta"],
            inputs["g"],
            initial_state=inputs["initial_state"],
            output_final_state=True,
        )
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actual_state, expected_state, atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()

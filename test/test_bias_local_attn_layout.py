import math
import unittest

import torch

from opentome.timm.bias_local_attn import (
    biased_local_attention,
    clear_cache,
    unbiased_local_attention,
)


def _reference_local_attention(q, k, v, window, bias=None, logical_dim=None):
    """FP32 reference. Inputs and output use BNHD layout."""
    dim = logical_dim if logical_dim is not None else q.shape[-1]
    q_bhnd = q[..., :dim].float().permute(0, 2, 1, 3)
    k_bhnd = k[..., :dim].float().permute(0, 2, 1, 3)
    v_bhnd = v[..., :dim].float().permute(0, 2, 1, 3)

    scores = torch.matmul(q_bhnd, k_bhnd.transpose(-2, -1)) / math.sqrt(dim)
    if bias is not None:
        scores = scores + bias.float()[:, None, None, :]

    token_count = q.shape[1]
    positions = torch.arange(token_count, device=q.device)
    local_mask = (positions[:, None] - positions[None, :]).abs() <= window
    scores = scores.masked_fill(~local_mask[None, None], float("-inf"))

    probabilities = scores.softmax(dim=-1)
    out_bhnd = torch.matmul(probabilities, v_bhnd)
    return out_bhnd.permute(0, 2, 1, 3).to(q.dtype)


def _to_layout(tensor, layout):
    if layout == "BNHD":
        return tensor.clone()
    if layout == "BHND":
        return tensor.transpose(1, 2)
    raise ValueError(layout)


def _to_bnhd(tensor, layout):
    return tensor if layout == "BNHD" else tensor.transpose(1, 2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required by flash-attn")
class FlashAttentionLayoutTest(unittest.TestCase):
    device = torch.device("cuda")
    dtype = torch.float16

    def setUp(self):
        clear_cache()
        torch.manual_seed(20260718)

    def tearDown(self):
        clear_cache()

    def assertAttentionClose(self, actual, expected):
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            rtol=2e-3,
            atol=2e-3,
        )

    def _random_bnhd(self, dim, batch=2, tokens=11, heads=3):
        shape = (batch, tokens, heads, dim)
        return tuple(
            torch.randn(shape, device=self.device, dtype=self.dtype)
            for _ in range(3)
        )

    def test_unbiased_matches_reference_for_both_layouts_and_padding_paths(self):
        for dim in (8, 7):
            q, k, v = self._random_bnhd(dim)
            expected = _reference_local_attention(q, k, v, window=2)

            for layout in ("BNHD", "BHND"):
                with self.subTest(dim=dim, layout=layout):
                    actual = unbiased_local_attention(
                        _to_layout(q, layout),
                        _to_layout(k, layout),
                        _to_layout(v, layout),
                        local_window=2,
                        training=True,
                    )
                    self.assertEqual(actual.shape, _to_layout(expected, layout).shape)
                    self.assertAttentionClose(
                        _to_bnhd(actual, layout),
                        expected,
                    )

    def test_biased_general_path_matches_reference_for_both_layouts(self):
        q, k, v = self._random_bnhd(dim=8)
        bias = torch.randn(
            q.shape[0], q.shape[1], device=self.device, dtype=self.dtype
        )
        expected = _reference_local_attention(q, k, v, window=2, bias=bias)

        for layout in ("BNHD", "BHND"):
            with self.subTest(layout=layout):
                actual = biased_local_attention(
                    _to_layout(q, layout),
                    _to_layout(k, layout),
                    _to_layout(v, layout),
                    bias=bias,
                    local_window=2,
                    training=True,
                )
                self.assertEqual(actual.shape, _to_layout(expected, layout).shape)
                self.assertAttentionClose(
                    _to_bnhd(actual, layout),
                    expected,
                )

    def test_biased_prealigned_fast_path_matches_reference_for_both_layouts(self):
        logical_dim = 8
        physical_dim = 16
        q_logic, k_logic, v_logic = self._random_bnhd(dim=logical_dim)
        physical_shape = q_logic.shape[:-1] + (physical_dim,)
        q = torch.zeros(physical_shape, device=self.device, dtype=self.dtype)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        q[..., :logical_dim] = q_logic
        k[..., :logical_dim] = k_logic
        v[..., :logical_dim] = v_logic
        bias = torch.randn(
            q.shape[0], q.shape[1], device=self.device, dtype=self.dtype
        )
        expected = _reference_local_attention(
            q, k, v, window=2, bias=bias, logical_dim=logical_dim
        )

        for layout in ("BNHD", "BHND"):
            with self.subTest(layout=layout):
                actual = biased_local_attention(
                    _to_layout(q, layout),
                    _to_layout(k, layout),
                    _to_layout(v, layout),
                    bias=bias,
                    local_window=2,
                    logical_dim=logical_dim,
                    training=False,
                )
                self.assertEqual(actual.shape, _to_layout(expected, layout).shape)
                self.assertAttentionClose(
                    _to_bnhd(actual, layout),
                    expected,
                )

    def test_window_is_applied_to_token_axis(self):
        batch, tokens, heads, dim = 1, 9, 3, 8
        q = torch.zeros(
            (batch, tokens, heads, dim), device=self.device, dtype=self.dtype
        )
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        v[:, 4, 1, 0] = 1.0

        actual_bhnd = unbiased_local_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            local_window=1,
            training=False,
        )
        actual = actual_bhnd.transpose(1, 2)

        expected = torch.zeros_like(actual)
        expected[:, 3:6, 1, 0] = 1.0 / 3.0
        self.assertAttentionClose(actual, expected)

    def test_unbiased_padded_training_backward(self):
        batch, tokens, heads, dim = 1, 9, 3, 7
        shape = (batch, heads, tokens, dim)
        q = torch.randn(
            shape, device=self.device, dtype=self.dtype, requires_grad=True
        )
        k = torch.randn(
            shape, device=self.device, dtype=self.dtype, requires_grad=True
        )
        v = torch.randn(
            shape, device=self.device, dtype=self.dtype, requires_grad=True
        )

        out = unbiased_local_attention(
            q, k, v, local_window=2, training=True
        )
        out.float().square().mean().backward()

        for name, tensor in (("q", q), ("k", k), ("v", v)):
            with self.subTest(tensor=name):
                self.assertIsNotNone(tensor.grad)
                self.assertTrue(torch.isfinite(tensor.grad).all())
                self.assertGreater(float(tensor.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()

"""Tests for dtype preservation in functional.quant.state_quant (#439)."""

import pytest
import torch

from snntorch.functional import quant


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("uniform", [True, False], ids=["uniform", "nonuniform"])
@pytest.mark.parametrize("thr_centered", [True, False])
def test_state_quant_preserves_dtype(dtype, uniform, thr_centered):
    x = torch.tensor([0.125, 0.875], dtype=dtype)
    q = quant.state_quant(
        num_bits=4, uniform=uniform, thr_centered=thr_centered, threshold=1.0
    )
    out = q(x)
    assert out.dtype == dtype
    # values must still be valid quantization levels
    assert out.numel() == x.numel()
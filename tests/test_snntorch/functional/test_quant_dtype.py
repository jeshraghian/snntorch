"""Tests for dtype preservation in spikegen.latency (#440) and
functional.quant.state_quant (#439)."""

import pytest
import torch

from snntorch import spikegen
from snntorch.functional import quant


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"normalize": True},
        {"linear": True, "normalize": True},
        {"interpolate": True, "normalize": True},
    ],
    ids=["default", "normalize", "linear+normalize", "interpolate+normalize"],
)
def test_latency_preserves_dtype(dtype, kwargs):
    x = torch.tensor([0.2, 0.8], dtype=dtype)
    out = spikegen.latency(x, num_steps=5, **kwargs)
    assert out.dtype == dtype


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

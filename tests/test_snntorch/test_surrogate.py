#!/usr/bin/env python

"""Tests for surrogate gradients."""

import pytest
import torch
from snntorch import surrogate


class TestSigmoidSurrogate:
    def test_sigmoid_backward_no_nan(self):
        # issue #427: exp(-slope * input_) overflows to inf for
        # slope * input_ < ~-88, giving inf / inf = nan gradients
        x = torch.tensor([-5.0], requires_grad=True)
        spk = surrogate.sigmoid()(x)
        spk.backward()
        assert torch.isfinite(x.grad).all()

    @pytest.mark.parametrize("slope", [1, 25, 100])
    def test_sigmoid_backward_finite_over_range(self, slope):
        x = torch.linspace(-100, 100, 201, requires_grad=True)
        spk = surrogate.sigmoid(slope=slope)(x)
        spk.sum().backward()
        assert torch.isfinite(x.grad).all()

    def test_sigmoid_backward_matches_analytical(self):
        # where the exponential form is stable, the rewritten gradient
        # must agree with slope * exp(-slope*u) / (exp(-slope*u) + 1)**2.
        # compared in float64 to keep the reference itself precise
        slope = 25
        x = torch.linspace(
            -0.5, 0.5, 101, dtype=torch.float64, requires_grad=True
        )
        spk = surrogate.sigmoid(slope=slope)(x)
        spk.sum().backward()

        exp_term = torch.exp(-slope * x.detach())
        expected = slope * exp_term / (exp_term + 1) ** 2
        assert torch.allclose(x.grad, expected, rtol=1e-9, atol=1e-12)

    def test_sigmoid_backward_saturates_to_zero(self):
        x = torch.tensor([-1000.0, 1000.0], requires_grad=True)
        spk = surrogate.sigmoid()(x)
        spk.sum().backward()
        assert torch.isfinite(x.grad).all()
        assert torch.allclose(x.grad, torch.zeros_like(x.grad))

#!/usr/bin/env python

"""Tests for surrogate gradients."""

import pytest
import torch
from snntorch import surrogate


class TestSigmoidSurrogate:
    def test_sigmoid_backward_no_nan(self):
        # issue #427: exp(-slope * input_) overflows to inf for
        # slope * input_ < ~-88 and gives inf / inf = nan gradients
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
        # where the exponential form is stable, the rewritten gradient must
        # agree with slope * exp(-slope * u) / (exp(-slope * u) + 1) ** 2
        slope = 25
        x = torch.linspace(
            -0.5, 0.5, 101, dtype=torch.float64, requires_grad=True
        )
        spk = surrogate.sigmoid(slope=slope)(x)
        spk.sum().backward()

        exp_term = torch.exp(-slope * x.detach())
        expected = slope * exp_term / (exp_term + 1) ** 2
        # comparison uses float64 to keep the reference itself precise
        assert torch.allclose(x.grad, expected, rtol=1e-9, atol=1e-12)

    def test_sigmoid_backward_saturates_to_zero(self):
        x = torch.tensor([-1000.0, 1000.0], requires_grad=True)
        spk = surrogate.sigmoid()(x)
        spk.sum().backward()
        assert torch.isfinite(x.grad).all()
        assert torch.allclose(x.grad, torch.zeros_like(x.grad))


class TestGaussianSurrogate:
    def test_gaussian_backward_finite(self):
        x = torch.linspace(-10, 10, 101, requires_grad=True)
        spk = surrogate.gaussian(sigma=1.5)(x)
        spk.sum().backward()
        assert torch.isfinite(x.grad).all()

    def test_gaussian_backward_analytical_match(self):
        sigma = 2.0
        x = torch.linspace(-1.0, 1.0, 50, dtype=torch.float64, requires_grad=True)
        spk = surrogate.gaussian(sigma=sigma)(x)
        spk.sum().backward()
        expected = (sigma / torch.sqrt(torch.tensor(2 * torch.pi, dtype=torch.float64))) * torch.exp(-0.5 * (sigma * x.detach()).pow(2))
        assert torch.allclose(x.grad, expected, rtol=1e-5, atol=1e-6)


class TestSmoothedHeavisideSurrogate:
    def test_smoothed_heaviside_backward_finite(self):
        x = torch.linspace(-10, 10, 101, requires_grad=True)
        spk = surrogate.smoothed_heaviside(alpha=2.0)(x)
        spk.sum().backward()
        assert torch.isfinite(x.grad).all()

    def test_smoothed_heaviside_analytical_match(self):
        alpha = 3.0
        x = torch.linspace(-1.0, 1.0, 50, dtype=torch.float64, requires_grad=True)
        spk = surrogate.smoothed_heaviside(alpha=alpha)(x)
        spk.sum().backward()
        expected = alpha / 2.0 / (1.0 + alpha * torch.abs(x.detach())).pow(2)
        assert torch.allclose(x.grad, expected, rtol=1e-5, atol=1e-6)


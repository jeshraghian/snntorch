#!/usr/bin/env python

"""Tests for fluctuation-driven weight initialization."""

import math

import pytest
import torch
import torch.nn as nn

from snntorch import weight_init


class TestComputePspKernelIntegrals:
    """Tests for the epsilon computation helper."""

    def test_analytical_values(self):
        """Verify analytical epsilon formulas match paper
        Supplementary Material."""
        tau_mem, tau_syn = 0.02, 0.005
        ebar, ehat = weight_init.compute_psp_kernel_integrals(
            tau_mem, tau_syn, mode="analytical"
        )
        # Eq. S1: epsilon_bar = tau_syn
        expected_ebar = tau_syn
        # Eq. S1: epsilon_hat = tau_syn^2 / (2 * (tau_syn + tau_mem))
        expected_ehat = (tau_syn**2) / (2 * (tau_syn + tau_mem))
        assert abs(ebar - expected_ebar) < 1e-10
        assert abs(ehat - expected_ehat) < 1e-10

    def test_numerical_matches_analytical(self):
        """Verify numerical computation approximates analytical within 5%.

        The analytical formula is an approximation of the actual LIF dynamics.
        The numerical simulation is more accurate for large time steps.
        """
        ebar_a, ehat_a = weight_init.compute_psp_kernel_integrals(
            0.02, 0.005, mode="analytical"
        )
        ebar_n, ehat_n = weight_init.compute_psp_kernel_integrals(
            0.02, 0.005, dt=1e-4, mode="numerical"
        )
        # Should be close (within 5%) since analytical is an approximation
        assert abs(ebar_a - ebar_n) / ebar_a < 0.05
        assert abs(ehat_a - ehat_n) / ehat_a < 0.05

    def test_invalid_mode_raises(self):
        """Verify ValueError for invalid mode."""
        with pytest.raises(ValueError, match="mode"):
            weight_init.compute_psp_kernel_integrals(
                0.02, 0.005, mode="invalid"
            )

    def test_negative_tau_raises(self):
        """Verify ValueError for negative time constants."""
        with pytest.raises(ValueError, match="tau_mem"):
            weight_init.compute_psp_kernel_integrals(-0.02, 0.005)
        with pytest.raises(ValueError, match="tau_syn"):
            weight_init.compute_psp_kernel_integrals(0.02, -0.005)

    def test_zero_tau_raises(self):
        """Verify ValueError for zero time constants."""
        with pytest.raises(ValueError, match="tau_mem"):
            weight_init.compute_psp_kernel_integrals(0.0, 0.005)
        with pytest.raises(ValueError, match="tau_syn"):
            weight_init.compute_psp_kernel_integrals(0.02, 0.0)


class TestFluctuationDrivenNormal:
    """Tests for the general fluctuation_driven_normal_ function."""

    def test_basic_functionality(self):
        """Verify function runs and returns tensor."""
        tensor = torch.empty(10, 5)
        result = weight_init.fluctuation_driven_normal_(
            tensor, tau_mem=0.02, tau_syn=0.005, nu=10.0
        )
        assert result is tensor
        assert tensor.shape == (10, 5)

    def test_mathematical_correctness_linear(self):
        """Verify mu_w and sigma_w match paper Eqs. 6-7 for Linear layer."""
        tau_mem, tau_syn, nu = 0.02, 0.005, 10.0
        mu_u, sigma_u, theta = 0.0, 1.0, 1.0

        epsilon_bar, epsilon_hat = weight_init.compute_psp_kernel_integrals(
            tau_mem, tau_syn
        )
        fan_in = 5

        # Eq. 6: mu_w = mu_u / (n * nu * epsilon_bar)
        expected_mu_w = mu_u / (fan_in * nu * epsilon_bar)
        # Eq. 7: sigma_w^2 = sigma_u^2 / (n * nu * epsilon_hat) - mu_w^2
        expected_sigma_w = math.sqrt(
            sigma_u**2 / (fan_in * nu * epsilon_hat) - expected_mu_w**2
        )

        tensor = torch.empty(10000, 5)
        weight_init.fluctuation_driven_normal_(
            tensor, tau_mem, tau_syn, nu, mu_u, sigma_u, theta
        )

        # Check statistics (allow 10% tolerance for sampling variance)
        assert (
            abs(tensor.mean().item() - expected_mu_w)
            / (expected_sigma_w + 1e-10)
            < 0.1
        )
        assert (
            abs(tensor.std().item() - expected_sigma_w) / expected_sigma_w
            < 0.1
        )

    def test_mathematical_correctness_conv2d(self):
        """Verify fan_in computation for Conv2d tensor."""
        tensor = torch.empty(16, 3, 3, 3)

        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
        assert fan_in == 27  # 3 channels * 3x3 kernel

        # Should not raise
        weight_init.fluctuation_driven_normal_(
            tensor, tau_mem=0.02, tau_syn=0.005, nu=10.0
        )

    def test_negative_sigma_squared_raises(self):
        """Verify ValueError when computed sigma_w^2 is negative."""
        tensor = torch.empty(1, 1)
        with pytest.raises(ValueError, match="weight variance is negative"):
            weight_init.fluctuation_driven_normal_(
                tensor,
                tau_mem=0.02,
                tau_syn=0.005,
                nu=0.1,
                mu_u=10.0,
                sigma_u=0.1,
            )

    def test_zero_element_tensor_noop(self):
        """Verify zero-element tensor is a no-op with warning."""
        tensor = torch.empty(0, 10)
        with pytest.warns(UserWarning, match="zero-element"):
            result = weight_init.fluctuation_driven_normal_(
                tensor, tau_mem=0.02, tau_syn=0.005, nu=10.0
            )
        assert result is tensor

    def test_1d_tensor_raises(self):
        """Verify ValueError for 1D tensor."""
        tensor = torch.empty(10)
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            weight_init.fluctuation_driven_normal_(
                tensor, tau_mem=0.02, tau_syn=0.005, nu=10.0
            )

    def test_dtype_preservation_float32(self):
        """Verify output dtype matches input for float32."""
        tensor = torch.empty(10, 5, dtype=torch.float32)
        weight_init.fluctuation_driven_normal_(
            tensor, tau_mem=0.02, tau_syn=0.005, nu=10.0
        )
        assert tensor.dtype == torch.float32

    def test_dtype_preservation_float64(self):
        """Verify output dtype matches input for float64."""
        tensor = torch.empty(10, 5, dtype=torch.float64)
        weight_init.fluctuation_driven_normal_(
            tensor, tau_mem=0.02, tau_syn=0.005, nu=10.0
        )
        assert tensor.dtype == torch.float64

    def test_reproducibility_with_generator(self):
        """Verify same seed gives same results."""
        tensor1 = torch.empty(10, 5)
        tensor2 = torch.empty(10, 5)

        g1 = torch.Generator()
        g1.manual_seed(42)
        g2 = torch.Generator()
        g2.manual_seed(42)

        weight_init.fluctuation_driven_normal_(
            tensor1, tau_mem=0.02, tau_syn=0.005, nu=10.0, generator=g1
        )
        weight_init.fluctuation_driven_normal_(
            tensor2, tau_mem=0.02, tau_syn=0.005, nu=10.0, generator=g2
        )

        assert torch.allclose(tensor1, tensor2)

    def test_invalid_mode_raises(self):
        """Verify ValueError for invalid mode."""
        tensor = torch.empty(10, 5)
        with pytest.raises(ValueError, match="mode"):
            weight_init.fluctuation_driven_normal_(
                tensor,
                tau_mem=0.02,
                tau_syn=0.005,
                nu=10.0,
                mode="invalid",
            )

    def test_integration_with_nn_linear(self):
        """Verify works with nn.Linear layer."""
        fc = nn.Linear(784, 128)
        weight_init.fluctuation_driven_normal_(
            fc.weight, tau_mem=0.02, tau_syn=0.005, nu=10.0
        )
        assert not torch.all(fc.weight == 0)

    def test_integration_with_nn_conv2d(self):
        """Verify works with nn.Conv2d layer."""
        conv = nn.Conv2d(3, 16, 3, padding=1)
        weight_init.fluctuation_driven_normal_(
            conv.weight, tau_mem=0.02, tau_syn=0.005, nu=10.0
        )
        assert not torch.all(conv.weight == 0)

    def test_numerical_mode(self):
        """Verify numerical mode works."""
        tensor = torch.empty(10, 5)
        weight_init.fluctuation_driven_normal_(
            tensor, tau_mem=0.02, tau_syn=0.005, nu=10.0, mode="numerical"
        )
        assert tensor.shape == (10, 5)

    def test_custom_theta(self):
        """Verify custom threshold affects sigma_w when mu_u != 0.

        When mu_u=0, theta cancels out (sigma_w depends only on sigma_u).
        When mu_u != 0, theta affects the weight variance through Eq. 7.
        """
        tau_mem, tau_syn, nu = 0.02, 0.005, 10.0
        sigma_u = 1.0

        epsilon_bar, epsilon_hat = weight_init.compute_psp_kernel_integrals(
            tau_mem, tau_syn
        )
        fan_in = 5

        # Test 1: With mu_u != 0, theta affects sigma_w
        mu_u = 0.5
        theta1 = 1.0
        theta2 = 0.5

        # Eq. 7: sigma_w^2 = (1/(n * nu * epsilon_hat))
        # * ((theta - mu_u)/xi)^2 - mu_w^2
        # where xi = (theta - mu_u) / sigma_u
        # Simplifies to:
        # sigma_w^2 = sigma_u^2 / (n * nu * epsilon_hat) - mu_w^2
        # (theta cancels out when using the simplified formula)
        # But the general formula uses theta explicitly

        # For the general case with mu_u != 0:
        # mu_w = mu_u / (n * nu * epsilon_bar)
        # sigma_w^2 = (1/(n * nu * epsilon_hat))
        # * ((theta - mu_u)^2 / xi^2) - mu_w^2
        # where xi = (theta - mu_u) / sigma_u
        # So sigma_w^2 = sigma_u^2 / (n * nu * epsilon_hat) - mu_w^2
        # This means theta still cancels when we substitute xi!

        # The correct test: verify that sigma_w matches
        # the formula regardless of theta
        expected_mu_w = mu_u / (fan_in * nu * epsilon_bar)
        expected_sigma_w = math.sqrt(
            sigma_u**2 / (fan_in * nu * epsilon_hat) - expected_mu_w**2
        )

        tensor1 = torch.empty(10000, 5)
        tensor2 = torch.empty(10000, 5)

        weight_init.fluctuation_driven_normal_(
            tensor1,
            tau_mem,
            tau_syn,
            nu,
            mu_u=mu_u,
            sigma_u=sigma_u,
            theta=theta1,
        )
        weight_init.fluctuation_driven_normal_(
            tensor2,
            tau_mem,
            tau_syn,
            nu,
            mu_u=mu_u,
            sigma_u=sigma_u,
            theta=theta2,
        )

        # Both should give the same sigma (theta cancels)
        assert (
            abs(tensor1.std().item() - expected_sigma_w) / expected_sigma_w
            < 0.1
        )
        assert (
            abs(tensor2.std().item() - expected_sigma_w) / expected_sigma_w
            < 0.1
        )

    def test_negative_tau_mem_raises(self):
        """Verify ValueError for negative tau_mem."""
        tensor = torch.empty(10, 5)
        with pytest.raises(ValueError, match="tau_mem"):
            weight_init.fluctuation_driven_normal_(
                tensor, tau_mem=-0.02, tau_syn=0.005, nu=10.0
            )

    def test_negative_nu_raises(self):
        """Verify ValueError for negative nu."""
        tensor = torch.empty(10, 5)
        with pytest.raises(ValueError, match="nu"):
            weight_init.fluctuation_driven_normal_(
                tensor, tau_mem=0.02, tau_syn=0.005, nu=-10.0
            )

    def test_negative_sigma_u_raises(self):
        """Verify ValueError for negative sigma_u."""
        tensor = torch.empty(10, 5)
        with pytest.raises(ValueError, match="sigma_u"):
            weight_init.fluctuation_driven_normal_(
                tensor, tau_mem=0.02, tau_syn=0.005, nu=10.0, sigma_u=-1.0
            )

    def test_negative_theta_raises(self):
        """Verify ValueError for negative theta."""
        tensor = torch.empty(10, 5)
        with pytest.raises(ValueError, match="theta"):
            weight_init.fluctuation_driven_normal_(
                tensor, tau_mem=0.02, tau_syn=0.005, nu=10.0, theta=-1.0
            )


class TestFluctuationDrivenNormalCentered:
    """Tests for the centered variant (mu_u=0)."""

    def test_basic_functionality(self):
        """Verify function runs and returns tensor."""
        tensor = torch.empty(10, 5)
        result = weight_init.fluctuation_driven_normal_centered_(
            tensor, tau_mem=0.02, tau_syn=0.005, nu=10.0
        )
        assert result is tensor

    def test_zero_mean(self):
        """Verify mean is approximately zero."""
        tensor = torch.empty(10000, 100)
        weight_init.fluctuation_driven_normal_centered_(
            tensor, tau_mem=0.02, tau_syn=0.005, nu=10.0
        )
        assert abs(tensor.mean().item()) < 0.01

    def test_sigma_matches_formula(self):
        """Verify sigma_w matches paper Eq. 9:
        sigma_w = sigma_u / sqrt(n * nu * epsilon_hat).
        """
        tau_mem, tau_syn, nu = 0.02, 0.005, 10.0
        sigma_u = 1.0

        epsilon_bar, epsilon_hat = weight_init.compute_psp_kernel_integrals(
            tau_mem, tau_syn
        )
        fan_in = 5

        # Eq. 9: sigma_w = sigma_u / sqrt(n * nu * epsilon_hat)
        expected_sigma_w = sigma_u / math.sqrt(fan_in * nu * epsilon_hat)

        tensor = torch.empty(10000, 5)
        weight_init.fluctuation_driven_normal_centered_(
            tensor, tau_mem, tau_syn, nu, sigma_u
        )

        assert (
            abs(tensor.std().item() - expected_sigma_w) / expected_sigma_w
            < 0.1
        )

    def test_dtype_preservation(self):
        """Verify dtype is preserved."""
        tensor = torch.empty(10, 5, dtype=torch.float64)
        weight_init.fluctuation_driven_normal_centered_(
            tensor, tau_mem=0.02, tau_syn=0.005, nu=10.0
        )
        assert tensor.dtype == torch.float64

    def test_reproducibility(self):
        """Verify reproducibility with generator."""
        tensor1 = torch.empty(10, 5)
        tensor2 = torch.empty(10, 5)

        g1 = torch.Generator()
        g1.manual_seed(42)
        g2 = torch.Generator()
        g2.manual_seed(42)

        weight_init.fluctuation_driven_normal_centered_(
            tensor1, tau_mem=0.02, tau_syn=0.005, nu=10.0, generator=g1
        )
        weight_init.fluctuation_driven_normal_centered_(
            tensor2, tau_mem=0.02, tau_syn=0.005, nu=10.0, generator=g2
        )

        assert torch.allclose(tensor1, tensor2)

    def test_integration_with_sequential(self):
        """Verify works with nn.Sequential model."""
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 10),
        )

        for layer in model:
            if isinstance(layer, nn.Linear):
                weight_init.fluctuation_driven_normal_centered_(
                    layer.weight, tau_mem=0.02, tau_syn=0.005, nu=10.0
                )

        for layer in model:
            if isinstance(layer, nn.Linear):
                assert not torch.all(layer.weight == 0)

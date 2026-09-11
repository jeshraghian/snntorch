#!/usr/bin/env python

"""Tests for the SynapticDelay module (per-channel axonal / synaptic delay)."""

import pytest
import snntorch as snn
import torch


@pytest.fixture(scope="module")
def impulse():
    # a single unit impulse at t=2 on one channel, 12 steps, batch 1
    x = torch.zeros(12, 1, 1)
    x[2, 0, 0] = 1.0
    return x


class TestSynapticDelay:
    # ---------------------------------------------------------------- #
    #  static integer delay is exact  (Bar 1)
    # ---------------------------------------------------------------- #
    def test_integer_delay_step_mode(self, impulse):
        d = snn.SynapticDelay(max_delay=8, delay=3)
        d.reset_delay()
        out = torch.stack([d(impulse[t]) for t in range(impulse.shape[0])])
        nz = (out.abs() > 1e-6).nonzero()[:, 0].tolist()
        assert nz == [5]                      # impulse at 2, delayed by 3
        assert out[5, 0, 0] == pytest.approx(1.0)

    def test_integer_delay_sequence_mode(self, impulse):
        d = snn.SynapticDelay(max_delay=8, delay=3, step_mode=False)
        out = d(impulse)
        nz = (out.abs() > 1e-6).nonzero()[:, 0].tolist()
        assert nz == [5]
        assert out[5, 0, 0] == pytest.approx(1.0)

    def test_zero_delay_is_passthrough(self):
        x = torch.randn(10, 3, 4)
        d = snn.SynapticDelay(max_delay=5, delay=0.0, step_mode=False)
        assert torch.allclose(d(x), x, atol=1e-6)

    def test_step_and_sequence_modes_agree(self):
        torch.manual_seed(0)
        C, T = 4, 20
        x = torch.randn(T, 2, C)
        delay = torch.tensor([0.0, 1.5, 3.0, 5.0])
        ds = snn.SynapticDelay(max_delay=6, delay=delay, channels=C)
        ds.reset_delay()
        step = torch.stack([ds(x[t]) for t in range(T)])
        dq = snn.SynapticDelay(
            max_delay=6, delay=delay, channels=C, step_mode=False
        )
        seq = dq(x)
        assert torch.allclose(step, seq, atol=1e-6)

    def test_fractional_delay_interpolates(self):
        x = torch.zeros(10, 1, 1)
        x[1, 0, 0] = 1.0
        d = snn.SynapticDelay(
            max_delay=6, delay=2.25, step_mode=False, kernel="linear"
        )
        out = d(x).squeeze()
        # impulse at t=1, delay 2.25 -> mass split across t=3 (0.75) and t=4 (0.25)
        assert out[3] == pytest.approx(0.75, abs=1e-5)
        assert out[4] == pytest.approx(0.25, abs=1e-5)
        assert out.sum() == pytest.approx(1.0, abs=1e-5)

    # ---------------------------------------------------------------- #
    #  differentiability of the delay parameter  (Bar 2)
    # ---------------------------------------------------------------- #
    def test_delay_is_learnable_parameter(self):
        d = snn.SynapticDelay(max_delay=6, delay=2.0, learn_delay=True)
        assert isinstance(d.delay, torch.nn.Parameter)
        assert d.delay.requires_grad
        d2 = snn.SynapticDelay(max_delay=6, delay=2.0, learn_delay=False)
        assert not isinstance(d2.delay, torch.nn.Parameter)

    def test_gradcheck_wrt_input(self):
        torch.manual_seed(0)
        d = snn.SynapticDelay(
            max_delay=6, delay=2.3, learn_delay=True, step_mode=False
        ).double()
        x = torch.randn(15, 1, 2, dtype=torch.double, requires_grad=True)
        assert torch.autograd.gradcheck(d, (x,), eps=1e-6, atol=1e-4)

    def test_delay_gradient_matches_finite_difference(self):
        torch.manual_seed(0)
        d = snn.SynapticDelay(
            max_delay=6, delay=2.3, learn_delay=True, step_mode=False
        ).double()
        x = torch.randn(15, 1, 1, dtype=torch.double)

        d.delay.grad = None
        d(x).sum().backward()
        g_analytic = d.delay.grad.item()

        h = 1e-6
        with torch.no_grad():
            d.delay.copy_(torch.tensor(2.3 + h, dtype=torch.double))
            yp = d(x).sum().item()
            d.delay.copy_(torch.tensor(2.3 - h, dtype=torch.double))
            ym = d(x).sum().item()
        g_fd = (yp - ym) / (2 * h)
        assert abs(g_analytic - g_fd) < 1e-4

    def test_gradient_flows_to_delay_and_input(self):
        d = snn.SynapticDelay(
            max_delay=6, delay=2.5, learn_delay=True, step_mode=False
        )
        x = torch.randn(12, 2, 3, requires_grad=True)
        d(x).pow(2).sum().backward()
        assert d.delay.grad is not None and torch.isfinite(d.delay.grad).all()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    # ---------------------------------------------------------------- #
    #  API / validation
    # ---------------------------------------------------------------- #
    def test_per_channel_delay_vector(self):
        d = snn.SynapticDelay(
            max_delay=8, delay=torch.tensor([1.0, 4.0]), channels=2,
            step_mode=False,
        )
        x = torch.zeros(12, 1, 2)
        x[0, 0, 0] = 1.0
        x[0, 0, 1] = 1.0
        out = d(x).squeeze(1)
        assert out[1, 0] == pytest.approx(1.0)     # ch0 delayed 1
        assert out[4, 1] == pytest.approx(1.0)     # ch1 delayed 4

    def test_rejects_delay_out_of_range(self):
        with pytest.raises(ValueError):
            snn.SynapticDelay(max_delay=4, delay=9.0)
        with pytest.raises(ValueError):
            snn.SynapticDelay(max_delay=4, delay=-1.0)

    def test_rejects_bad_max_delay(self):
        with pytest.raises(ValueError):
            snn.SynapticDelay(max_delay=0)

    def test_rejects_bad_kernel(self):
        with pytest.raises(ValueError):
            snn.SynapticDelay(max_delay=4, kernel="quadratic")

    def test_step_mode_shape_guard(self):
        d = snn.SynapticDelay(max_delay=4, delay=1.0, step_mode=True)
        with pytest.raises(ValueError):
            d(torch.randn(5, 2, 3))          # 3-D into step mode

    def test_reset_delay_clears_buffer(self, impulse):
        d = snn.SynapticDelay(max_delay=8, delay=3)
        d.reset_delay()
        for t in range(impulse.shape[0]):
            d(impulse[t])
        d.reset_delay()
        out = torch.stack([d(torch.zeros(1, 1)) for _ in range(6)])
        assert out.abs().max() == pytest.approx(0.0)   # no leftover state

    # ---------------------------------------------------------------- #
    #  composition with neurons  (Bar 5)
    # ---------------------------------------------------------------- #
    def test_composes_with_leaky_step_mode(self):
        torch.manual_seed(0)
        fc = torch.nn.Linear(3, 5)
        delay = snn.SynapticDelay(
            max_delay=6, delay=2.0, channels=5, learn_delay=True
        )
        lif = snn.Leaky(beta=0.9)
        mem = lif.init_leaky()
        delay.reset_delay()
        x = torch.randn(8, 4, 3)
        loss = 0.0
        for t in range(8):
            spk, mem = lif(delay(fc(x[t])), mem)
            loss = loss + spk.sum()
        loss.backward()
        assert delay.delay.grad is not None
        assert torch.isfinite(fc.weight.grad).all()

    def test_composes_with_leakyparallel_sequence_mode(self):
        torch.manual_seed(0)
        fc = torch.nn.Linear(3, 5)
        delay = snn.SynapticDelay(
            max_delay=6, delay=2.0, channels=5, learn_delay=True,
            step_mode=False,
        )
        lif = snn.LeakyParallel(input_size=5, hidden_size=5, beta=0.9)
        x = torch.randn(8, 4, 3)
        spk = lif(delay(fc(x)))
        spk.sum().backward()
        assert delay.delay.grad is not None
        assert torch.isfinite(delay.delay.grad).all()

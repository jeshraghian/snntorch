#!/usr/bin/env python

"""Tests for the homeostatic (adaptive, time-varying) threshold, added to
SpikingNeuron and inherited by every neuron model (issue #37)."""

import math

import pytest
import snntorch as snn
import torch


class TestHomeostasis:
    # ---------------------------------------------------------------- #
    #  disabled by default -> zero regression
    # ---------------------------------------------------------------- #
    def test_disabled_by_default(self):
        lif = snn.Leaky(beta=0.9)
        assert lif._homeostasis_enabled is False

    def test_matches_stock_behaviour_when_disabled(self):
        torch.manual_seed(0)
        x = torch.rand(20, 4, 3) * 3
        lif_a = snn.Leaky(beta=0.9)
        lif_b = snn.Leaky(beta=0.9)
        mem_a = lif_a.init_leaky()
        mem_b = lif_b.init_leaky()
        for t in range(20):
            spk_a, mem_a = lif_a(x[t], mem_a)
            spk_b, mem_b = lif_b(x[t], mem_b)
            assert torch.equal(spk_a, spk_b)
            assert torch.equal(mem_a, mem_b)

    # ---------------------------------------------------------------- #
    #  raises the effective threshold on firing  (Bar: SFA)
    # ---------------------------------------------------------------- #
    def test_threshold_rises_after_spiking(self):
        lif = snn.Leaky(beta=0.9)
        lif.enable_homeostasis(increment=1.0, tau=20.0)
        mem = lif.init_leaky()
        spk, mem = lif(torch.tensor([[2.0]]), mem)
        assert spk.item() == 1.0
        assert lif._threshold_adapt.item() == pytest.approx(1.0)

    def test_adaptation_suppresses_firing_rate_under_constant_drive(self):
        torch.manual_seed(0)
        x = torch.full((1, 8), 2.0)

        lif = snn.Leaky(beta=0.9)
        lif.enable_homeostasis(increment=1.2, tau=25.0)
        mem = lif.init_leaky()
        rates = []
        for t in range(200):
            spk, mem = lif(x, mem)
            rates.append(spk.mean().item())
        early = sum(rates[:20]) / 20
        late = sum(rates[180:200]) / 20
        # spike-frequency adaptation: firing rate under constant drive
        # settles LOWER than it started, and stays there (steady state)
        assert late < early
        assert sum(rates[100:120]) / 20 == pytest.approx(late, abs=0.1)

        # a non-adaptive control fires at a constant (non-decreasing) rate
        lif_ctrl = snn.Leaky(beta=0.9)
        mem_c = lif_ctrl.init_leaky()
        rates_c = []
        for t in range(200):
            spk, mem_c = lif_ctrl(x, mem_c)
            rates_c.append(spk.mean().item())
        early_c = sum(rates_c[:20]) / 20
        late_c = sum(rates_c[180:200]) / 20
        assert late_c == pytest.approx(early_c, abs=1e-6)

    # ---------------------------------------------------------------- #
    #  decays back to steady state  (Bar: decay time-constant)
    # ---------------------------------------------------------------- #
    def test_decays_exponentially_with_tau_when_drive_stops(self):
        lif = snn.Leaky(beta=0.9)
        lif.enable_homeostasis(increment=2.0, tau=10.0)
        mem = lif.init_leaky()
        for _ in range(20):
            _, mem = lif(torch.tensor([[2.0]]), mem)
        a0 = lif._threshold_adapt.clone()

        zero = torch.tensor([[0.0]])
        for _ in range(15):
            _, mem = lif(zero, mem)
        a15 = lif._threshold_adapt.item()

        expected = a0.item() * math.exp(-15.0 / 10.0)
        assert a15 == pytest.approx(expected, rel=1e-4)

    # ---------------------------------------------------------------- #
    #  saturates  (Bar: max_adapt)
    # ---------------------------------------------------------------- #
    def test_saturates_at_max_adapt(self):
        lif = snn.Leaky(beta=0.99)
        lif.enable_homeostasis(increment=5.0, tau=1000.0, max_adapt=3.0)
        mem = lif.init_leaky()
        strong = torch.tensor([[50.0]])
        for _ in range(50):
            _, mem = lif(strong, mem)
        assert lif._threshold_adapt.item() <= 3.0 + 1e-6

    def test_no_cap_when_max_adapt_none(self):
        lif = snn.Leaky(beta=0.99)
        lif.enable_homeostasis(increment=5.0, tau=1000.0, max_adapt=None)
        mem = lif.init_leaky()
        strong = torch.tensor([[50.0]])
        for _ in range(20):
            _, mem = lif(strong, mem)
        assert lif._threshold_adapt.item() > 3.0

    # ---------------------------------------------------------------- #
    #  reset / disable
    # ---------------------------------------------------------------- #
    def test_reset_homeostasis_clears_state(self):
        lif = snn.Leaky(beta=0.9)
        lif.enable_homeostasis(increment=1.0, tau=10.0)
        mem = lif.init_leaky()
        for _ in range(10):
            _, mem = lif(torch.tensor([[2.0]]), mem)
        assert lif._threshold_adapt.item() > 0
        lif.reset_homeostasis()
        assert lif._threshold_adapt.item() == 0.0

    def test_disable_homeostasis_reverts_to_static_threshold(self):
        lif = snn.Leaky(beta=0.9)
        lif.enable_homeostasis(increment=1.0, tau=10.0)
        mem = lif.init_leaky()
        for _ in range(10):
            _, mem = lif(torch.tensor([[2.0]]), mem)
        assert lif._threshold_adapt.item() > 0
        lif.disable_homeostasis()
        # once disabled, fire() uses the static threshold again -- the
        # (retained) adaptive component is simply not applied
        spk, mem2 = lif(torch.tensor([[2.0]]), mem)
        lif2 = snn.Leaky(beta=0.9)
        spk2, _ = lif2(torch.tensor([[2.0]]), mem)
        assert torch.equal(spk, spk2)

    # ---------------------------------------------------------------- #
    #  available on every neuron model, unmodified files  (the actual ask)
    # ---------------------------------------------------------------- #
    @pytest.mark.parametrize(
        "make_neuron",
        [
            lambda: snn.Leaky(beta=0.9),
            lambda: snn.Synaptic(alpha=0.9, beta=0.8),
            lambda: snn.Lapicque(beta=0.9),
        ],
    )
    def test_enable_homeostasis_available_on_every_lif_variant(
        self, make_neuron
    ):
        neuron = make_neuron()
        assert hasattr(neuron, "enable_homeostasis")
        neuron.enable_homeostasis(increment=0.5, tau=15.0)
        assert neuron._homeostasis_enabled is True

    # ---------------------------------------------------------------- #
    #  detached from the graph (per the issue's own speculation)
    # ---------------------------------------------------------------- #
    def test_adaptive_component_does_not_require_grad(self):
        lif = snn.Leaky(beta=0.9, learn_beta=True)
        lif.enable_homeostasis(increment=1.0, tau=10.0)
        mem = lif.init_leaky()
        x = torch.tensor([[2.0]], requires_grad=True)
        spk, mem = lif(x, mem)
        assert lif._threshold_adapt.requires_grad is False
        spk.sum().backward()
        assert x.grad is not None
        assert lif.beta.grad is not None

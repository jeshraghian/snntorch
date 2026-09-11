#!/usr/bin/env python

"""Tests for neuron_from_equations() / CustomNeuron (issue #79 "custom
neuron generators")."""

import pytest
import snntorch as snn
import torch


def _leaky_update(states, I):
    return {"mem": 0.9 * states["mem"] + I}


class TestCustomNeuronGenerator:
    # ---------------------------------------------------------------- #
    #  correctness of the generated update / reset math
    # ---------------------------------------------------------------- #
    def test_matches_leaky_with_immediate_reset(self):
        """A 1-state generated neuron running Leaky's own beta*mem+I update
        must match snn.Leaky(reset_delay=False) bit-for-bit -- CustomNeuron
        always resets in the same step as the firing decision (see the
        class docstring's note on reset timing)."""
        MyLeaky = snn.neuron_from_equations(
            "MyLeaky", state_names=("mem",), update_fn=_leaky_update
        )
        custom = MyLeaky(threshold=1.0)
        ref = snn.Leaky(beta=0.9, reset_delay=False)

        torch.manual_seed(0)
        x = torch.rand(20, 3, 4) * 0.6
        mem_c = torch.zeros_like(x[0])
        mem_r = ref.init_leaky()
        for t in range(20):
            spk_c, mem_c = custom(x[t], mem_c)
            spk_r, mem_r = ref(x[t], mem_r)
            assert torch.allclose(spk_c, spk_r, atol=1e-5)
            assert torch.allclose(mem_c, mem_r, atol=1e-5)

    @pytest.mark.parametrize("mechanism", ["subtract", "zero", "none"])
    def test_reset_mechanisms_are_correct(self, mechanism):
        MyLeaky = snn.neuron_from_equations(
            "MyLeaky", state_names=("mem",), update_fn=_leaky_update
        )
        neuron = MyLeaky(threshold=1.0, reset_mechanism=mechanism)

        torch.manual_seed(1)
        x = torch.rand(15, 1) * 0.6
        mem_ref = torch.zeros(1)
        mem = torch.zeros(1)
        for t in range(15):
            mem_ref = 0.9 * mem_ref + x[t]
            s = (mem_ref >= 1.0).float()
            if mechanism == "subtract":
                mem_ref = mem_ref - s * 1.0
            elif mechanism == "zero":
                mem_ref = (1 - s) * mem_ref
            # "none": no reset

            spk, mem = neuron(x[t], mem)
            assert spk.item() == pytest.approx(s.item())
            assert mem.item() == pytest.approx(mem_ref.item(), abs=1e-5)

    def test_multistate_neuron_with_custom_reset_fn(self):
        """A genuinely different neuron (2 states, non-subtract reset) --
        Izhikevich, expressed purely through the generator, no bespoke
        subclass. Must show burst-capable dynamics an LIF cannot."""

        def izh_update(states, I):
            v, u = states["v"], states["u"]
            dv = 0.04 * v ** 2 + 5 * v + 140 - u + I
            du = 0.02 * (0.2 * v - u)
            return {"v": v + dv, "u": u + du}

        def izh_reset(states, spk, threshold):
            v, u = states["v"], states["u"]
            fired = spk.detach().bool()
            v = torch.where(fired, torch.full_like(v, -65.0), v)
            u = torch.where(fired, u + 8.0, u)
            return {"v": v, "u": u}

        Izhikevich = snn.neuron_from_equations(
            "Izhikevich",
            state_names=("v", "u"),
            update_fn=izh_update,
            spike_state="v",
            reset_fn=izh_reset,
            init_values={"v": -65.0, "u": -13.0},
        )
        izh = Izhikevich(threshold=30.0)
        v = torch.tensor([-65.0])
        u = torch.tensor([-13.0])
        spikes = 0
        for _ in range(300):
            spk, v, u = izh(torch.tensor([10.0]), v, u)
            spikes += spk.item()
        # a real Izhikevich neuron at this current bursts a handful of
        # times over 300 steps; it must not saturate (fire every step,
        # like a mis-wired LIF would) or stay silent (dead neuron)
        assert 1 <= spikes <= 60

    # ---------------------------------------------------------------- #
    #  API: init_hidden / output / inhibition / gradients / validation
    # ---------------------------------------------------------------- #
    def test_init_hidden_and_output(self):
        Hid = snn.neuron_from_equations(
            "Hid", state_names=("mem",), update_fn=_leaky_update
        )
        neuron = Hid(threshold=1.0, init_hidden=True, output=True)
        for _ in range(5):
            spk, mem = neuron(torch.rand(2, 3))
        assert spk.shape == (2, 3)
        assert mem.shape == (2, 3)

    def test_init_hidden_rejects_explicit_state(self):
        Hid = snn.neuron_from_equations(
            "Hid", state_names=("mem",), update_fn=_leaky_update
        )
        neuron = Hid(threshold=1.0, init_hidden=True)
        with pytest.raises(TypeError):
            neuron(torch.rand(2, 3), torch.zeros(2, 3))

    def test_gradients_flow_to_input_and_threshold(self):
        MyLeaky = snn.neuron_from_equations(
            "MyLeaky", state_names=("mem",), update_fn=_leaky_update
        )
        neuron = MyLeaky(threshold=1.0, learn_threshold=True)
        mem = torch.zeros(1)
        x = torch.tensor([2.0], requires_grad=True)
        spk, mem = neuron(x, mem)
        spk.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert neuron.threshold.grad is not None

    def test_reset_state_zeros_everything(self):
        def two_state_update(states, I):
            return {"v": states["v"] + I, "u": states["u"] + I}

        Izh = snn.neuron_from_equations(
            "Izh", state_names=("v", "u"), update_fn=two_state_update,
            init_values={"v": -65.0, "u": -13.0},
        )
        neuron = Izh(threshold=30.0)
        v, u = torch.tensor([-65.0]), torch.tensor([-13.0])
        _, v, u = neuron(torch.tensor([1.0]), v, u)
        v0, u0 = neuron.reset_state()
        assert v0.item() == 0.0 and u0.item() == 0.0

    def test_rejects_missing_state_names(self):
        with pytest.raises(TypeError):
            snn.CustomNeuron()

    def test_rejects_bad_spike_state(self):
        with pytest.raises(ValueError):
            snn.neuron_from_equations(
                "Bad", state_names=("mem",), update_fn=_leaky_update,
                spike_state="not_a_state",
            )()

    def test_rejects_wrong_state_count(self):
        MyLeaky = snn.neuron_from_equations(
            "MyLeaky", state_names=("mem",), update_fn=_leaky_update
        )
        neuron = MyLeaky(threshold=1.0)
        with pytest.raises(TypeError):
            neuron(torch.rand(2, 3), torch.zeros(2, 3), torch.zeros(2, 3))

    def test_update_fn_must_return_every_state(self):
        def broken_update(states, I):
            return {}  # missing 'mem'

        Bad = snn.neuron_from_equations(
            "Bad", state_names=("mem",), update_fn=broken_update
        )
        neuron = Bad(threshold=1.0)
        with pytest.raises(ValueError):
            neuron(torch.rand(2, 3), torch.zeros(2, 3))

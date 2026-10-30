#!/usr/bin/env python

"""Tests for Synaptic neuron."""

import pytest
import snntorch as snn
import torch
import torch._dynamo as dynamo

RESET_MECHANISMS = ["subtract", "zero", "none"]


@pytest.fixture(scope="module")
def input_():
    return torch.Tensor([0.25, 0]).unsqueeze(-1)


@pytest.fixture(scope="module")
def spiking_input_():
    return torch.Tensor([1.2, 0]).unsqueeze(-1)


@pytest.fixture(scope="module")
def synaptic_instance():
    return snn.Synaptic(alpha=0.5, beta=0.5)


@pytest.fixture(scope="module")
def synaptic_instance_surrogate():
    return snn.Synaptic(alpha=0.5, beta=0.5, surrogate_disable=True)


@pytest.fixture(scope="module")
def synaptic_hidden_instance():
    return snn.Synaptic(alpha=0.5, beta=0.5, init_hidden=True)


class TestSynaptic:
    def test_synaptic(self, synaptic_instance, input_):
        syn, mem = synaptic_instance.init_synaptic()

        syn_rec = []
        mem_rec = []
        spk_rec = []

        for i in range(2):
            spk, syn, mem = synaptic_instance(input_[i], syn, mem)
            syn_rec.append(syn)
            mem_rec.append(mem)
            spk_rec.append(spk)

        assert syn_rec[0] == 2 * syn_rec[1]
        assert mem_rec[1] == mem_rec[0] * 0.5 + syn_rec[1]
        assert spk_rec[0] == spk_rec[1]

    @pytest.mark.parametrize(
        ("mechanism", "reset_delay", "expected_mem"),
        [
            ("subtract", True, [1.2, 0.62]),
            ("zero", True, [1.2, 0.6]),
            ("none", True, [1.2, 0.72]),
            ("subtract", False, [0.2, 0.62]),
            ("zero", False, [0.0, 0.6]),
            ("none", False, [1.2, 0.72]),
        ],
    )
    def test_synaptic_reset(
        self, mechanism, reset_delay, expected_mem, spiking_input_
    ):
        neuron = snn.Synaptic(
            alpha=0.5,
            beta=0.1,
            reset_mechanism=mechanism,
            reset_delay=reset_delay,
        )
        syn = torch.zeros(1)
        mem = torch.zeros(1)

        for i, expected in enumerate(expected_mem):
            spk, syn, mem = neuron(spiking_input_[i], syn, mem)

            assert spk.item() == pytest.approx(1 if i == 0 else 0)
            assert mem.item() == pytest.approx(expected)

    def test_synaptic_reset_setter(self, synaptic_instance):
        for val, mechanism in enumerate(RESET_MECHANISMS):
            synaptic_instance.reset_mechanism = mechanism
            assert synaptic_instance.reset_mechanism_val == val

    @pytest.mark.parametrize("mechanism", RESET_MECHANISMS)
    def test_synaptic_init_hidden(self, mechanism, spiking_input_):
        hidden_instance = snn.Synaptic(
            alpha=0.5,
            beta=0.1,
            init_hidden=True,
            reset_mechanism=mechanism,
        )
        spk_rec = []

        for i in range(2):
            spk_rec.append(hidden_instance(spiking_input_[i]))

        assert spk_rec[0].item() == pytest.approx(1)
        assert spk_rec[1].item() == pytest.approx(0)

    def test_synaptic_cases(self, synaptic_hidden_instance, input_):
        with pytest.raises(TypeError):
            synaptic_hidden_instance(input_, input_)

    def test_synaptic_compile_fullgraph(
        self, synaptic_instance_surrogate, input_
    ):
        explanation = dynamo.explain(synaptic_instance_surrogate)(input_[0])

        assert explanation.graph_break_count == 0

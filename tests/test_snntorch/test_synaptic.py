#!/usr/bin/env python

"""Tests for Synaptic neuron."""

import pytest
import snntorch as snn
import torch
import torch._dynamo as dynamo


@pytest.fixture(scope="module")
def input_():
    return torch.Tensor([0.25, 0]).unsqueeze(-1)


@pytest.fixture(scope="module")
def synaptic_instance():
    return snn.Synaptic(alpha=0.5, beta=0.5)

@pytest.fixture(scope="module")
def synaptic_instance_surrogate():
    return snn.Synaptic(alpha=0.5, beta=0.5, surrogate_disable=True)


@pytest.fixture(scope="module")
def synaptic_reset_zero_instance():
    return snn.Synaptic(alpha=0.5, beta=0.5, reset_mechanism="zero")


@pytest.fixture(scope="module")
def synaptic_reset_none_instance():
    return snn.Synaptic(alpha=0.5, beta=0.5, reset_mechanism="none")


@pytest.fixture(scope="module")
def synaptic_hidden_instance():
    return snn.Synaptic(alpha=0.5, beta=0.5, init_hidden=True)


@pytest.fixture(scope="module")
def synaptic_hidden_reset_zero_instance():
    return snn.Synaptic(
        alpha=0.5, beta=0.5, init_hidden=True, reset_mechanism="zero"
    )


@pytest.fixture(scope="module")
def synaptic_hidden_reset_none_instance():
    return snn.Synaptic(
        alpha=0.5, beta=0.5, init_hidden=True, reset_mechanism="none"
    )


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

    def test_synaptic_reset(
        self,
        synaptic_instance,
        synaptic_reset_zero_instance,
        synaptic_reset_none_instance,
    ):
        lif1 = synaptic_instance
        lif2 = synaptic_reset_zero_instance
        lif3 = synaptic_reset_none_instance

        assert lif1.reset_mechanism_val == 0
        assert lif2.reset_mechanism_val == 1
        assert lif3.reset_mechanism_val == 2

        lif1.reset_mechanism = "zero"
        lif2.reset_mechanism = "none"
        lif3.reset_mechanism = "subtract"

        assert lif1.reset_mechanism_val == 1
        assert lif2.reset_mechanism_val == 2
        assert lif3.reset_mechanism_val == 0

    def test_synaptic_init_hidden(self, synaptic_hidden_instance, input_):

        spk_rec = []

        for i in range(2):
            spk = synaptic_hidden_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_synaptic_init_hidden_reset_zero(
        self, synaptic_hidden_reset_zero_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = synaptic_hidden_reset_zero_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_synaptic_init_hidden_reset_none(
        self, synaptic_hidden_reset_none_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = synaptic_hidden_reset_none_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_synaptic_cases(self, synaptic_hidden_instance, input_):
        with pytest.raises(TypeError):
            synaptic_hidden_instance(input_, input_)

    def test_synaptic_compile_fullgraph(self, synaptic_instance_surrogate, input_):
        explanation = dynamo.explain(synaptic_instance_surrogate)(input_[0])

        assert explanation.graph_break_count == 0
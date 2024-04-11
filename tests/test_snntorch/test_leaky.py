#!/usr/bin/env python

"""Tests for Leaky neuron."""

import pytest
import snntorch as snn
import torch
import torch._dynamo as dynamo


@pytest.fixture(scope="module")
def input_():
    return torch.Tensor([0.25, 0]).unsqueeze(-1)


@pytest.fixture(scope="module")
def leaky_instance():
    return snn.Leaky(beta=0.5)


@pytest.fixture(scope="module")
def leaky_instance_surrogate():
    return snn.Leaky(beta=0.5, surrogate_disable=True)


@pytest.fixture(scope="module")
def leaky_reset_zero_instance():
    return snn.Leaky(beta=0.5, reset_mechanism="zero")


@pytest.fixture(scope="module")
def leaky_reset_none_instance():
    return snn.Leaky(beta=0.5, reset_mechanism="none")


@pytest.fixture(scope="module")
def leaky_hidden_instance():
    return snn.Leaky(beta=0.5, init_hidden=True)


@pytest.fixture(scope="module")
def leaky_hidden_reset_zero_instance():
    return snn.Leaky(beta=0.5, init_hidden=True, reset_mechanism="zero")


@pytest.fixture(scope="module")
def leaky_hidden_reset_none_instance():
    return snn.Leaky(beta=0.5, init_hidden=True, reset_mechanism="none")


@pytest.fixture(scope="module")
def leaky_hidden_learn_graded_instance():
    return snn.Leaky(
        beta=0.5, init_hidden=True, learn_graded_spikes_factor=True
    )


class TestLeaky:
    def test_leaky(self, leaky_instance, input_):
        mem = leaky_instance.init_leaky()

        mem_rec = []
        spk_rec = []

        for i in range(2):

            spk, mem = leaky_instance(input_[i], mem)
            mem_rec.append(mem)
            spk_rec.append(spk)

        assert mem_rec[1] == mem_rec[0] * 0.5 + input_[1]
        assert spk_rec[0] == spk_rec[1]

    def test_leaky_reset(
        self,
        leaky_instance,
        leaky_reset_zero_instance,
        leaky_reset_none_instance,
    ):
        lif1 = leaky_instance
        lif2 = leaky_reset_zero_instance
        lif3 = leaky_reset_none_instance

        assert lif1.reset_mechanism_val == 0
        assert lif2.reset_mechanism_val == 1
        assert lif3.reset_mechanism_val == 2

        lif1.reset_mechanism = "zero"
        lif2.reset_mechanism = "none"
        lif3.reset_mechanism = "subtract"

        assert lif1.reset_mechanism_val == 1
        assert lif2.reset_mechanism_val == 2
        assert lif3.reset_mechanism_val == 0

    def test_leaky_init_hidden(self, leaky_hidden_instance, input_):

        spk_rec = []

        for i in range(2):
            spk = leaky_hidden_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_leaky_init_hidden_reset_zero(
        self, leaky_hidden_reset_zero_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = leaky_hidden_reset_zero_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_leaky_init_hidden_reset_none(
        self, leaky_hidden_reset_none_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = leaky_hidden_reset_none_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_leaky_cases(self, leaky_hidden_instance, input_):
        with pytest.raises(TypeError):
            leaky_hidden_instance(input_, input_)

    def test_leaky_hidden_learn_graded_instance(
        self, leaky_hidden_learn_graded_instance
    ):
        factor = leaky_hidden_learn_graded_instance.graded_spikes_factor

        assert factor.requires_grad

    def test_leaky_compile_fullgraph(self, leaky_instance_surrogate, input_):
        explanation = dynamo.explain(leaky_instance_surrogate)(input_[0])

        assert explanation.graph_break_count == 0

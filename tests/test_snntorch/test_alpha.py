#!/usr/bin/env python

"""Tests for Alpha neuron."""

import pytest
import snntorch as snn
import torch
import torch._dynamo as dynamo


@pytest.fixture(scope="module")
def input_():
    return torch.Tensor([0.25, 0]).unsqueeze(-1)


@pytest.fixture(scope="module")
def alpha_instance():
    return snn.Alpha(alpha=0.6, beta=0.5, reset_mechanism="subtract")

@pytest.fixture(scope="module")
def alpha_instance_surrogate():
    return snn.Alpha(alpha=0.6, beta=0.5, reset_mechanism="subtract", surrogate_disable=True)


@pytest.fixture(scope="module")
def alpha_reset_zero_instance():
    return snn.Alpha(alpha=0.6, beta=0.5, reset_mechanism="zero")


@pytest.fixture(scope="module")
def alpha_reset_none_instance():
    return snn.Alpha(alpha=0.6, beta=0.5, reset_mechanism="none")


@pytest.fixture(scope="module")
def alpha_hidden_instance():
    return snn.Alpha(alpha=0.6, beta=0.5, init_hidden=True)


@pytest.fixture(scope="module")
def alpha_hidden_reset_zero_instance():
    return snn.Alpha(
        alpha=0.6, beta=0.5, init_hidden=True, reset_mechanism="zero"
    )


@pytest.fixture(scope="module")
def alpha_hidden_reset_none_instance():
    return snn.Alpha(
        alpha=0.6, beta=0.5, init_hidden=True, reset_mechanism="none"
    )


class TestAlpha:

    with pytest.raises(ValueError):
        snn.Alpha(0.5, 0.5)

    def test_alpha(self, alpha_instance, input_):
        syn_exc, syn_inh, mem = alpha_instance.init_alpha()

        syn_exc_rec = []
        syn_inh_rec = []
        mem_rec = []
        spk_rec = []

        for i in range(2):
            spk, syn_exc, syn_inh, mem = alpha_instance(
                input_[i], syn_exc, syn_inh, mem
            )

            syn_exc_rec.append(syn_exc)
            syn_inh_rec.append(syn_inh)
            mem_rec.append(mem)
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]
        print(spk_rec)
        print(syn_exc_rec)
        print(syn_inh_rec)
        assert syn_exc_rec[0] + syn_inh_rec[0] == 0
        assert syn_exc_rec[1] + syn_inh_rec[1] > 0
        assert mem_rec[0] < mem_rec[1]

    def test_alpha_reset(
        self,
        alpha_instance,
        alpha_reset_zero_instance,
        alpha_reset_none_instance,
    ):
        lif1 = alpha_instance
        lif2 = alpha_reset_zero_instance
        lif3 = alpha_reset_none_instance

        assert lif1.reset_mechanism_val == 0
        assert lif2.reset_mechanism_val == 1
        assert lif3.reset_mechanism_val == 2

        lif1.reset_mechanism = "zero"
        lif2.reset_mechanism = "none"
        lif3.reset_mechanism = "subtract"

        assert lif1.reset_mechanism_val == 1
        assert lif2.reset_mechanism_val == 2
        assert lif3.reset_mechanism_val == 0

    def test_alpha_init_hidden(self, alpha_hidden_instance, input_):

        spk_rec = []

        for i in range(2):
            spk = alpha_hidden_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_alpha_init_hidden_reset_zero(
        self, alpha_hidden_reset_zero_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = alpha_hidden_reset_zero_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_alpha_init_hidden_reset_none(
        self, alpha_hidden_reset_none_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = alpha_hidden_reset_none_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_alpha_cases(self, alpha_hidden_instance, input_):
        with pytest.raises(TypeError):
            alpha_hidden_instance(input_, input_)


    def test_alpha_compile_fullgraph(self, alpha_instance_surrogate, input_):
        explanation = dynamo.explain(alpha_instance_surrogate)(input_[0])

        assert explanation.graph_break_count == 0
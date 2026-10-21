#!/usr/bin/env python

"""Tests for Leaky neuron."""

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


@pytest.fixture(scope="function")
def leaky_instance():
    return snn.Leaky(beta=0.5)


@pytest.fixture(scope="module")
def leaky_instance_surrogate():
    return snn.Leaky(beta=0.5, surrogate_disable=True)


@pytest.fixture(scope="module")
def leaky_hidden_instance():
    return snn.Leaky(beta=0.5, init_hidden=True)


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


    @pytest.mark.parametrize(
        ("mechanism", "reset_delay", "expected_mem"),
        [
            ("subtract", True, [1.2, 0.1]),
            ("zero", True, [1.2, 0.0]),
            ("none", True, [1.2, 0.6]),
            ("subtract", False, [0.2, 0.1]),
            ("zero", False, [0.0, 0.0]),
            ("none", False, [1.2, 0.6]),
        ],
    )
    def test_leaky_reset(
        self, mechanism, reset_delay, expected_mem, spiking_input_
    ):
        neuron = snn.Leaky(
            beta=0.5,
            reset_mechanism=mechanism,
            reset_delay=reset_delay,
        )
        mem = torch.zeros(1)

        for i, expected in enumerate(expected_mem):
            spk, mem = neuron(spiking_input_[i], mem)

            assert spk.item() == pytest.approx(1 if i == 0 else 0)
            assert mem.item() == pytest.approx(expected)

    def test_leaky_reset_setter(self, leaky_instance):
        for val, mechanism in enumerate(RESET_MECHANISMS):
            leaky_instance.reset_mechanism = mechanism
            assert leaky_instance.reset_mechanism_val == val

    @pytest.mark.parametrize("mechanism", RESET_MECHANISMS)
    def test_leaky_init_hidden(self, mechanism, spiking_input_):
        hidden_instance = snn.Leaky(
            beta=0.5, 
            init_hidden=True, 
            reset_mechanism=mechanism
        )

        spk_rec = []

        for i in range(2):
            spk = hidden_instance(spiking_input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] != spk_rec[1]

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

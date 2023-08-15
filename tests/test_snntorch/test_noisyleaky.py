#!/usr/bin/env python

"""Tests for Noisy Leaky neuron."""

import pytest
import snntorch as snn
import torch


@pytest.fixture(scope="module")
def input_():
    return torch.Tensor([0.25, 0]).unsqueeze(-1)


@pytest.fixture(scope="module")
def noisyleaky_instance():
    return snn.NoisyLeaky(beta=0.5)


@pytest.fixture(scope="module")
def noisyleaky_reset_zero_instance():
    return snn.NoisyLeaky(beta=0.5, reset_mechanism="zero")


@pytest.fixture(scope="module")
def noisyleaky_reset_none_instance():
    return snn.NoisyLeaky(beta=0.5, reset_mechanism="none")


@pytest.fixture(scope="module")
def noisyleaky_hidden_instance():
    return snn.NoisyLeaky(beta=0.5, init_hidden=True)


@pytest.fixture(scope="module")
def noisyleaky_hidden_reset_zero_instance():
    return snn.NoisyLeaky(beta=0.5, init_hidden=True, reset_mechanism="zero")


@pytest.fixture(scope="module")
def noisyleaky_hidden_reset_none_instance():
    return snn.NoisyLeaky(beta=0.5, init_hidden=True, reset_mechanism="none")


@pytest.fixture(scope="module")
def noisyleaky_hidden_learn_graded_instance():
    return snn.NoisyLeaky(
        beta=0.5, init_hidden=True, learn_graded_spikes_factor=True
    )


class TestNoisyLeaky:
    def test_noisyleaky(self, noisyleaky_instance, input_):
        mem = noisyleaky_instance.init_noisyleaky()

        mem_rec = []
        spk_rec = []

        for i in range(2):

            spk, mem = noisyleaky_instance(input_[i], mem)
            mem_rec.append(mem)
            spk_rec.append(spk)

        assert mem_rec[1] == mem_rec[0] * 0.5 + input_[1]
        assert spk_rec[0] == spk_rec[1]

    def test_noisyleaky_reset(
        self,
        noisyleaky_instance,
        noisyleaky_reset_zero_instance,
        noisyleaky_reset_none_instance,
    ):
        lif1 = noisyleaky_instance
        lif2 = noisyleaky_reset_zero_instance
        lif3 = noisyleaky_reset_none_instance

        assert lif1.reset_mechanism_val == 0
        assert lif2.reset_mechanism_val == 1
        assert lif3.reset_mechanism_val == 2

        lif1.reset_mechanism = "zero"
        lif2.reset_mechanism = "none"
        lif3.reset_mechanism = "subtract"

        assert lif1.reset_mechanism_val == 1
        assert lif2.reset_mechanism_val == 2
        assert lif3.reset_mechanism_val == 0

    def test_noisyleaky_init_hidden(
        self, noisyleaky_hidden_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = noisyleaky_hidden_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_noisyleaky_init_hidden_reset_zero(
        self, noisyleaky_hidden_reset_zero_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = noisyleaky_hidden_reset_zero_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_noisyleaky_init_hidden_reset_none(
        self, noisyleaky_hidden_reset_none_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = noisyleaky_hidden_reset_none_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_noisyleaky_cases(self, noisyleaky_hidden_instance, input_):
        with pytest.raises(TypeError):
            noisyleaky_hidden_instance(input_, input_)

    def test_noisyleaky_hidden_learn_graded_instance(
            self, noisyleaky_hidden_learn_graded_instance
    ):
        factor = noisyleaky_hidden_learn_graded_instance.graded_spikes_factor

        assert factor.requires_grad

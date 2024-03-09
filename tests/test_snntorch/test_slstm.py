#!/usr/bin/env python

"""Tests for SLSTM neuron."""

import pytest
import snntorch as snn
import torch
import torch._dynamo as dynamo

# TO-DO: add avg/max-pooling tests


@pytest.fixture(scope="module")
def input_():
    # 2 steps, 1 batch
    return torch.rand(2, 1, 1)


@pytest.fixture(scope="module")
def slstm_instance():
    return snn.SLSTM(1, 2)


@pytest.fixture(scope="module")
def slstm_instance_surrogate():
    return snn.SLSTM(1, 2, surrogate_disable=True)


@pytest.fixture(scope="module")
def slstm_reset_zero_instance():
    return snn.SLSTM(1, 2, reset_mechanism="zero")


@pytest.fixture(scope="module")
def slstm_reset_subtract_instance():
    return snn.SLSTM(1, 2, reset_mechanism="subtract")


@pytest.fixture(scope="module")
def slstm_hidden_instance():
    return snn.SLSTM(1, 2, init_hidden=True)


@pytest.fixture(scope="module")
def slstm_hidden_reset_zero_instance():
    return snn.SLSTM(1, 2, init_hidden=True, reset_mechanism="zero")


@pytest.fixture(scope="module")
def slstm_hidden_reset_subtract_instance():
    return snn.SLSTM(1, 2, init_hidden=True, reset_mechanism="subtract")


class TestSLSTM:
    def test_slstm(self, slstm_instance, input_):
        c, h = slstm_instance.init_slstm()

        h_rec = []
        c_rec = []
        spk_rec = []

        for i in range(2):
            spk, c, h = slstm_instance(input_[i], c, h)

            c_rec.append(c)
            h_rec.append(h)
            spk_rec.append(spk)

        assert c.size() == (1, 2)
        assert h.size() == (1, 2)
        assert spk.size() == (1, 2)

    def test_slstm_reset(
        self,
        slstm_instance,
        slstm_reset_zero_instance,
        slstm_reset_subtract_instance,
    ):

        lif1 = slstm_reset_subtract_instance
        lif2 = slstm_reset_zero_instance
        lif3 = slstm_instance

        assert lif1.reset_mechanism_val == 0
        assert lif2.reset_mechanism_val == 1
        assert lif3.reset_mechanism_val == 2

        lif1.reset_mechanism = "none"
        lif2.reset_mechanism = "zero"
        lif3.reset_mechanism = "subtract"

        assert lif1.reset_mechanism_val == 2
        assert lif2.reset_mechanism_val == 1
        assert lif3.reset_mechanism_val == 0

    def test_slstm_init_hidden(self, slstm_hidden_instance, input_):

        spk_rec = []

        for i in range(2):
            spk = slstm_hidden_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0].size() == (1, 2)

    def test_slstm_init_hidden_reset_zero(
        self, slstm_hidden_reset_zero_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = slstm_hidden_reset_zero_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0].size() == (1, 2)

    def test_slstm_init_hidden_reset_subtract(
        self, slstm_hidden_reset_subtract_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = slstm_hidden_reset_subtract_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0].size() == (1, 2)

    def test_slstm_compile_fullgraph(
        self, slstm_instance_surrogate, input_
    ):
        explanation = dynamo.explain(slstm_instance_surrogate)(input_[0])

        assert explanation.graph_break_count == 0

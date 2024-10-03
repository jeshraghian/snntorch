#!/usr/bin/env python

"""Tests for SConv2dLSTM neuron."""

import pytest
import snntorch as snn
import torch
import torch._dynamo as dynamo

# TO-DO: add avg/max-pooling tests


@pytest.fixture(scope="module")
def input_():
    # 2 steps, 1 batch, 1 channel, 4-h, 4-w
    return torch.rand(2, 1, 1, 4, 4)


@pytest.fixture(scope="module")
def sconv2dlstm_instance():
    return snn.SConv2dLSTM(1, 8, 3)


@pytest.fixture(scope="module")
def sconv2dlstm_instance_surrogate():
    return snn.SConv2dLSTM(1, 8, 3, surrogate_disable=True)


@pytest.fixture(scope="module")
def sconv2dlstm_reset_zero_instance():
    return snn.SConv2dLSTM(1, 8, 3, reset_mechanism="zero")


@pytest.fixture(scope="module")
def sconv2dlstm_reset_subtract_instance():
    return snn.SConv2dLSTM(1, 8, 3, reset_mechanism="subtract")


@pytest.fixture(scope="module")
def sconv2dlstm_hidden_instance():
    return snn.SConv2dLSTM(1, 8, 3, init_hidden=True)


@pytest.fixture(scope="module")
def sconv2dlstm_hidden_reset_zero_instance():
    return snn.SConv2dLSTM(1, 8, 3, init_hidden=True, reset_mechanism="zero")


@pytest.fixture(scope="module")
def sconv2dlstm_hidden_reset_subtract_instance():
    return snn.SConv2dLSTM(
        1, 8, 3, init_hidden=True, reset_mechanism="subtract"
    )


class TestSConv2dLSTM:
    def test_sconv2dlstm(self, sconv2dlstm_instance, input_):
        c, h = sconv2dlstm_instance.init_sconv2dlstm()

        h_rec = []
        c_rec = []
        spk_rec = []

        for i in range(2):
            spk, c, h = sconv2dlstm_instance(input_[i], c, h)

            c_rec.append(c)
            h_rec.append(h)
            spk_rec.append(spk)

        assert c.size() == (1, 8, 4, 4)
        assert h.size() == (1, 8, 4, 4)
        assert spk.size() == (1, 8, 4, 4)

    def test_sconv2dlstm_reset(
        self,
        sconv2dlstm_instance,
        sconv2dlstm_reset_zero_instance,
        sconv2dlstm_reset_subtract_instance,
    ):

        lif1 = sconv2dlstm_reset_subtract_instance
        lif2 = sconv2dlstm_reset_zero_instance
        lif3 = sconv2dlstm_instance

        assert lif1.reset_mechanism_val == 0
        assert lif2.reset_mechanism_val == 1
        assert lif3.reset_mechanism_val == 2

        lif1.reset_mechanism = "none"
        lif2.reset_mechanism = "zero"
        lif3.reset_mechanism = "subtract"

        assert lif1.reset_mechanism_val == 2
        assert lif2.reset_mechanism_val == 1
        assert lif3.reset_mechanism_val == 0

    def test_sconv2dlstm_init_hidden(
        self, sconv2dlstm_hidden_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = sconv2dlstm_hidden_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0].size() == (1, 8, 4, 4)

    def test_sconv2dlstm_init_hidden_reset_zero(
        self, sconv2dlstm_hidden_reset_zero_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = sconv2dlstm_hidden_reset_zero_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0].size() == (1, 8, 4, 4)

    def test_sconv2dlstm_init_hidden_reset_subtract(
        self, sconv2dlstm_hidden_reset_subtract_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = sconv2dlstm_hidden_reset_subtract_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0].size() == (1, 8, 4, 4)

    def test_sconv2dlstm_compile_fullgraph(
        self, sconv2dlstm_instance_surrogate, input_
    ):
        explanation = dynamo.explain(sconv2dlstm_instance_surrogate)(input_[0])

        assert explanation.graph_break_count == 0

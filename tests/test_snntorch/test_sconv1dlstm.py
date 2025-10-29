#!/usr/bin/env python

"""Tests for SConv1dLSTM neuron."""

import pytest
import snntorch as snn
import torch
import torch._dynamo as dynamo


@pytest.fixture(scope="module")
def input_():
    # 2 steps, 1 batch, 1 channel, 8 length
    return torch.rand(2, 1, 1, 8)


@pytest.fixture(scope="module")
def sconv1dlstm_instance():
    return snn.SConv1dLSTM(1, 8, 3)


@pytest.fixture(scope="module")
def sconv1dlstm_instance_surrogate():
    return snn.SConv1dLSTM(1, 8, 3, surrogate_disable=True)


@pytest.fixture(scope="module")
def sconv1dlstm_reset_zero_instance():
    return snn.SConv1dLSTM(1, 8, 3, reset_mechanism="zero")


@pytest.fixture(scope="module")
def sconv1dlstm_reset_subtract_instance():
    return snn.SConv1dLSTM(1, 8, 3, reset_mechanism="subtract")


@pytest.fixture(scope="module")
def sconv1dlstm_hidden_instance():
    return snn.SConv1dLSTM(1, 8, 3, init_hidden=True)


@pytest.fixture(scope="module")
def sconv1dlstm_hidden_reset_zero_instance():
    return snn.SConv1dLSTM(1, 8, 3, init_hidden=True, reset_mechanism="zero")


@pytest.fixture(scope="module")
def sconv1dlstm_hidden_reset_subtract_instance():
    return snn.SConv1dLSTM(
        1, 8, 3, init_hidden=True, reset_mechanism="subtract"
    )


class TestSConv1dLSTM:
    def test_sconv1dlstm(self, sconv1dlstm_instance, input_):
        c, h = sconv1dlstm_instance.init_sconv1dlstm()

        h_rec = []
        c_rec = []
        spk_rec = []

        for i in range(2):
            spk, c, h = sconv1dlstm_instance(input_[i], c, h)

            c_rec.append(c)
            h_rec.append(h)
            spk_rec.append(spk)

        assert c.size() == (1, 8, 8)
        assert h.size() == (1, 8, 8)
        assert spk.size() == (1, 8, 8)

    def test_sconv1dlstm_reset(
        self,
        sconv1dlstm_instance,
        sconv1dlstm_reset_zero_instance,
        sconv1dlstm_reset_subtract_instance,
    ):

        lif1 = sconv1dlstm_reset_subtract_instance
        lif2 = sconv1dlstm_reset_zero_instance
        lif3 = sconv1dlstm_instance

        assert lif1.reset_mechanism_val == 0
        assert lif2.reset_mechanism_val == 1
        assert lif3.reset_mechanism_val == 2

        lif1.reset_mechanism = "none"
        lif2.reset_mechanism = "zero"
        lif3.reset_mechanism = "subtract"

        assert lif1.reset_mechanism_val == 2
        assert lif2.reset_mechanism_val == 1
        assert lif3.reset_mechanism_val == 0

    def test_sconv1dlstm_init_hidden(
        self, sconv1dlstm_hidden_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = sconv1dlstm_hidden_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0].size() == (1, 8, 8)

    def test_sconv1dlstm_init_hidden_reset_zero(
        self, sconv1dlstm_hidden_reset_zero_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = sconv1dlstm_hidden_reset_zero_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0].size() == (1, 8, 8)

    def test_sconv1dlstm_init_hidden_reset_subtract(
        self, sconv1dlstm_hidden_reset_subtract_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = sconv1dlstm_hidden_reset_subtract_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0].size() == (1, 8, 8)

    def test_sconv1dlstm_compile_fullgraph(
        self, sconv1dlstm_instance_surrogate, input_
    ):
        explanation = dynamo.explain(sconv1dlstm_instance_surrogate)(input_[0])

        assert explanation.graph_break_count == 0

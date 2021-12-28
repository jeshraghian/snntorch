#!/usr/bin/env python

"""Tests for SConvLSTM neuron."""

import pytest
import snntorch as snn
import torch

# TO-DO: add avg/max-pooling tests


@pytest.fixture(scope="module")
def input_():
    # 2 steps, 1 batch, 1 channel, 4-h, 4-w
    return torch.rand(2, 1, 1, 4, 4)


@pytest.fixture(scope="module")
def sconvlstm_instance():
    return snn.SConvLSTM(1, 8, 3)


@pytest.fixture(scope="module")
def sconvlstm_reset_zero_instance():
    return snn.SConvLSTM(1, 8, 3, reset_mechanism="zero")


@pytest.fixture(scope="module")
def sconvlstm_reset_subtract_instance():
    return snn.SConvLSTM(1, 8, 3, reset_mechanism="subtract")


@pytest.fixture(scope="module")
def sconvlstm_hidden_instance():
    return snn.SConvLSTM(1, 8, 3, init_hidden=True)


@pytest.fixture(scope="module")
def sconvlstm_hidden_reset_zero_instance():
    return snn.SConvLSTM(1, 8, 3, init_hidden=True, reset_mechanism="zero")


@pytest.fixture(scope="module")
def sconvlstm_hidden_reset_subtract_instance():
    return snn.SConvLSTM(1, 8, 3, init_hidden=True, reset_mechanism="subtract")


class TestSConvLSTM:
    def test_sconvlstm(self, sconvlstm_instance, input_):
        c, h = sconvlstm_instance.init_sconvlstm()

        h_rec = []
        c_rec = []
        spk_rec = []

        for i in range(2):
            spk, c, h = sconvlstm_instance(input_[i], c, h)

            c_rec.append(c)
            h_rec.append(h)
            spk_rec.append(spk)

        assert c.size() == (1, 8, 4, 4)
        assert h.size() == (1, 8, 4, 4)
        assert spk.size() == (1, 8, 4, 4)

    def test_sconvlstm_reset(
        self,
        sconvlstm_instance,
        sconvlstm_reset_zero_instance,
        sconvlstm_reset_subtract_instance,
    ):

        lif1 = sconvlstm_reset_subtract_instance
        lif2 = sconvlstm_reset_zero_instance
        lif3 = sconvlstm_instance

        assert lif1.reset_mechanism_val == 0
        assert lif2.reset_mechanism_val == 1
        assert lif3.reset_mechanism_val == 2

        lif1.reset_mechanism = "none"
        lif2.reset_mechanism = "zero"
        lif3.reset_mechanism = "subtract"

        assert lif1.reset_mechanism_val == 2
        assert lif2.reset_mechanism_val == 1
        assert lif3.reset_mechanism_val == 0

    def test_sconvlstm_init_hidden(self, sconvlstm_hidden_instance, input_):

        spk_rec = []

        for i in range(2):
            spk = sconvlstm_hidden_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0].size() == (1, 8, 4, 4)

    def test_sconvlstm_init_hidden_reset_zero(
        self, sconvlstm_hidden_reset_zero_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = sconvlstm_hidden_reset_zero_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0].size() == (1, 8, 4, 4)

    def test_sconvlstm_init_hidden_reset_subtract(
        self, sconvlstm_hidden_reset_subtract_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = sconvlstm_hidden_reset_subtract_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0].size() == (1, 8, 4, 4)

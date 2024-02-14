#!/usr/bin/env python

"""Tests for Lapicque neuron."""

import pytest
import snntorch as snn
import torch
import torch._dynamo as dynamo


@pytest.fixture(scope="module")
def input_():
    return torch.Tensor([0.25, 0]).unsqueeze(-1)


@pytest.fixture(scope="module")
def lapicque_instance():
    return snn.Lapicque(beta=0.5)


@pytest.fixture(scope="module")
def lapicque_instance_surrogate():
    return snn.Lapicque(beta=0.5, surrogate_disable=True)


@pytest.fixture(scope="module")
def lapicque_reset_zero_instance():
    return snn.Lapicque(beta=0.5, reset_mechanism="zero")


@pytest.fixture(scope="module")
def lapicque_reset_none_instance():
    return snn.Lapicque(beta=0.5, reset_mechanism="none")


@pytest.fixture(scope="module")
def lapicque_hidden_instance():
    return snn.Lapicque(beta=0.5, init_hidden=True)


@pytest.fixture(scope="module")
def lapicque_hidden_reset_zero_instance():
    return snn.Lapicque(beta=0.5, init_hidden=True, reset_mechanism="zero")


@pytest.fixture(scope="module")
def lapicque_hidden_reset_none_instance():
    return snn.Lapicque(beta=0.5, init_hidden=True, reset_mechanism="none")


class TestLapicque:
    def test_lapicque(self, lapicque_instance, input_):
        mem = lapicque_instance.init_lapicque()
        # assert len(mem) == 1

        mem_rec = []
        spk_rec = []

        for i in range(2):
            spk, mem = lapicque_instance(input_[i], mem)
            mem_rec.append(mem)
            spk_rec.append(spk)

        assert mem_rec[1] == mem_rec[0] * (
            1
            - (
                lapicque_instance.time_step
                / (lapicque_instance.R * lapicque_instance.C)
            )
        ) + input_[1] * lapicque_instance.R * (
            1
            / lapicque_instance.R
            * lapicque_instance.C
            * lapicque_instance.time_step
        )
        assert spk_rec[0] == spk_rec[1]

    def test_lapicque_reset(
        self,
        lapicque_instance,
        lapicque_reset_zero_instance,
        lapicque_reset_none_instance,
    ):
        lif1 = lapicque_instance
        lif2 = lapicque_reset_zero_instance
        lif3 = lapicque_reset_none_instance

        assert lif1.reset_mechanism_val == 0
        assert lif2.reset_mechanism_val == 1
        assert lif3.reset_mechanism_val == 2

        lif1.reset_mechanism = "zero"
        lif2.reset_mechanism = "none"
        lif3.reset_mechanism = "subtract"

        assert lif1.reset_mechanism_val == 1
        assert lif2.reset_mechanism_val == 2
        assert lif3.reset_mechanism_val == 0

    def test_lapicque_init_hidden(self, lapicque_hidden_instance, input_):

        spk_rec = []

        for i in range(2):
            spk = lapicque_hidden_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_lapicque_init_hidden_reset_zero(
        self, lapicque_hidden_reset_zero_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = lapicque_hidden_reset_zero_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_lapicque_init_hidden_reset_none(
        self, lapicque_hidden_reset_none_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = lapicque_hidden_reset_none_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_lapicque_cases(self, lapicque_hidden_instance, input_):
        with pytest.raises(TypeError):
            lapicque_hidden_instance(input_, input_)

    def test_lapicque_compile_fullgraph(self, lapicque_instance_surrogate, input_):
        explanation = dynamo.explain(lapicque_instance_surrogate)(input_[0])

        assert explanation.graph_break_count == 0

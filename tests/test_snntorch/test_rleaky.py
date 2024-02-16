#!/usr/bin/env python

"""Tests for RLeaky neuron."""

import pytest
import snntorch as snn
import torch
import torch._dynamo as dynamo


@pytest.fixture(scope="module")
def input_():
    return torch.Tensor([0.25, 0]).unsqueeze(-1)


@pytest.fixture(scope="module")
def rleaky_instance():
    return snn.RLeaky(beta=0.5, V=0.5, all_to_all=False)


@pytest.fixture(scope="module")
def rleaky_instance_surrogate():
    return snn.RLeaky(
        beta=0.5, V=0.5, all_to_all=False, surrogate_disable=True
    )


@pytest.fixture(scope="module")
def rleaky_reset_zero_instance():
    return snn.RLeaky(
        beta=0.5, V=0.5, all_to_all=False, reset_mechanism="zero"
    )


@pytest.fixture(scope="module")
def rleaky_reset_none_instance():
    return snn.RLeaky(
        beta=0.5, V=0.5, all_to_all=False, reset_mechanism="none"
    )


@pytest.fixture(scope="module")
def rleaky_hidden_instance():
    return snn.RLeaky(beta=0.5, V=0.5, all_to_all=False, init_hidden=True)


@pytest.fixture(scope="module")
def rleaky_hidden_reset_zero_instance():
    return snn.RLeaky(
        beta=0.5,
        V=0.5,
        all_to_all=False,
        init_hidden=True,
        reset_mechanism="zero",
    )


@pytest.fixture(scope="module")
def rleaky_hidden_reset_none_instance():
    return snn.RLeaky(
        beta=0.5,
        V=0.5,
        all_to_all=False,
        init_hidden=True,
        reset_mechanism="none",
    )


class TestRLeaky:
    def test_rleaky(self, rleaky_instance, input_):
        spk, mem = rleaky_instance.init_rleaky()

        mem_rec = []
        spk_rec = []

        for i in range(2):

            spk, mem = rleaky_instance(input_[i], spk, mem)
            mem_rec.append(mem)
            spk_rec.append(spk)

        assert mem_rec[1] == mem_rec[0] * 0.5 + input_[1] + spk_rec[0]
        assert spk_rec[0] == spk_rec[1]

    def test_rleaky_reset(
        self,
        rleaky_instance,
        rleaky_reset_zero_instance,
        rleaky_reset_none_instance,
    ):
        lif1 = rleaky_instance
        lif2 = rleaky_reset_zero_instance
        lif3 = rleaky_reset_none_instance

        assert lif1.reset_mechanism_val == 0
        assert lif2.reset_mechanism_val == 1
        assert lif3.reset_mechanism_val == 2

        lif1.reset_mechanism = "zero"
        lif2.reset_mechanism = "none"
        lif3.reset_mechanism = "subtract"

        assert lif1.reset_mechanism_val == 1
        assert lif2.reset_mechanism_val == 2
        assert lif3.reset_mechanism_val == 0

    def test_rleaky_init_hidden(self, rleaky_hidden_instance, input_):

        spk_rec = []

        for i in range(2):
            spk = rleaky_hidden_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_rleaky_init_hidden_reset_zero(
        self, rleaky_hidden_reset_zero_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = rleaky_hidden_reset_zero_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_rleaky_init_hidden_reset_none(
        self, rleaky_hidden_reset_none_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = rleaky_hidden_reset_none_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_lreaky_cases(self, rleaky_hidden_instance, input_):
        with pytest.raises(TypeError):
            rleaky_hidden_instance(input_, input_, input_)

    def test_rleaky_compile_fullgraph(
        self, rleaky_instance_surrogate, input_
    ):
        explanation = dynamo.explain(rleaky_instance_surrogate)(input_[0])

        assert explanation.graph_break_count == 0

#!/usr/bin/env python

"""Tests for RLeaky neuron."""

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
def rleaky_instance():
    return snn.RLeaky(beta=0.5, V=0.5, all_to_all=False)


@pytest.fixture(scope="module")
def rleaky_instance_surrogate():
    return snn.RLeaky(
        beta=0.5, V=0.5, all_to_all=False, surrogate_disable=True
    )


@pytest.fixture(scope="module")
def rleaky_hidden_instance():
    return snn.RLeaky(beta=0.5, V=0.5, all_to_all=False, init_hidden=True)


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

    @pytest.mark.parametrize(
        ("mechanism", "reset_delay", "expected_mem"),
        [
            ("subtract", True, [1.2, 0.2]),
            ("zero", True, [1.2, 0.1]),
            ("none", True, [1.2, 0.7]),
            ("subtract", False, [0.2, 0.2]),
            ("zero", False, [0.0, 0.1]),
            ("none", False, [1.2, 0.7]),
        ],
    )
    def test_rleaky_reset(
        self, mechanism, reset_delay, expected_mem, spiking_input_
    ):
        neuron = snn.RLeaky(
            beta=0.5,
            V=0.1,
            all_to_all=False,
            reset_mechanism=mechanism,
            reset_delay=reset_delay,
        )
        spk = torch.zeros(1)
        mem = torch.zeros(1)

        for i, expected in enumerate(expected_mem):
            spk, mem = neuron(spiking_input_[i], spk, mem)

            assert spk.item() == pytest.approx(1 if i == 0 else 0)
            assert mem.item() == pytest.approx(expected)

    def test_rleaky_reset_setter(self, rleaky_instance):
        for val, mechanism in enumerate(RESET_MECHANISMS):
            rleaky_instance.reset_mechanism = mechanism
            assert rleaky_instance.reset_mechanism_val == val

    @pytest.mark.parametrize("mechanism", RESET_MECHANISMS)
    def test_rleaky_init_hidden(self, mechanism, spiking_input_):
        hidden_instance = snn.RLeaky(
            beta=0.5,
            V=0.1,
            all_to_all=False,
            init_hidden=True,
            reset_mechanism=mechanism,
        )
        spk_rec = []

        for i in range(2):
            spk = hidden_instance(spiking_input_[i])
            spk_rec.append(spk)

        assert spk_rec[0].item() == pytest.approx(1)
        assert spk_rec[1].item() == pytest.approx(0)

    def test_lreaky_cases(self, rleaky_hidden_instance, input_):
        with pytest.raises(TypeError):
            rleaky_hidden_instance(input_, input_, input_)

    def test_rleaky_compile_fullgraph(self, rleaky_instance_surrogate, input_):
        explanation = dynamo.explain(rleaky_instance_surrogate)(input_[0])

        assert explanation.graph_break_count == 0

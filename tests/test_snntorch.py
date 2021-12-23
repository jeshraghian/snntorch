#!/usr/bin/env python

"""Tests for `snntorch` package."""

import pytest
import snntorch as snn
import torch


@pytest.fixture(scope="module")
def leaky_instance():
    return snn.Leaky(beta=0.5)


@pytest.fixture(scope="module")
def leaky_reset_zero_instance():
    return snn.Leaky(beta=0.5, reset_mechanism="zero")


@pytest.fixture(scope="module")
def leaky_reset_none_instance():
    return snn.Leaky(beta=0.5, reset_mechanism="none")


@pytest.fixture(scope="module")
def leaky_hidden_instance():
    return snn.Leaky(beta=0.5, init_hidden=True)


@pytest.fixture(scope="module")
def leaky_hidden_reset_zero_instance():
    return snn.Leaky(beta=0.5, init_hidden=True, reset_mechanism="zero")


@pytest.fixture(scope="module")
def leaky_hidden_reset_none_instance():
    return snn.Leaky(beta=0.5, init_hidden=True, reset_mechanism="none")


@pytest.fixture(scope="module")
def lapicque_instance():
    return snn.Lapicque(beta=0.5)


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


@pytest.fixture(scope="module")
def synaptic_instance():
    return snn.Synaptic(alpha=0.5, beta=0.5)


@pytest.fixture(scope="module")
def synaptic_reset_zero_instance():
    return snn.Synaptic(alpha=0.5, beta=0.5, reset_mechanism="zero")


@pytest.fixture(scope="module")
def synaptic_reset_none_instance():
    return snn.Synaptic(alpha=0.5, beta=0.5, reset_mechanism="none")


@pytest.fixture(scope="module")
def synaptic_hidden_instance():
    return snn.Synaptic(alpha=0.5, beta=0.5, init_hidden=True)


@pytest.fixture(scope="module")
def synaptic_hidden_reset_zero_instance():
    return snn.Synaptic(alpha=0.5, beta=0.5, init_hidden=True, reset_mechanism="zero")


@pytest.fixture(scope="module")
def synaptic_hidden_reset_none_instance():
    return snn.Synaptic(alpha=0.5, beta=0.5, init_hidden=True, reset_mechanism="none")


@pytest.fixture(scope="module")
def alpha_instance():
    return snn.Alpha(alpha=0.6, beta=0.5)


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
    return snn.Alpha(alpha=0.6, beta=0.5, init_hidden=True, reset_mechanism="zero")


@pytest.fixture(scope="module")
def alpha_hidden_reset_none_instance():
    return snn.Alpha(alpha=0.6, beta=0.5, init_hidden=True, reset_mechanism="none")


@pytest.fixture(scope="module")
def input_():
    return torch.Tensor([0.25, 0]).unsqueeze(-1)


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

    def test_leaky_reset(
        self, leaky_instance, leaky_reset_zero_instance, leaky_reset_none_instance
    ):
        lif1 = leaky_instance
        lif2 = leaky_reset_zero_instance
        lif3 = leaky_reset_none_instance

        assert lif1.reset_mechanism_val == 0
        assert lif2.reset_mechanism_val == 1
        assert lif3.reset_mechanism_val == 2

        lif1.reset_mechanism = "zero"
        lif2.reset_mechanism = "none"
        lif3.reset_mechanism = "subtract"

        assert lif1.reset_mechanism_val == 1
        assert lif2.reset_mechanism_val == 2
        assert lif3.reset_mechanism_val == 0

    def test_leaky_init_hidden(self, leaky_hidden_instance, input_):

        spk_rec = []

        for i in range(2):
            spk = leaky_hidden_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_leaky_init_hidden_reset_zero(
        self, leaky_hidden_reset_zero_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = leaky_hidden_reset_zero_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_leaky_init_hidden_reset_none(
        self, leaky_hidden_reset_none_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = leaky_hidden_reset_none_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_leaky_cases(self, leaky_hidden_instance, input_):
        with pytest.raises(TypeError):
            leaky_hidden_instance(input_, input_)


class TestSynaptic:
    def test_synaptic(self, synaptic_instance, input_):
        syn, mem = synaptic_instance.init_synaptic()

        syn_rec = []
        mem_rec = []
        spk_rec = []

        for i in range(2):
            spk, syn, mem = synaptic_instance(input_[i], syn, mem)
            syn_rec.append(syn)
            mem_rec.append(mem)
            spk_rec.append(spk)

        assert syn_rec[0] == 2 * syn_rec[1]
        assert mem_rec[1] == mem_rec[0] * 0.5 + syn_rec[1]
        assert spk_rec[0] == spk_rec[1]

    def test_synaptic_reset(
        self,
        synaptic_instance,
        synaptic_reset_zero_instance,
        synaptic_reset_none_instance,
    ):
        lif1 = synaptic_instance
        lif2 = synaptic_reset_zero_instance
        lif3 = synaptic_reset_none_instance

        assert lif1.reset_mechanism_val == 0
        assert lif2.reset_mechanism_val == 1
        assert lif3.reset_mechanism_val == 2

        lif1.reset_mechanism = "zero"
        lif2.reset_mechanism = "none"
        lif3.reset_mechanism = "subtract"

        assert lif1.reset_mechanism_val == 1
        assert lif2.reset_mechanism_val == 2
        assert lif3.reset_mechanism_val == 0

    def test_synaptic_init_hidden(self, synaptic_hidden_instance, input_):

        spk_rec = []

        for i in range(2):
            spk = synaptic_hidden_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_synaptic_init_hidden_reset_zero(
        self, synaptic_hidden_reset_zero_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = synaptic_hidden_reset_zero_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_synaptic_init_hidden_reset_none(
        self, synaptic_hidden_reset_none_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = synaptic_hidden_reset_none_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_synaptic_cases(self, synaptic_hidden_instance, input_):
        with pytest.raises(TypeError):
            synaptic_hidden_instance(input_, input_)


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
            1 / lapicque_instance.R * lapicque_instance.C * lapicque_instance.time_step
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
        self, alpha_instance, alpha_reset_zero_instance, alpha_reset_none_instance
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


def test_fire():
    synaptic = snn.Synaptic(alpha=0.5, beta=0.5)
    input_large = torch.Tensor([synaptic.threshold * 10])
    assert synaptic.fire(input_large)[0] == 1


def test_instances():
    snn.LIF.instances = []
    snn.Synaptic(alpha=0.5, beta=0.5)
    snn.Alpha(alpha=0.5, beta=0.4)
    assert len(snn.LIF.instances) == 2

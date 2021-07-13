#!/usr/bin/env python

"""Tests for `snntorch` package."""

import pytest

# from click.testing import CliRunner

import snntorch as snn
import torch

# from snntorch import cli


# @pytest.fixture
# def response():
#     """Sample pytest fixture.

#     See more at: http://doc.pytest.org/en/latest/fixture.html
#     """


# def test_command_line_interface():
#     """Test the CLI."""
#     runner = CliRunner()
#     result = runner.invoke(cli.main)
#     assert result.exit_code == 0
#     assert "snntorch.cli.main" in result.output
#     help_result = runner.invoke(cli.main, ["--help"])
#     assert help_result.exit_code == 0
#     assert "--help  Show this message and exit." in help_result.output


@pytest.fixture(scope="module")
def leaky_instance():
    return snn.Leaky(beta=0.5)


@pytest.fixture(scope="module")
def lapicque_instance():
    return snn.Lapicque(beta=0.5)


@pytest.fixture(scope="module")
def synaptic_instance():
    return snn.Synaptic(alpha=0.5, beta=0.5)


@pytest.fixture(scope="module")
def stein_instance():
    return snn.Stein(alpha=0.5, beta=0.5)


@pytest.fixture(scope="module")
def srm0_instance():
    return snn.SRM0(alpha=0.6, beta=0.5)


@pytest.fixture(scope="module")
def input_():
    return torch.Tensor([0.25, 0])


class TestLeaky:
    def test_leaky(self, leaky_instance, input_):
        spk, mem = leaky_instance.init_leaky(1)
        assert len(spk) == 1
        assert len(mem) == 1

        mem_rec = []
        spk_rec = []

        for i in range(2):
            spk, mem = leaky_instance(input_[i], mem)
            mem_rec.append(mem)
            spk_rec.append(spk)

        assert mem_rec[1] == mem_rec[0] * 0.5 + input_[1]
        assert spk_rec[0] == spk_rec[1]

    def test_leaky_hidden_init(self):

        with pytest.raises(ValueError):
            snn.Leaky(beta=0.5, hidden_init=True)
            snn.Leaky(beta=0.5, num_inputs=1, hidden_init=True)
            snn.Leaky(beta=0.5, batch_size=1, hidden_init=True)

        lif1 = snn.Leaky(beta=0.5, num_inputs=1, batch_size=1, hidden_init=True)

        assert lif1.spk == 0
        assert lif1.mem == 0


class TestSynaptic:
    def test_stein(self, synaptic_instance, input_):
        spk, syn, mem = synaptic_instance.init_synaptic(1)
        assert len(spk) == 1
        assert len(syn) == 1
        assert len(mem) == 1

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

    def test_synaptic_hidden_init(self):

        with pytest.raises(ValueError):
            snn.Synaptic(alpha=0.5, beta=0.5, hidden_init=True)
            snn.Synaptic(alpha=0.5, beta=0.5, num_inputs=1, hidden_init=True)
            snn.Synaptic(alpha=0.5, beta=0.5, batch_size=1, hidden_init=True)

        lif2 = snn.Synaptic(
            alpha=0.5, beta=0.5, num_inputs=1, batch_size=1, hidden_init=True
        )

        assert lif2.spk == 0
        assert lif2.syn == 0
        assert lif2.mem == 0


class TestLapicque:
    def test_lapicque(self, lapicque_instance, input_):
        spk, mem = lapicque_instance.init_lapicque(1)
        assert len(spk) == 1
        assert len(mem) == 1

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

    def test_lapicque_hidden_init(self):

        with pytest.raises(ValueError):
            snn.Lapicque(beta=0.5, hidden_init=True)
            snn.Lapicque(beta=0.5, num_inputs=1, hidden_init=True)
            snn.Lapicque(beta=0.5, batch_size=1, hidden_init=True)

        lapicque = snn.Lapicque(beta=0.5, num_inputs=1, batch_size=1, hidden_init=True)

        assert lapicque.spk == 0
        assert lapicque.mem == 0


class TestStein:
    def test_stein(self, stein_instance, input_):
        spk, syn, mem = stein_instance.init_stein(1)
        assert len(spk) == 1
        assert len(syn) == 1
        assert len(mem) == 1

        syn_rec = []
        mem_rec = []
        spk_rec = []

        for i in range(2):
            spk, syn, mem = stein_instance(input_[i], syn, mem)
            syn_rec.append(syn)
            mem_rec.append(mem)
            spk_rec.append(spk)

        assert syn_rec[0] == 2 * syn_rec[1]
        assert mem_rec[1] == mem_rec[0] * 0.5 + syn_rec[1]
        assert spk_rec[0] == spk_rec[1]

    def test_stein_hidden_init(self):

        with pytest.raises(ValueError):
            snn.Stein(alpha=0.5, beta=0.5, hidden_init=True)
            snn.Stein(alpha=0.5, beta=0.5, num_inputs=1, hidden_init=True)
            snn.Stein(alpha=0.5, beta=0.5, batch_size=1, hidden_init=True)

        stein = snn.Stein(
            alpha=0.5, beta=0.5, num_inputs=1, batch_size=1, hidden_init=True
        )

        assert stein.spk == 0
        assert stein.syn == 0
        assert stein.mem == 0


class TestSRM0:

    with pytest.raises(ValueError):
        snn.SRM0(0.5, 0.5)

    def test_srm0(self, srm0_instance, input_):
        spk, syn_pre, syn_post, mem = srm0_instance.init_srm0(1)

        assert len(spk) == 1
        assert len(syn_pre) == 1
        assert len(syn_post) == 1
        assert len(mem) == 1

        syn_pre_rec = []
        syn_post_rec = []
        mem_rec = []
        spk_rec = []

        for i in range(2):
            spk, syn_pre, syn_post, mem = srm0_instance(
                input_[i], syn_pre, syn_post, mem
            )
            syn_pre_rec.append(syn_pre)
            syn_post_rec.append(syn_post)
            mem_rec.append(mem)
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]
        assert syn_pre_rec[0] + syn_post_rec[0] == 0
        assert syn_pre_rec[1] + syn_post_rec[1] > 0
        assert mem_rec[0] < mem_rec[1]

    def test_srm0_hidden_init(self):
        with pytest.raises(ValueError):
            snn.SRM0(alpha=0.6, beta=0.5, hidden_init=True)
            snn.SRM0(alpha=0.6, beta=0.5, num_inputs=1, hidden_init=True)
            snn.SRM0(alpha=0.6, beta=0.5, batch_size=1, hidden_init=True)

        srm0 = snn.SRM0(
            alpha=0.6, beta=0.5, num_inputs=1, batch_size=1, hidden_init=True
        )

        assert srm0.spk == 0
        assert srm0.syn_pre == 0
        assert srm0.syn_post == 0
        assert srm0.mem == 0


def test_fire():
    stein = snn.Stein(alpha=0.5, beta=0.5)
    input_large = torch.Tensor([stein.threshold * 10])
    assert stein.fire(input_large)[0] == 1


def test_instances():
    snn.LIF.instances = []
    snn.Stein(alpha=0.5, beta=0.5)
    snn.SRM0(alpha=0.5, beta=0.4)
    assert len(snn.LIF.instances) == 2

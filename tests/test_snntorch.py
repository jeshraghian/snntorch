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
def alpha_instance():
    return snn.Alpha(alpha=0.6, beta=0.5)


@pytest.fixture(scope="module")
def input_():
    return torch.Tensor([0.25, 0]).unsqueeze(-1)


class TestLeaky:
    def test_leaky(self, leaky_instance, input_):
        mem = leaky_instance.init_leaky()
        # assert len(mem) == 1

        mem_rec = []
        spk_rec = []

        for i in range(2):

            spk, mem = leaky_instance(input_[i], mem)
            mem_rec.append(mem)
            spk_rec.append(spk)

        assert mem_rec[1] == mem_rec[0] * 0.5 + input_[1]
        assert spk_rec[0] == spk_rec[1]

    # def test_leaky_init_hidden(self):

    #     with pytest.raises(ValueError):
    #         snn.Leaky(beta=0.5, init_hidden=True)
    #         snn.Leaky(beta=0.5, num_inputs=1, init_hidden=True)
    #         snn.Leaky(beta=0.5, batch_size=1, init_hidden=True)

    #     lif1 = snn.Leaky(beta=0.5, num_inputs=1, batch_size=1, init_hidden=True)

    #     assert lif1.spk == 0
    #     assert lif1.mem == 0


class TestSynaptic:
    def test_synaptic(self, synaptic_instance, input_):
        syn, mem = synaptic_instance.init_synaptic()
        # assert len(syn) == 1
        # assert len(mem) == 1

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

    # def test_synaptic_init_hidden(self):

    #     with pytest.raises(ValueError):
    #         snn.Synaptic(alpha=0.5, beta=0.5, init_hidden=True)
    #         snn.Synaptic(alpha=0.5, beta=0.5, num_inputs=1, init_hidden=True)
    #         snn.Synaptic(alpha=0.5, beta=0.5, batch_size=1, init_hidden=True)

    #     lif2 = snn.Synaptic(
    #         alpha=0.5, beta=0.5, num_inputs=1, batch_size=1, init_hidden=True
    #     )

    #     assert lif2.spk == 0
    #     assert lif2.syn == 0
    #     assert lif2.mem == 0


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

    # def test_lapicque_init_hidden(self):

    #     with pytest.raises(ValueError):
    #         snn.Lapicque(beta=0.5, init_hidden=True)
    #         snn.Lapicque(beta=0.5, num_inputs=1, init_hidden=True)
    #         snn.Lapicque(beta=0.5, batch_size=1, init_hidden=True)

    #     lapicque = snn.Lapicque(beta=0.5, num_inputs=1, batch_size=1, init_hidden=True)

    #     assert lapicque.spk == 0
    #     assert lapicque.mem == 0


# class TestStein:
#     def test_stein(self, stein_instance, input_):
#         syn, mem = stein_instance.init_stein()
#         # assert len(syn) == 1
#         # assert len(mem) == 1

#         syn_rec = []
#         mem_rec = []
#         spk_rec = []

#         for i in range(2):
#             spk, syn, mem = stein_instance(input_[i], syn, mem)
#             syn_rec.append(syn)
#             mem_rec.append(mem)
#             spk_rec.append(spk)

#         assert syn_rec[0] == 2 * syn_rec[1]
#         assert mem_rec[1] == mem_rec[0] * 0.5 + syn_rec[1]
#         assert spk_rec[0] == spk_rec[1]

# def test_stein_init_hidden(self):

#     with pytest.raises(ValueError):
#         snn.Stein(alpha=0.5, beta=0.5, init_hidden=True)
#         snn.Stein(alpha=0.5, beta=0.5, num_inputs=1, init_hidden=True)
#         snn.Stein(alpha=0.5, beta=0.5, batch_size=1, init_hidden=True)

#     stein = snn.Stein(
#         alpha=0.5, beta=0.5, num_inputs=1, batch_size=1, init_hidden=True
#     )

#     assert stein.spk == 0
#     assert stein.syn == 0
#     assert stein.mem == 0


class TestAlpha:

    with pytest.raises(ValueError):
        snn.Alpha(0.5, 0.5)

    def test_alpha(self, alpha_instance, input_):
        syn_exc, syn_inh, mem = alpha_instance.init_alpha()

        # assert len(syn_exc) == 1
        # assert len(syn_inh) == 1
        # assert len(mem) == 1

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
        assert syn_exc_rec[0] + syn_inh_rec[0] == 0
        assert syn_exc_rec[1] + syn_inh_rec[1] > 0
        assert mem_rec[0] < mem_rec[1]

    # def test_alpha_init_hidden(self):
    #     with pytest.raises(ValueError):
    #         snn.Alpha(alpha=0.6, beta=0.5, init_hidden=True)
    #         snn.Alpha(alpha=0.6, beta=0.5, num_inputs=1, init_hidden=True)
    #         snn.Alpha(alpha=0.6, beta=0.5, batch_size=1, init_hidden=True)

    #     alpha_response = snn.Alpha(
    #         alpha=0.6, beta=0.5, num_inputs=1, batch_size=1, init_hidden=True
    #     )

    #     assert alpha_response.spk == 0
    #     assert alpha_response.syn_exc == 0
    #     assert alpha_response.syn_inh == 0
    #     assert alpha_response.mem == 0


def test_fire():
    stein = snn.Stein(alpha=0.5, beta=0.5)
    input_large = torch.Tensor([stein.threshold * 10])
    assert stein.fire(input_large)[0] == 1


def test_instances():
    snn.LIF.instances = []
    snn.Stein(alpha=0.5, beta=0.5)
    snn.Alpha(alpha=0.5, beta=0.4)
    assert len(snn.LIF.instances) == 2

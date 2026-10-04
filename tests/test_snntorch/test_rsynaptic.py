#!/usr/bin/env python

"""Tests for RSynaptic neuron."""

import pytest
import snntorch as snn
import torch
import torch._dynamo as dynamo


@pytest.fixture(scope="module")
def input_():
    return torch.Tensor([0.25, 0]).unsqueeze(-1)


@pytest.fixture(scope="module")
def rsynaptic_instance():
    return snn.RSynaptic(
        alpha=0.5,
        beta=0.5,
        V=0.5,
        all_to_all=False,
    )


@pytest.fixture(scope="module")
def rsynaptic_instance_surrogate():
    return snn.RSynaptic(
        alpha=0.5, beta=0.5, V=0.5, all_to_all=False, surrogate_disable=True
    )


@pytest.fixture(scope="module")
def rsynaptic_reset_zero_instance():
    return snn.RSynaptic(
        alpha=0.5, beta=0.5, V=0.5, all_to_all=False, reset_mechanism="zero"
    )


@pytest.fixture(scope="module")
def rsynaptic_reset_none_instance():
    return snn.RSynaptic(
        alpha=0.5, beta=0.5, V=0.5, all_to_all=False, reset_mechanism="none"
    )


@pytest.fixture(scope="module")
def rsynaptic_hidden_instance():
    return snn.RSynaptic(
        alpha=0.5, beta=0.5, V=0.5, all_to_all=False, init_hidden=True
    )


@pytest.fixture(scope="module")
def rsynaptic_hidden_reset_zero_instance():
    return snn.RSynaptic(
        alpha=0.5,
        beta=0.5,
        V=0.5,
        init_hidden=True,
        all_to_all=False,
        reset_mechanism="zero",
    )


@pytest.fixture(scope="module")
def rsynaptic_hidden_reset_none_instance():
    return snn.RSynaptic(
        alpha=0.5,
        beta=0.5,
        V=0.5,
        init_hidden=True,
        all_to_all=False,
        reset_mechanism="none",
    )


class TestRSynaptic:
    def test_rsynaptic(self, rsynaptic_instance, input_):
        spk, syn, mem = rsynaptic_instance.init_rsynaptic()

        syn_rec = []
        mem_rec = []
        spk_rec = []

        for i in range(2):
            spk, syn, mem = rsynaptic_instance(input_[i], spk, syn, mem)
            syn_rec.append(syn)
            mem_rec.append(mem)
            spk_rec.append(spk)

        assert syn_rec[0] == 2 * syn_rec[1]
        assert mem_rec[1] == mem_rec[0] * 0.5 + syn_rec[1] + spk_rec[0]
        assert spk_rec[0] == spk_rec[1]

    def test_rsynaptic_reset(
        self,
        rsynaptic_instance,
        rsynaptic_reset_zero_instance,
        rsynaptic_reset_none_instance,
    ):
        lif1 = rsynaptic_instance
        lif2 = rsynaptic_reset_zero_instance
        lif3 = rsynaptic_reset_none_instance

        assert lif1.reset_mechanism_val == 0
        assert lif2.reset_mechanism_val == 1
        assert lif3.reset_mechanism_val == 2

        lif1.reset_mechanism = "zero"
        lif2.reset_mechanism = "none"
        lif3.reset_mechanism = "subtract"

        assert lif1.reset_mechanism_val == 1
        assert lif2.reset_mechanism_val == 2
        assert lif3.reset_mechanism_val == 0

    def test_rsynaptic_init_hidden(self, rsynaptic_hidden_instance, input_):

        spk_rec = []

        for i in range(2):
            spk = rsynaptic_hidden_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_rsynaptic_init_hidden_reset_zero(
        self, rsynaptic_hidden_reset_zero_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = rsynaptic_hidden_reset_zero_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_rsynaptic_init_hidden_reset_none(
        self, rsynaptic_hidden_reset_none_instance, input_
    ):

        spk_rec = []

        for i in range(2):
            spk = rsynaptic_hidden_reset_none_instance(input_[i])
            spk_rec.append(spk)

        assert spk_rec[0] == spk_rec[1]

    def test_rsynaptic_cases(self, rsynaptic_hidden_instance, input_):
        with pytest.raises(TypeError):
            rsynaptic_hidden_instance(input_, input_)

    def test_rsynaptic_compile_fullgraph(
        self, rsynaptic_instance_surrogate, input_
    ):
        explanation = dynamo.explain(rsynaptic_instance_surrogate)(input_[0])

        assert explanation.graph_break_count == 0


class TestRSynapticHiddenState:
    """Regression tests for issue #323.

    RSynaptic.detach_hidden / reset_hidden must treat ``spk`` the same
    way RLeaky does (PR #108): detach it from the graph and zero it on
    reset. ``spk`` is registered hidden state via ``register_buffer``.
    """

    def setup_method(self):
        snn.SpikingNeuron.init()

    def teardown_method(self):
        snn.SpikingNeuron.init()

    @staticmethod
    def _make_layer():
        return snn.RSynaptic(
            alpha=0.5,
            beta=0.5,
            V=0.5,
            all_to_all=False,
            init_hidden=True,
        )

    def test_reset_hidden_clears_spk_syn_mem(self):
        lif = self._make_layer()
        _ = lif(torch.ones(1, 4) * 5.0)
        lif.spk = torch.ones_like(lif.spk)
        assert bool(lif.spk.any())

        snn.RSynaptic.reset_hidden()

        assert not bool(lif.spk.any())
        assert not bool(lif.syn.any())
        assert not bool(lif.mem.any())

    def test_detach_hidden_detaches_spk_syn_mem(self):
        lif = self._make_layer()
        _ = lif(torch.ones(1, 4) * 5.0)
        assert lif.spk.grad_fn is not None
        assert lif.syn.grad_fn is not None
        assert lif.mem.grad_fn is not None

        snn.RSynaptic.detach_hidden()

        assert lif.spk.grad_fn is None
        assert lif.syn.grad_fn is None
        assert lif.mem.grad_fn is None

    def test_hidden_reset_and_detach_before_first_forward(self):
        lif = self._make_layer()
        snn.RSynaptic.reset_hidden()
        snn.RSynaptic.detach_hidden()
        assert not bool(lif.spk.any())
        assert not bool(lif.syn.any())
        assert not bool(lif.mem.any())

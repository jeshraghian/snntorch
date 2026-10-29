#!/usr/bin/env python

"""Tests for Alpha neuron."""

import pytest
import snntorch as snn
import torch
import torch._dynamo as dynamo


@pytest.fixture(scope="module")
def input_():
    return torch.Tensor([0.25, 0]).unsqueeze(-1)


@pytest.fixture(scope="module")
def alpha_instance():
    return snn.Alpha(alpha=0.6, beta=0.5, reset_mechanism="subtract")


@pytest.fixture(scope="module")
def alpha_instance_surrogate():
    return snn.Alpha(
        alpha=0.6, beta=0.5, reset_mechanism="subtract", surrogate_disable=True
    )


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
    return snn.Alpha(
        alpha=0.6, beta=0.5, init_hidden=True, reset_mechanism="zero"
    )


@pytest.fixture(scope="module")
def alpha_hidden_reset_none_instance():
    return snn.Alpha(
        alpha=0.6, beta=0.5, init_hidden=True, reset_mechanism="none"
    )


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
        self,
        alpha_instance,
        alpha_reset_zero_instance,
        alpha_reset_none_instance,
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

    def test_alpha_compile_fullgraph(self, alpha_instance_surrogate, input_):
        explanation = dynamo.explain(alpha_instance_surrogate)(input_[0])

        assert explanation.graph_break_count == 0


def _reference_alpha_step(input_, syn_exc, syn_inh, mem, alpha, beta, mech):
    # Alpha update written out from its equations, one step at a time
    threshold = 1.0
    reset = (mem - threshold > 0).float()
    alpha = alpha.clamp(0, 1)
    beta = beta.clamp(0, 1)
    tau_alpha = torch.log(alpha) / (torch.log(beta) - torch.log(alpha)) + 1
    new_syn_exc = alpha * syn_exc + input_
    new_syn_inh = beta * syn_inh - input_
    new_mem = tau_alpha * (new_syn_exc + new_syn_inh)
    if mech == "subtract":
        new_syn_exc = new_syn_exc - threshold * reset
        new_syn_inh = new_syn_inh - (beta * syn_inh - input_) * reset
        new_mem = new_mem + syn_inh * reset
    elif mech == "zero":
        new_syn_exc = new_syn_exc - new_syn_exc * reset
        new_syn_inh = new_syn_inh - new_syn_inh * reset
        new_mem = new_mem - new_mem * reset
    spk = (new_mem - threshold > 0).float()
    return spk, new_syn_exc, new_syn_inh, new_mem


@pytest.mark.parametrize("mech", ["subtract", "zero", "none"])
def test_alpha_matches_reference(mech):
    torch.manual_seed(0)
    alpha = torch.tensor([0.95, 0.9, 0.8, 0.7, 0.6])
    beta = alpha - 0.2
    inputs = 2 * torch.rand(30, 4, 5)

    lif = snn.Alpha(
        alpha=alpha.clone(),
        beta=beta.clone(),
        learn_alpha=True,
        learn_beta=True,
        reset_mechanism=mech,
    )
    syn_exc, syn_inh, mem = lif.reset_mem()
    ref_alpha = alpha.clone().requires_grad_(True)
    ref_beta = beta.clone().requires_grad_(True)
    ref_syn_exc = ref_syn_inh = ref_mem = torch.zeros(4, 5)

    loss = ref_loss = 0
    for x in inputs:
        spk, syn_exc, syn_inh, mem = lif(x, syn_exc, syn_inh, mem)
        ref_spk, ref_syn_exc, ref_syn_inh, ref_mem = _reference_alpha_step(
            x, ref_syn_exc, ref_syn_inh, ref_mem, ref_alpha, ref_beta, mech
        )
        assert torch.equal(spk, ref_spk)
        torch.testing.assert_close(mem, ref_mem)
        torch.testing.assert_close(syn_exc, ref_syn_exc)
        torch.testing.assert_close(syn_inh, ref_syn_inh)
        loss = loss + mem.sum()
        ref_loss = ref_loss + ref_mem.sum()

    loss.backward()
    ref_loss.backward()
    torch.testing.assert_close(lif.alpha.grad, ref_alpha.grad)
    torch.testing.assert_close(lif.beta.grad, ref_beta.grad)

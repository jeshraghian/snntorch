#!/usr/bin/env python

"""Tests for SpikingNeuron.zeros (issue #423)."""

import torch
import snntorch as snn
from snntorch._neurons.neurons import SpikingNeuron


class TestSpikingNeuronZeros:
    def test_zeros_clears_a_plain_tensor_in_place(self):
        """It was a no-op: ``state = torch.zeros_like(state)`` only rebinds
        the loop variable. It must zero the caller's tensor in place."""
        t = torch.ones(3, 4)
        ptr = t.data_ptr()
        SpikingNeuron.zeros(t)
        assert torch.equal(t, torch.zeros_like(t))
        assert t.data_ptr() == ptr  # same storage, not a new tensor

    def test_zeros_multiple_args(self):
        t1 = torch.ones(3)
        t2 = torch.full((2, 2), 5.0)
        SpikingNeuron.zeros(t1, t2)
        assert torch.equal(t1, torch.zeros_like(t1))
        assert torch.equal(t2, torch.zeros_like(t2))

    def test_zeros_clears_hidden_state_after_forward(self):
        lif = snn.Leaky(beta=0.5, init_hidden=True)
        _ = lif(torch.ones(2, 4))
        assert (lif.mem != 0).any()
        SpikingNeuron.zeros(lif.mem)
        assert torch.equal(lif.mem, torch.zeros_like(lif.mem))

    def test_zeros_on_grad_requiring_leaf(self):
        """Resetting hidden state to zero *before* the first forward pass
        is a normal training pattern. At that point the state is a
        grad-requiring leaf, and a bare ``state.zero_()`` raises
        ``RuntimeError: a leaf Variable that requires grad is being used
        in an in-place operation``. ``zeros()`` must handle it and leave
        ``requires_grad`` unchanged."""
        mem = torch.ones(2, 4, requires_grad=True)  # non-zero grad-leaf
        assert mem.is_leaf and mem.requires_grad

        SpikingNeuron.zeros(mem)

        assert torch.equal(mem, torch.zeros(2, 4))
        assert mem.requires_grad  # untouched
        assert mem.grad is None

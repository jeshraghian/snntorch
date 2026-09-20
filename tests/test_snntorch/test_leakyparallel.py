#!/usr/bin/env python

"""Tests for LeakyParallel neuron."""

import pytest
import snntorch as snn
import torch


class TestLeakyParallel:
    def test_leakyparallel_scalar_beta_sets_diagonal(self):
        lp = snn.LeakyParallel(input_size=4, hidden_size=6, beta=0.42)
        diag = torch.diagonal(lp.rnn.weight_hh_l0.detach())
        assert torch.allclose(diag, torch.full((6,), 0.42), atol=1e-6)

    def test_leakyparallel_per_neuron_beta_sets_diagonal(self):
        """Regression: a per-neuron ``beta`` (length == hidden_size) must be
        written to the diagonal of ``weight_hh_l0``. It was previously
        dropped silently -- the length check was unreachable -- so the
        layer kept the RNN's random recurrent init."""
        beta = torch.linspace(0.1, 0.9, 6)
        lp = snn.LeakyParallel(input_size=4, hidden_size=6, beta=beta)
        whh = lp.rnn.weight_hh_l0.detach()
        assert torch.allclose(torch.diagonal(whh), beta, atol=1e-6)
        # weight_hh is forced diagonal by default
        off_diag = whh - torch.diag(torch.diagonal(whh))
        assert off_diag.abs().max().item() == 0.0

    def test_leakyparallel_length_one_beta_tensor(self):
        lp = snn.LeakyParallel(
            input_size=4, hidden_size=6, beta=torch.tensor([0.3])
        )
        diag = torch.diagonal(lp.rnn.weight_hh_l0.detach())
        assert torch.allclose(diag, torch.full((6,), 0.3), atol=1e-6)

    def test_leakyparallel_bad_beta_length_raises(self):
        with pytest.raises(ValueError):
            snn.LeakyParallel(
                input_size=4, hidden_size=6, beta=torch.tensor([0.3, 0.4])
            )

    def test_leakyparallel_per_neuron_beta_drives_recurrence(self):
        """With ``weight_hh`` diagonal, the per-neuron beta *is* the
        recurrent decay. Two layers identical except for beta -- inputs,
        ``weight_ih`` and both biases shared -- must produce different
        outputs. Without the per-neuron beta being written, both layers
        run at the same (default) recurrent weights and this fails."""
        torch.manual_seed(0)
        x = torch.rand(8, 2, 4) * 3.0
        a = snn.LeakyParallel(
            input_size=4, hidden_size=6, beta=torch.full((6,), 0.05)
        )
        b = snn.LeakyParallel(
            input_size=4, hidden_size=6, beta=torch.full((6,), 0.95)
        )
        b.load_state_dict(a.state_dict())  # copy everything...
        b._beta_buffer(torch.full((6,), 0.95), learn_beta=False)
        b._beta_to_weight_hh()  # ...then set only b's recurrent weights from its beta
        with torch.no_grad():
            assert not torch.equal(a(x), b(x))

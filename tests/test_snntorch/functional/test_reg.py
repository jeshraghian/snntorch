import pytest
import torch
from snntorch.functional import kl_rate_sparsity, temporal_sparsity, l1_rate_sparsity


class TestRegularization:
    def test_kl_rate_sparsity(self):
        # spk shape: [time_steps, batch_size, features]
        spk = torch.zeros(10, 5, 20)
        spk[0, :, :] = 1.0  # 10% average firing rate

        loss_fn = kl_rate_sparsity(target_rate=0.05, Lambda=1e-3)
        loss = loss_fn(spk)

        assert torch.isfinite(loss)
        assert loss > 0.0

    def test_temporal_sparsity(self):
        # Constant spikes across time -> zero difference -> zero penalty
        spk_const = torch.ones(10, 5, 20)
        loss_fn = temporal_sparsity(Lambda=1e-3)
        loss_const = loss_fn(spk_const)
        assert torch.allclose(loss_const, torch.tensor(0.0))

        # Alternating spikes -> high difference -> non-zero penalty
        spk_alt = torch.zeros(10, 5, 20)
        spk_alt[::2] = 1.0
        loss_alt = loss_fn(spk_alt)
        assert loss_alt > 0.0

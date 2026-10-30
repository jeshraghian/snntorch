import torch
import snntorch.functional as SF
from snntorch.functional.loss import SpikeTime


def _reference_first_spike_grad(spk_rec, grad_output):
    # Original per-element implementation of FirstSpike.backward
    first_spike_time = SpikeTime.FirstSpike.apply(spk_rec.detach())
    spk_time_grad = torch.zeros_like(spk_rec)
    for i in range(first_spike_time.size(0)):
        for j in range(first_spike_time.size(1)):
            spk_time_grad[first_spike_time[i, j].long(), i, j] = 1.0
    return -grad_output * spk_time_grad


def _reference_multi_spike_grad(spk_time_final, spk_rec, grad_output):
    # Original per-element implementation of MultiSpike.backward
    spk_time_grad = torch.zeros_like(spk_rec)
    for i in range(spk_time_final.size(0)):
        for j in range(spk_time_final.size(1)):
            for k in range(spk_time_final.size(2)):
                spk_time_grad[spk_time_final[i, j, k].long(), j, k] = (
                    -grad_output[i, j, k]
                )
    return spk_time_grad


def test_first_spike_backward_matches_reference():
    torch.manual_seed(0)
    spk_rec = (torch.rand(25, 8, 5) > 0.9).float()
    spk_rec[:, 0, 0] = 0  # neuron that never fires
    spk_rec = spk_rec.requires_grad_(True)
    grad_output = torch.randn(8, 5)

    SpikeTime.FirstSpike.apply(spk_rec).backward(grad_output)

    assert torch.equal(
        spk_rec.grad, _reference_first_spike_grad(spk_rec, grad_output)
    )


def test_ce_temporal_loss_backward():
    torch.manual_seed(0)
    spk_rec = (torch.rand(25, 8, 5) > 0.9).float().requires_grad_(True)
    targets = torch.randint(0, 5, (8,))

    SF.ce_temporal_loss()(spk_rec, targets).backward()

    assert spk_rec.grad.shape == spk_rec.shape
    # at most one gradient entry per (sample, class): its first spike time
    assert ((spk_rec.grad != 0).sum(0) <= 1).all()


def test_multi_spike_backward_matches_reference():
    torch.manual_seed(0)
    num_spikes = 4
    spk_rec = (torch.rand(25, 8, 5) > 0.85).float()
    spk_rec[:, 0, 0] = 0  # never fires: all spike times map to the last step
    spk_rec[:, 0, 1] = 0
    spk_rec[3, 0, 1] = 1  # fires fewer times than num_spikes
    spk_rec = spk_rec.requires_grad_(True)
    spk_time_final = SpikeTime.MultiSpike.apply(
        spk_rec.detach(), num_spikes, "cpu"
    )
    grad_output = torch.randn(num_spikes, 8, 5)

    SpikeTime.MultiSpike.apply(spk_rec, num_spikes, "cpu").backward(
        grad_output
    )

    assert torch.equal(
        spk_rec.grad,
        _reference_multi_spike_grad(spk_time_final, spk_rec, grad_output),
    )


def test_mse_temporal_loss_multi_spike_backward():
    torch.manual_seed(0)
    spk_rec = (torch.rand(120, 4, 3) > 0.9).float().requires_grad_(True)
    targets = torch.randint(0, 3, (4,))
    loss_fn = SF.mse_temporal_loss(
        on_target=torch.tensor((5, 10)),
        off_target=torch.tensor((100, 105)),
        multi_spike=True,
    )

    loss_fn(spk_rec, targets).backward()

    assert spk_rec.grad.shape == spk_rec.shape
    assert (spk_rec.grad != 0).any()

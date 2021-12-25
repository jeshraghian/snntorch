import torch
import numpy as np


def accuracy_rate(spk_out, targets, population_code=False, num_classes=False):
    """Use spike count to measure accuracy.

    :param spk_out: Output spikes of shape [num_steps x batch_size x num_outputs]
    :type spk_out: torch.Tensor

    :param targets: Target tensor (without one-hot-encoding) of shape [batch_size]
    :type targets: torch.Tensor

    :return: accuracy
    :rtype: numpy.float64
    """
    if population_code:
        _, _, num_outputs = _prediction_check(spk_out)
        _, idx = _population_code(spk_out, num_classes, num_outputs).max(1)

    else:
        _, idx = spk_out.sum(dim=0).max(1)
    accuracy = np.mean((targets == idx).detach().cpu().numpy())

    return accuracy


def accuracy_temporal(spk_out, targets):

    device, _, _ = _prediction_check(spk_out)
    # convert spk_out into first spike
    spk_time = (
        spk_out.transpose(0, -1)
        * (torch.arange(0, spk_out.size(0)).detach().to(device) + 1)
    ).transpose(0, -1)

    """extact first spike time. Will be used to pass into loss function."""
    first_spike_time = torch.zeros_like(spk_time[0])
    for step in range(spk_time.size(0)):
        first_spike_time += (
            spk_time[step] * ~first_spike_time.bool()
        )  # mask out subsequent spikes

    """override element 0 (no spike) with shadow spike @ final time step, then offset by -1
    s.t. first_spike is at t=0."""
    first_spike_time += ~first_spike_time.bool() * (spk_time.size(0))
    first_spike_time -= 1  # fix offset

    # take idx of torch.min, see if it matches targets
    _, idx = first_spike_time.min(1)
    accuracy = np.mean((targets == idx).detach().cpu().numpy())

    return accuracy


def _prediction_check(spk_out):
    device = "cpu"
    if spk_out.is_cuda:
        device = "cuda"

    num_steps = spk_out.size(0)
    num_outputs = spk_out.size(-1)

    return device, num_steps, num_outputs


def _population_code(spk_out, num_classes, num_outputs):
    """Count up spikes sequentially from output classes."""
    if not num_classes:
        raise Exception(
            "``num_classes`` must be specified if ``population_code=True``."
        )
    if num_outputs % num_classes:
        raise Exception(
            f"``num_outputs {num_outputs} must be a factor of num_classes {num_classes}."
        )
    device = "cpu"
    if spk_out.is_cuda:
        device = "cuda"
    pop_code = torch.zeros(tuple([spk_out.size(1)] + [num_classes])).to(device)
    for idx in range(num_classes):
        pop_code[:, idx] = (
            spk_out[
                :,
                :,
                int(num_outputs * idx / num_classes) : int(
                    num_outputs * (idx + 1) / num_classes
                ),
            ]
            .sum(-1)
            .sum(0)
        )
    return pop_code

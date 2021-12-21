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

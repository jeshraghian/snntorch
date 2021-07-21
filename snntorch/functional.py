import torch
import torch.nn as nn
from snntorch import spikegen
import numpy as np

dtype = torch.float


def ce_rate_loss(spk_out, targets):
    """Cross Entropy Spike Rate Loss.
    The spikes at each time step are sequentially passed through the Cross Entropy Loss function.
    This criterion combines log_softmax and NLLLoss in a single function.
    The losses are accumulated over time steps to give the final loss.
    The Cross Entropy Loss encourages the correct class to fire at all time steps, and aims to suppress incorrect classes from firing.

    The Cross Entropy Rate Loss applies the Cross Entropy function at every time step. In contrast, the Cross Entropy Count Loss accumulates spikes first, and applies Cross Entropy Loss only once.

    :param spk_out: Output spikes of shape [num_steps x batch_size x num_classes]
    :type spk_out: torch.Tensor

    :param targets: Target tensor (without one-hot-encoding) of shape [batch_size]
    :type targets: torch.Tensor

    :return: cross entropy rate loss
    :rtype: torch.Tensor
    """

    device, num_steps, _ = _prediction_check(spk_out)
    log_softmax_fn = nn.LogSoftmax(dim=-1)
    loss_fn = nn.NLLLoss()

    log_p_y = log_softmax_fn(spk_out)
    loss = torch.zeros((1), dtype=dtype, device=device)

    for step in range(num_steps):
        loss += loss_fn(log_p_y[step], targets)

    return loss / num_steps


def ce_count_loss(spk_out, targets):
    """Cross Entropy Spike Count Loss.
    The spikes at each time step are accumulated and then passed through the Cross Entropy Loss function.
    This criterion combines log_softmax and NLLLoss in a single function.
    The Cross Entropy Loss encourages the correct class to fire at all time steps, and aims to suppress incorrect classes from firing.

    The Cross Entropy Count Loss accumulates spikes first, and applies Cross Entropy Loss only once. In contrast, the Cross Entropy Rate Loss applies the Cross Entropy function at every time step.

    :param spk_out: Output spikes of shape [num_steps x batch_size x num_classes]
    :type spk_out: torch.Tensor

    :param targets: Target tensor (without one-hot-encoding) of shape [batch_size]
    :type targets: torch.Tensor

    :return: cross entropy count loss
    :rtype: torch.Tensor
    """

    log_softmax_fn = nn.LogSoftmax(dim=-1)
    loss_fn = nn.NLLLoss()

    spike_count = torch.sum(spk_out, 0)  # B x C
    log_p_y = log_softmax_fn(spike_count)

    loss = loss_fn(log_p_y, targets)

    return loss


def ce_max_membrane_loss(mem_out, targets):
    """Cross Entropy Max Membrane Loss.
    The maximum membrane potential value for each output neuron is sampled and passed through the Cross Entropy Loss Function.
    This criterion combines log_softmax and NLLLoss in a single function.
    The Cross Entropy Loss encourages the maximum membrane potential of the correct class to increase, while suppressing the maximum membrane potential of incorrect classes.
    This function is adopted from SpyTorch by Friedemann Zenke.

    :param mem_out: Output membrane potential of shape [num_steps x batch_size x num_classes]
    :type mem_out: torch.Tensor

    :param targets: Target tensor (without one-hot-encoding) of shape [batch_size]
    :type targets: torch.Tensor

    :return: cross entropy max membrane loss
    :rtype: torch.Tensor
    """

    log_softmax_fn = nn.LogSoftmax(dim=-1)
    loss_fn = nn.NLLLoss()

    max_mem_out, _ = torch.max(mem_out, 0)
    log_p_y = log_softmax_fn(max_mem_out)

    loss = loss_fn(log_p_y, targets)

    return loss


def mse_count_loss(spk_out, targets, correct_rate=1, incorrect_rate=0):
    """Mean Square Error Spike Count Loss.
    The total spike count is accumulated over time for each neuron.
    The target spike count for correct classes is set to (num_steps * correct_rate), and for incorrect classes (num_steps * incorrect_rate).
    The spike counts and target spike counts are then applied to a Mean Square Error Loss Function.
    This function is adopted from SLAYER by Sumit Bam Shrestha and Garrick Orchard.

    :param spk_out: Output spikes of shape [num_steps x batch_size x num_classes]
    :type spk_out: torch.Tensor

    :param targets: Target tensor (without one-hot-encoding) of shape [batch_size]
    :type targets: torch.Tensor

    :param correct_rate: Firing frequency of correct class as a ratio, e.g., ``1`` promotes firing at every step; ``0.5`` promotes firing at 50% of steps, ``0`` discourages any firing, defaults to ``1``
    :type correct_rate: float, optional

    :param incorrect_rate: Firing frequency of incorrect class(es) as a ratio, e.g., ``1`` promotes firing at every step; ``0.5`` promotes firing at 50% of steps, ``0`` discourages any firing, defaults to ``1``
    :type incorrect_rate: float, optional

    :return: mean square error spike count loss
    :rtype: torch.Tensor
    """

    _, num_steps, num_classes = _prediction_check(spk_out)
    loss_fn = nn.MSELoss()

    # generate ideal spike-count in C sized vector
    on_target = int(num_steps * correct_rate)
    off_target = int(num_steps * incorrect_rate)
    spike_count_target = spikegen.targets_convert(
        targets, num_classes=num_classes, on_target=on_target, off_target=off_target
    )

    spike_count = torch.sum(spk_out, 0)  # B x C

    loss = loss_fn(spike_count, spike_count_target)

    return loss / num_steps


def mse_membrane_loss(mem_out, targets, time_var_targets=False):
    """Mean Square Error Membrane Loss.
    A target membrane potential is specified for every time step.
    The membrane potential and target are then applied to a Mean Square Error Loss Function.
    This function is adopted from Spike-Op by Jason K. Eshraghian.

    :param mem_out: Output membrane of shape [num_steps x batch_size x num_classes]
    :type mem_out: torch.Tensor

    :param targets: Target tensor of membrane potential. If ``time_var_targets=False``, targets should be of shape [batch_size]. If it is set to ``True``, targets should be of shape [num_steps x batch_size].
    :type targets: torch.Tensor

    :param time_var_targets: Specifies whether the targets are time-varying, defaults to ``False``
    :type correct_rate: bool, optional

    :return: mean square error membrane loss
    :rtype: torch.Tensor
    """

    device, num_steps, _ = _prediction_check(mem_out)
    loss = torch.zeros((1), dtype=dtype, device=device)
    loss_fn = nn.MSELoss()

    if time_var_targets:
        for step in range(num_steps):
            loss += loss_fn(mem_out[step], targets[step])
    else:
        for step in range(num_steps):
            loss += loss_fn(mem_out[step], targets)

    return loss / num_steps


# def ce_temporal_loss(spk_out, targets):
#     """Cross Entropy Temporal Loss.
#     The first spike time for each neuron is sampled from and the reciprocal is taken (i.e., 5-->1/5).
#     The cross entropy loss encourages the correct class to fire first (i.e., increasing the reciprocal spike time = reducing the spike time).
#     This function assumes at least one spike has occurred for correct and incorrect neurons.

#     :param spk_out: Output spikes of shape [num_steps x batch_size x num_classes]
#     :type spk_out: torch.Tensor

#     :param targets: Target tensor (without one-hot-encoding) of shape [batch_size]
#     :type targets: torch.Tensor

#     :return: cross entropy temporal loss
#     :rtype: torch.Tensor
#     """

#     device, _, _ = _prediction_check(spk_out)

#     log_softmax_fn = nn.LogSoftmax(dim=-1)
#     loss_fn = nn.NLLLoss()

#     _, first_spike_time = torch.max(spk_out, dim=0) # B x C
#     first_spike_time = first_spike_time.to(device)
#     log_p_y = log_softmax_fn(1/first_spike_time)
#     loss = loss_fn(log_p_y, targets)

#     return loss


def accuracy_rate(spk_out, targets):
    """Use spike count to measure accuracy.

    :param spk_out: Output spikes of shape [num_steps x batch_size x num_classes]
    :type spk_out: torch.Tensor

    :param targets: Target tensor (without one-hot-encoding) of shape [batch_size]
    :type targets: torch.Tensor

    :return: accuracy
    :rtype: numpy.float64
    """
    _, idx = spk_out.sum(dim=0).max(1)
    accuracy = np.mean((targets == idx).detach().cpu().numpy())

    return accuracy


def l1_sparsity(spk_out, Lambda=1e-5):
    """L1 regularization using total spike count as the penalty term."""
    return Lambda * torch.sum(spk_out)


# def l2_sparsity(mem_out, Lambda=1e-6):
#     """L2 regularization using accumulated membrane potential as the penalty term."""
#     return Lambda * (torch.sum(mem_out)**2)


def _prediction_check(spk_out):
    device = "cpu"
    if spk_out.is_cuda:
        device = "cuda"

    num_steps = spk_out.size(0)
    num_classes = spk_out.size(-1)

    return device, num_steps, num_classes

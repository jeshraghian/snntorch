import torch
import torch.nn as nn
from snntorch import spikegen
import numpy as np

#### Note: when adding new loss or regularization functions, be sure to update criterion_dict / reg_dict in backprop.py

dtype = torch.float


class LossFunctions:
    def _prediction_check(self, spk_out):
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


class ce_rate_loss(LossFunctions):
    """Cross Entropy Spike Rate Loss.
    When called, the spikes at each time step are sequentially passed through the Cross Entropy Loss function.
    This criterion combines log_softmax and NLLLoss in a single function.
    The losses are accumulated over time steps to give the final loss.
    The Cross Entropy Loss encourages the correct class to fire at all time steps, and aims to suppress incorrect classes from firing.

    The Cross Entropy Rate Loss applies the Cross Entropy function at every time step. In contrast, the Cross Entropy Count Loss accumulates spikes first, and applies Cross Entropy Loss only once.


    Example::

        import snntorch.functional as SF

        loss_fn = SF.ce_rate_loss()
        loss = loss_fn(outputs, targets)

    """

    def __init__(self):
        self.__name__ = "ce_rate_loss"

    def __call__(self, spk_out, targets):
        device, num_steps, _ = self._prediction_check(spk_out)
        log_softmax_fn = nn.LogSoftmax(dim=-1)
        loss_fn = nn.NLLLoss()

        log_p_y = log_softmax_fn(spk_out)
        loss = torch.zeros((1), dtype=dtype, device=device)

        for step in range(num_steps):
            loss += loss_fn(log_p_y[step], targets)

        return loss / num_steps


class ce_count_loss(LossFunctions):
    """Cross Entropy Spike Count Loss.

    The spikes at each time step [num_steps x batch_size x num_outputs] are accumulated and then passed through the Cross Entropy Loss function.
    This criterion combines log_softmax and NLLLoss in a single function.
    The Cross Entropy Loss encourages the correct class to fire at all time steps, and aims to suppress incorrect classes from firing.

    The Cross Entropy Count Loss accumulates spikes first, and applies Cross Entropy Loss only once.
    In contrast, the Cross Entropy Rate Loss applies the Cross Entropy function at every time step.

    Example::

        import snntorch.functional as SF

        # if not using population codes (i.e., more output neurons than there are classes)
        loss_fn = ce_count_loss()
        loss = loss_fn(spk_out, targets)

        # if using population codes; e.g., 200 output neurons, 10 output classes --> 20 output neurons p/class
        loss_fn = ce_count_loss(population_code=True, num_classes=10)
        loss = loss_fn(spk_out, targets)

    :param population_code: Specify if a population code is applied, i.e., the number of outputs is greater than the number of classes. Defaults to ``False``
    :type population_code: bool, optional

    :param num_classes: Number of output classes must be specified if ``population_code=True``. Must be a factor of the number of output neurons if population code is enabled. Defaults to ``False``
    :type num_classes: int, optional

    """

    def __init__(self, population_code=False, num_classes=False):
        self.population_code = population_code
        self.num_classes = num_classes
        self.__name__ = "ce_count_loss"

    def __call__(self, spk_out, targets):
        log_softmax_fn = nn.LogSoftmax(dim=-1)
        loss_fn = nn.NLLLoss()

        if self.population_code:
            _, _, num_outputs = self._prediction_check(spk_out)
            spike_count = _population_code(spk_out, self.num_classes, num_outputs)
        else:
            spike_count = torch.sum(spk_out, 0)  # B x C
        log_p_y = log_softmax_fn(spike_count)

        loss = loss_fn(log_p_y, targets)

        return loss


class ce_max_membrane_loss(LossFunctions):
    """Cross Entropy Max Membrane Loss.
    When called, the maximum membrane potential value for each output neuron is sampled and passed through the Cross Entropy Loss Function.
    This criterion combines log_softmax and NLLLoss in a single function.
    The Cross Entropy Loss encourages the maximum membrane potential of the correct class to increase, while suppressing the maximum membrane potential of incorrect classes.
    This function is adopted from SpyTorch by Friedemann Zenke.

    Example::

        import snntorch.functional as SF

        loss_fn = SF.ce_max_membrane_loss()
        loss = loss_fn(outputs, targets)

    """

    def __init__(self):
        self.__name__ = "ce_max_membrane_loss"

    def __call__(self, mem_out, targets):
        log_softmax_fn = nn.LogSoftmax(dim=-1)
        loss_fn = nn.NLLLoss()

        max_mem_out, _ = torch.max(mem_out, 0)
        log_p_y = log_softmax_fn(max_mem_out)

        loss = loss_fn(log_p_y, targets)

        return loss


class mse_count_loss(LossFunctions):
    """Mean Square Error Spike Count Loss.
    When called, the total spike count is accumulated over time for each neuron.
    The target spike count for correct classes is set to (num_steps * correct_rate), and for incorrect classes (num_steps * incorrect_rate).
    The spike counts and target spike counts are then applied to a Mean Square Error Loss Function.
    This function is adopted from SLAYER by Sumit Bam Shrestha and Garrick Orchard.

    Example::

        import snntorch.functional as SF

        loss_fn = SF.mse_count_loss(correct_rate=0.75, incorrect_rate=0.25)
        loss = loss_fn(outputs, targets)


    :param correct_rate: Firing frequency of correct class as a ratio, e.g., ``1`` promotes firing at every step; ``0.5`` promotes firing at 50% of steps, ``0`` discourages any firing, defaults to ``1``
    :type correct_rate: float, optional

    :param incorrect_rate: Firing frequency of incorrect class(es) as a ratio, e.g., ``1`` promotes firing at every step; ``0.5`` promotes firing at 50% of steps, ``0`` discourages any firing, defaults to ``1``
    :type incorrect_rate: float, optional

    :param population_code: Specify if a population code is applied, i.e., the number of outputs is greater than the number of classes. Defaults to ``False``
    :type population_code: bool, optional

    :param num_classes: Number of output classes must be specified if ``population_code=True``. Must be a factor of the number of output neurons if population code is enabled. Defaults to ``False``
    :type num_classes: int, optional


    """

    def __init__(
        self, correct_rate=1, incorrect_rate=0, population_code=False, num_classes=False
    ):
        self.correct_rate = correct_rate
        self.incorrect_rate = incorrect_rate
        self.population_code = population_code
        self.num_classes = num_classes
        self.__name__ = "mse_count_loss"

    def __call__(self, spk_out, targets):
        _, num_steps, num_outputs = self._prediction_check(spk_out)
        loss_fn = nn.MSELoss()

        if not self.population_code:

            # generate ideal spike-count in C sized vector
            on_target = int(num_steps * self.correct_rate)
            off_target = int(num_steps * self.incorrect_rate)
            spike_count_target = spikegen.targets_convert(
                targets,
                num_classes=num_outputs,
                on_target=on_target,
                off_target=off_target,
            )

            spike_count = torch.sum(spk_out, 0)  # B x C

        else:
            on_target = int(
                num_steps * self.correct_rate * (num_outputs / self.num_classes)
            )
            off_target = int(
                num_steps * self.incorrect_rate * (num_outputs / self.num_classes)
            )
            spike_count_target = spikegen.targets_convert(
                targets,
                num_classes=self.num_classes,
                on_target=on_target,
                off_target=off_target,
            )
            spike_count = _population_code(spk_out, self.num_classes, num_outputs)

        loss = loss_fn(spike_count, spike_count_target)
        return loss / num_steps


class mse_membrane_loss(LossFunctions):
    """Mean Square Error Membrane Loss.
    When called, pass the output membrane of shape [num_steps x batch_size x num_outputs] and the target tensor of membrane potential.
    The membrane potential and target are then applied to a Mean Square Error Loss Function.
    This function is adopted from Spike-Op by Jason K. Eshraghian.

    Example::

        import snntorch.functional as SF

        # if targets are the same at each time-step
        loss_fn = mse_membrane_loss(time_var_targets=False)
        loss = loss_fn(outputs, targets)

        # if targets are time-varying
        loss_fn = mse_membrane_loss(time_var_targets=True)
        loss = loss_fn(outputs, targets)

    :param time_var_targets: Specifies whether the targets are time-varying, defaults to ``False``
    :type correct_rate: bool, optional

    :param on_target: Specify target membrane potential for correct class, defaults to ``1``
    :type on_target: float, optional

    :param off_target: Specify target membrane potential for incorrect class, defaults to ``0``
    :type off_target: float, optional


    """

    #  to-do: add **kwargs to modify other keyword args in spikegen.targets_convert
    def __init__(self, time_var_targets=False, on_target=1, off_target=0):
        self.time_var_targets = time_var_targets
        self.on_target = on_target
        self.off_target = off_target
        self.__name__ = "mse_membrane_loss"

    def __call__(self, mem_out, targets):
        device, num_steps, num_outputs = self._prediction_check(mem_out)
        targets = spikegen.targets_convert(
            targets,
            num_classes=num_outputs,
            on_target=self.on_target,
            off_target=self.off_target,
        )
        loss = torch.zeros((1), dtype=dtype, device=device)
        loss_fn = nn.MSELoss()

        if self.time_var_targets:
            for step in range(num_steps):
                loss += loss_fn(mem_out[step], targets[step])
        else:
            for step in range(num_steps):
                loss += loss_fn(mem_out[step], targets)

        return loss / num_steps


# assumes that spikes already exist
# def ce_temporal_loss(spk_out, targets):
#     """Cross Entropy Temporal Loss.
#     The first spike time for each neuron is sampled from and the reciprocal is taken (i.e., 5-->1/5).
#     The cross entropy loss encourages the correct class to fire first (i.e., increasing the reciprocal spike time = reducing the spike time).
#     This function assumes at least one spike has occurred for correct and incorrect neurons.

#     :param spk_out: Output spikes of shape [num_steps x batch_size x num_outputs]
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


class l1_rate_sparsity:
    """L1 regularization using total spike count as the penalty term.
    Lambda is a scalar factor for regularization."""

    def __init__(self, Lambda=1e-5):
        self.Lambda = Lambda
        self.__name__ = "l1_rate_sparsity"

    def __call__(self, spk_out):
        return self.Lambda * torch.sum(spk_out)


# # def l2_sparsity(mem_out, Lambda=1e-6):
# #     """L2 regularization using accumulated membrane potential as the penalty term."""
# #     return Lambda * (torch.sum(mem_out)**2)


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

import torch
from torch._C import Value
import torch.nn as nn
from snntorch import spikegen

###############################################################
# When adding new loss/reg functions, update criterion_dict   #
#                and reg_dict in backprop.py                  #
###############################################################


dtype = torch.float


class LossFunctions:
    def __init__(self, reduction, weight):
        self.reduction = reduction
        self.weight = weight

    def __call__(self, spk_out, targets):
        loss = self._compute_loss(spk_out, targets)
        return self._reduce(loss)

    def _prediction_check(self, spk_out):
        device = spk_out.device

        num_steps = spk_out.size(0)
        num_outputs = spk_out.size(-1)

        return device, num_steps, num_outputs

    def _population_code(self, spk_out, num_classes, num_outputs):
        """Count up spikes sequentially from output classes."""
        if not num_classes:
            raise Exception(
                "``num_classes`` must be specified if "
                "``population_code=True``."
            )
        if num_outputs % num_classes:
            raise Exception(
                f"``num_outputs {num_outputs} must be a factor "
                f"of num_classes {num_classes}."
            )
        device = spk_out.device
        pop_code = torch.zeros(tuple([spk_out.size(1)] + [num_classes])).to(
            device
        )
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

    def _intermediate_reduction(self):
        return self.reduction if self.weight is None else 'none'

    def _reduce(self, loss):
        # if reduction was delayed due to weight
        requires_reduction = self.weight is not None and self.reduction == 'mean'
        return loss.mean() if requires_reduction else loss


class ce_rate_loss(LossFunctions):
    """Cross Entropy Spike Rate Loss.
    When called, the spikes at each time step are sequentially passed
    through the Cross Entropy Loss function.
    This criterion combines log_softmax and NLLLoss in a single function.
    The losses are accumulated over time steps to give the final loss.
    The Cross Entropy Loss encourages the correct class to fire at all
    time steps, and aims to suppress incorrect classes from firing.

    The Cross Entropy Rate Loss applies the Cross Entropy function at
    every time step. In contrast, the Cross Entropy Count Loss accumulates
    spikes first, and applies Cross Entropy Loss only once.


    Example::

        import snntorch.functional as SF

        loss_fn = SF.ce_rate_loss()
        loss = loss_fn(outputs, targets)

    :return: Loss
    :rtype: torch.Tensor (single element)

    """

    def __init__(self, reduction='mean', weight=None):
        super().__init__(reduction=reduction, weight=weight)
        self.__name__ = "ce_rate_loss"

    def _compute_loss(self, spk_out, targets):
        device, num_steps, _ = self._prediction_check(spk_out)
        log_softmax_fn = nn.LogSoftmax(dim=-1)
        loss_fn = nn.NLLLoss(reduction=self._intermediate_reduction(), weight=self.weight)

        log_p_y = log_softmax_fn(spk_out)

        loss_shape = (spk_out.size(1)) if self._intermediate_reduction() == 'none' else (1)
        loss = torch.zeros(loss_shape, dtype=dtype, device=device)

        for step in range(num_steps):
            loss += loss_fn(log_p_y[step], targets)

        return loss / num_steps


class ce_count_loss(LossFunctions):
    """Cross Entropy Spike Count Loss.

    The spikes at each time step [num_steps x batch_size x num_outputs]
    are accumulated and then passed through the Cross Entropy Loss function.
    This criterion combines log_softmax and NLLLoss in a single function.
    The Cross Entropy Loss encourages the correct class to fire at all
    time steps, and aims to suppress incorrect classes from firing.

    The Cross Entropy Count Loss accumulates spikes first, and applies
    Cross Entropy Loss only once.
    In contrast, the Cross Entropy Rate Loss applies the Cross Entropy
    function at every time step.

    Example::

        import snntorch.functional as SF

        # if not using population codes (i.e., more output neurons than
        there are classes)
        loss_fn = ce_count_loss()
        loss = loss_fn(spk_out, targets)

        # if using population codes; e.g., 200 output neurons, 10 output
        classes --> 20 output neurons p/class
        loss_fn = ce_count_loss(population_code=True, num_classes=10)
        loss = loss_fn(spk_out, targets)

    :param population_code: Specify if a population code is applied, i.e.,
        the number of outputs is greater than the number of classes. Defaults
        to ``False``
    :type population_code: bool, optional

    :param num_classes: Number of output classes must be specified if
        ``population_code=True``. Must be a factor of the number of output
        neurons if population code is enabled. Defaults to ``False``
    :type num_classes: int, optional

    :return: Loss
    :rtype: torch.Tensor (single element)

    """

    def __init__(self, population_code=False, num_classes=False, reduction='mean', weight=None):
        super().__init__(reduction=reduction, weight=weight)
        self.population_code = population_code
        self.num_classes = num_classes
        self.__name__ = "ce_count_loss"

    def _compute_loss(self, spk_out, targets):
        log_softmax_fn = nn.LogSoftmax(dim=-1)
        loss_fn = nn.NLLLoss(reduction=self._intermediate_reduction(), weight=self.weight)

        if self.population_code:
            _, _, num_outputs = self._prediction_check(spk_out)
            spike_count = self._population_code(
                spk_out, self.num_classes, num_outputs
            )
        else:
            spike_count = torch.sum(spk_out, 0)  # B x C
        log_p_y = log_softmax_fn(spike_count)

        loss = loss_fn(log_p_y, targets)

        return loss


class ce_max_membrane_loss(LossFunctions):
    """Cross Entropy Max Membrane Loss.
    When called, the maximum membrane potential value for each output
    neuron is sampled and passed through the Cross Entropy Loss Function.
    This criterion combines log_softmax and NLLLoss in a single function.
    The Cross Entropy Loss encourages the maximum membrane potential of
    the correct class to increase, while suppressing the maximum membrane
    potential of incorrect classes.
    This function is adopted from SpyTorch by Friedemann Zenke.

    Example::

        import snntorch.functional as SF

        loss_fn = SF.ce_max_membrane_loss()
        loss = loss_fn(outputs, targets)

    :param mem_out: The output tensor of the SNN's membrane potential,
        of the dimension timestep * batch_size * num_output_neurons
    :type mem_out: torch.Tensor
    :param targets: The tensor containing the targets of the current
        mini-batch, of the dimension batch_size
    :type targets: torch.Tensor

    :return: Loss
    :rtype: torch.Tensor (single element)

    """

    def __init__(self, reduction='mean', weight=None):
        super().__init__(reduction=reduction, weight=weight)
        self.__name__ = "ce_max_membrane_loss"

    def _compute_loss(self, mem_out, targets):
        log_softmax_fn = nn.LogSoftmax(dim=-1)
        loss_fn = nn.NLLLoss(reduction=self._intermediate_reduction(), weight=self.weight)

        max_mem_out, _ = torch.max(mem_out, 0)
        log_p_y = log_softmax_fn(max_mem_out)

        loss = loss_fn(log_p_y, targets)

        return loss


class mse_count_loss(LossFunctions):
    """Mean Square Error Spike Count Loss.
    When called, the total spike count is accumulated over time for
    each neuron.
    The target spike count for correct classes is set to
    (num_steps * correct_rate), and for incorrect classes
    (num_steps * incorrect_rate).
    The spike counts and target spike counts are then applied to a
     Mean Square Error Loss Function.
    This function is adopted from SLAYER by Sumit Bam Shrestha and
    Garrick Orchard.

    Example::

        import snntorch.functional as SF

        loss_fn = SF.mse_count_loss(correct_rate=0.75, incorrect_rate=0.25)
        loss = loss_fn(outputs, targets)


    :param correct_rate: Firing frequency of correct class as a ratio, e.g.,
        ``1`` promotes firing at every step; ``0.5`` promotes firing at 50% of
        steps, ``0`` discourages any firing, defaults to ``1``
    :type correct_rate: float, optional

    :param incorrect_rate: Firing frequency of incorrect class(es) as a
        ratio, e.g., ``1`` promotes firing at every step; ``0.5`` promotes
        firing at 50% of steps, ``0`` discourages any firing, defaults to ``1``
    :type incorrect_rate: float, optional

    :param population_code: Specify if a population code is applied, i.e., the
        number of outputs is greater than the number of classes. Defaults to
        ``False``
    :type population_code: bool, optional

    :param num_classes: Number of output classes must be specified if
        ``population_code=True``. Must be a factor of the number of output
        neurons if population code is enabled. Defaults to ``False``
    :type num_classes: int, optional

    :return: Loss
    :rtype: torch.Tensor (single element)

    """

    def __init__(
        self,
        correct_rate=1,
        incorrect_rate=0,
        population_code=False,
        num_classes=False,
        reduction='mean',
        weight=None
    ):
        super().__init__(reduction=reduction, weight=weight)
        self.correct_rate = correct_rate
        self.incorrect_rate = incorrect_rate
        self.population_code = population_code
        self.num_classes = num_classes
        self.__name__ = "mse_count_loss"

    def _compute_loss(self, spk_out, targets):
        _, num_steps, num_outputs = self._prediction_check(spk_out)
        loss_fn = nn.MSELoss(reduction=self._intermediate_reduction())

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
                num_steps
                * self.correct_rate
                * (num_outputs / self.num_classes)
            )
            off_target = int(
                num_steps
                * self.incorrect_rate
                * (num_outputs / self.num_classes)
            )
            spike_count_target = spikegen.targets_convert(
                targets,
                num_classes=self.num_classes,
                on_target=on_target,
                off_target=off_target,
            )
            spike_count = self._population_code(
                spk_out, self.num_classes, num_outputs
            )

        loss = loss_fn(spike_count, spike_count_target)

        if self.weight is not None:
            loss = loss * self.weight[targets]

        return loss / num_steps


class mse_membrane_loss(LossFunctions):
    """Mean Square Error Membrane Loss.
    When called, pass the output membrane of shape [num_steps x batch_size x
    num_outputs] and the target tensor of membrane potential.
    The membrane potential and target are then applied to a Mean Square Error
    Loss Function.
    This function is adopted from Spike-Op by Jason K. Eshraghian.

    Example::

        import snntorch.functional as SF

        # if targets are the same at each time-step
        loss_fn = mse_membrane_loss(time_var_targets=False)
        loss = loss_fn(outputs, targets)

        # if targets are time-varying
        loss_fn = mse_membrane_loss(time_var_targets=True)
        loss = loss_fn(outputs, targets)

    :param time_var_targets: Specifies whether the targets are time-varying,
        defaults to ``False``
    :type correct_rate: bool, optional

    :param on_target: Specify target membrane potential for correct class,
        defaults to ``1``
    :type on_target: float, optional

    :param off_target: Specify target membrane potential for incorrect class,
        defaults to ``0``
    :type off_target: float, optional

    :return: Loss
    :rtype: torch.Tensor (single element)

    """

    #  to-do: add **kwargs to modify other keyword args in
    #  spikegen.targets_convert
    def __init__(self, time_var_targets=False, on_target=1, off_target=0, reduction='mean', weight=None):
        super().__init__(reduction=reduction, weight=weight)
        self.time_var_targets = time_var_targets
        self.on_target = on_target
        self.off_target = off_target
        self.__name__ = "mse_membrane_loss"

    def _compute_loss(self, mem_out, targets):
        device, num_steps, num_outputs = self._prediction_check(mem_out)
        targets_spikes = spikegen.targets_convert(
            targets,
            num_classes=num_outputs,
            on_target=self.on_target,
            off_target=self.off_target,
        )

        loss_shape = mem_out[0].shape if self._intermediate_reduction() == 'none' else (1)
        loss = torch.zeros(loss_shape, dtype=dtype, device=device)

        loss_fn = nn.MSELoss(reduction=self._intermediate_reduction())

        if self.time_var_targets:
            for step in range(num_steps):
                loss += loss_fn(mem_out[step], targets_spikes[step])
        else:
            for step in range(num_steps):
                loss += loss_fn(mem_out[step], targets_spikes)

        if self.weight is not None:
            loss = loss * self.weight[targets]

        return loss / num_steps


# Uses a sign estimator - approximates leaky as gradient is undefined.
# for neurons with defined gradients, this leads to an approximation.

# Use labels by default unless target_is_time = True
class SpikeTime(nn.Module):
    """Used by ce_temporal_loss and mse_temporal_loss to convert spike
    outputs into spike times."""

    def __init__(
        self,
        target_is_time=False,
        on_target=0,
        off_target=-1,
        tolerance=0,
        multi_spike=False,
    ):
        super().__init__()

        self.target_is_time = target_is_time
        self.tolerance = tolerance
        self.tolerance_fn = self.Tolerance.apply
        self.multi_spike = multi_spike

        if not self.target_is_time:
            self.on_target = on_target
            self.off_target = off_target  # override this with final step

        # function used to extract the first F spike times. If
        # multi_spike=False, F=1.
        if self.multi_spike:
            self.first_spike_fn = self.MultiSpike.apply
        else:
            self.first_spike_fn = self.FirstSpike.apply

    # spiking output from final layer is a recording: T x B x N
    # targets can either be labels or spike times
    def forward(self, spk_out, targets):
        self.device, num_steps, num_outputs = self._prediction_check(spk_out)

        # convert labels to spike times
        if not self.target_is_time:
            targets = self.labels_to_spike_times(targets, num_outputs)

        # convert negative spike times to time steps: -1 -->
        # ( num_steps+ (-1) )
        targets[targets < 0] = spk_out.size(0) + targets[targets < 0]

        # now operating in the spike-time domain rather than with labels
        # Consider merging multi-spike and single-spike?
        # single-spike is faster, so keep them separate for now.
        if self.multi_spike:
            self.spike_count = targets.size(0)
            spk_time_final = self.first_spike_fn(
                spk_out, self.spike_count, self.device
            )  # spk_time_final here means the first spike time
        else:
            spk_time_final = self.first_spike_fn(spk_out, self.device)

        # next need to check how tolerance copes with multi-spikes
        if self.tolerance:
            spk_time_final = self.tolerance_fn(
                spk_time_final, targets, self.tolerance
            )

        return spk_time_final, targets

    def _prediction_check(self, spk_out):
        # device = "cpu"
        # if spk_out.is_cuda:
        #     device = "cuda"
        device = spk_out.device

        num_steps = spk_out.size(0)
        num_outputs = spk_out.size(-1)

        return device, num_steps, num_outputs

    @staticmethod
    class FirstSpike(torch.autograd.Function):
        """Convert spk_rec of 1/0s [TxBxN] --> first spike time [BxN].
        Linearize df/dS=-1 if spike, 0 if no spike."""

        @staticmethod
        def forward(ctx, spk_rec, device="cpu"):
            """Convert spk_rec of 1/0s [TxBxN] --> spk_time [TxBxN].
            0's indicate no spike --> +1 is first time step.
            Transpose accounts for broadcasting along final dimension
            (i.e., multiply along T)."""
            spk_time = (
                spk_rec.transpose(0, -1)
                * (torch.arange(0, spk_rec.size(0)).detach().to(device) + 1)
            ).transpose(0, -1)

            """extact first spike time. Will be used to pass into loss
            function."""
            first_spike_time = torch.zeros_like(spk_time[0])
            for step in range(spk_time.size(0)):
                first_spike_time += (
                    spk_time[step] * ~first_spike_time.bool()
                )  # mask out subsequent spikes

            """override element 0 (no spike) with shadow spike @ final time
            step, then offset by -1
            s.t. first_spike is at t=0."""
            first_spike_time += ~first_spike_time.bool() * (spk_time.size(0))
            first_spike_time -= 1  # fix offset
            ctx.save_for_backward(first_spike_time, spk_rec)
            return first_spike_time

        @staticmethod
        def backward(ctx, grad_output):
            (first_spike_time, spk_rec) = ctx.saved_tensors
            spk_time_grad = torch.zeros_like(spk_rec)  # T x B x N

            """spike extraction step/indexing @ each step is
            non-differentiable.
            Apply sign estimator by substituting gradient for -1 ONLY at
            first spike time."""
            for i in range(first_spike_time.size(0)):
                for j in range(first_spike_time.size(1)):
                    spk_time_grad[first_spike_time[i, j].long(), i, j] = 1.0
            grad = -grad_output * spk_time_grad
            return grad, None

    @staticmethod
    class MultiSpike(torch.autograd.Function):
        """Convert spk_rec of 1/0s [TxBxN] --> first F spike times [FxBxN].
        Linearize df/dS=-1 if spike, 0 if no spike."""

        @staticmethod
        def forward(ctx, spk_rec, spk_count, device="cpu"):
            spk_rec_tmp = spk_rec.clone()
            spk_time_rec = []

            for step in range(spk_count):
                """Convert spk_rec of 1/0s [TxBxN] --> spk_time [TxBxN].
                0's indicate no spike --> +1 is first time step.
                Transpose accounts for broadcasting along final dimension
                (i.e., multiply along T)."""
                spk_time = (
                    spk_rec_tmp.transpose(0, -1)
                    * (
                        torch.arange(0, spk_rec_tmp.size(0))
                        .detach()
                        .to(device)
                        + 1
                    )
                ).transpose(0, -1)

                """extact n-th spike time (n=step) up to F."""
                nth_spike_time = torch.zeros_like(spk_time[0])
                for step in range(spk_time.size(0)):
                    nth_spike_time += (
                        spk_time[step] * ~nth_spike_time.bool()
                    )  # mask out subsequent spikes

                """override element 0 (no spike) with shadow spike @ final
                time step, then offset by -1
                s.t. first_spike is at t=0."""
                nth_spike_time += ~nth_spike_time.bool() * (
                    spk_time.size(0)
                )  # populate non-spiking with total size
                nth_spike_time -= 1  # fix offset
                spk_time_rec.append(nth_spike_time)

                """before looping, eliminate n-th spike. this avoids double
                counting spikes."""
                spk_rec_tmp[nth_spike_time.long()] = 0

            """Pass this into loss function."""
            spk_time_rec = torch.stack(spk_time_rec)

            ctx.save_for_backward(spk_time_rec, spk_rec)

            return spk_time_rec

        @staticmethod
        def backward(ctx, grad_output):
            (spk_time_final, spk_rec) = ctx.saved_tensors
            spk_time_grad = torch.zeros_like(spk_rec)  # T x B x N

            """spike extraction step/indexing @ each step is
            non-differentiable.
            Apply sign estimator by substituting gradient for -1 ONLY at
            F-th spike time."""
            for i in range(spk_time_final.size(0)):
                for j in range(spk_time_final.size(1)):
                    for k in range(spk_time_final.size(2)):
                        spk_time_grad[
                            spk_time_final[i, j, k].long(), j, k
                        ] = -grad_output[i, j, k]
            grad = spk_time_grad
            return grad, None, None

    @staticmethod
    class Tolerance(torch.autograd.Function):
        """If spike time is 'close enough' to target spike within tolerance,
        set the time to target for loss calc only."""

        # TO-DO: remove ctx?
        @staticmethod
        def forward(ctx, spk_time, target, tolerance):
            spk_time_clone = (
                spk_time.clone()
            )  # spk_time_clone: BxN (FxBxN for multi-spike); target: TxBxN
            spk_time_clone[torch.abs(spk_time - target) < tolerance] = (
                torch.ones_like(spk_time) * target
            )[torch.abs(spk_time - target) < tolerance]
            return spk_time_clone

        @staticmethod
        def backward(ctx, grad_output):
            grad = grad_output
            return grad, None, None

    def labels_to_spike_times(self, targets, num_outputs):
        """Convert index labels [B] into spike times."""

        if not self.multi_spike:
            targets = self.label_to_single_spike(targets, num_outputs)

        # pass in labels --> output multiple spikes
        # assumes on_target & off_target are iterable
        else:
            targets = self.label_to_multi_spike(targets, num_outputs)

        return targets

    def label_to_single_spike(self, targets, num_outputs):
        """Convert labels from neuron index (dim: B) to first spike time
        (dim: B x N)."""

        # guess: i designed this code with on_target >> off_target in mind
        targets = spikegen.targets_convert(
            targets,
            num_classes=num_outputs,
            on_target=self.on_target,
            off_target=self.off_target,
        )

        return targets

    def label_to_multi_spike(self, targets, num_outputs):
        """Convert labels from neuron index (dim: B) to multiple spike times
        (dim: F x B x N).
        F is the number of spikes per neuron. Assumes target is iterable
        along F."""

        num_spikes_on = len(self.on_target)
        num_spikes_off = len(self.off_target)

        if num_spikes_on != num_spikes_off:
            raise IndexError(
                f"`on_target` (length: {num_spikes_on}) must have the same "
                f"length as `off_target` (length: {num_spikes_off}."
            )

        # iterate through each spike
        targets_rec = []
        for step in range(num_spikes_on):
            target_step = spikegen.targets_convert(
                targets,
                num_classes=num_outputs,
                on_target=self.on_target[step],
                off_target=self.off_target[step],
            )
            targets_rec.append(target_step)
        targets_rec = torch.stack(targets_rec)

        return targets_rec


class mse_temporal_loss:
    """Mean Square Error Temporal Loss.

    The first spike time of each output neuron [batch_size x num_outputs] is
    measured against the desired spike time with the Mean Square Error Loss
    Function.
    Note that the derivative of each spike time with respect to the spike
    df/dU is non-differentiable for most neuron classes, and is set to a sign
    estimator of -1.
    I.e., increasing membrane potential causes a proportionately earlier
    firing time.

    The Mean Square Error Temporal Loss can account for multiple spikes by
    setting ``multi_spike=True``.
    If the actual spike time is close enough to the target spike time within
    a given tolerance, e.g., ``tolerance = 5`` time steps, then it does not
    contribute to the loss.

    Index labels are passed as the target by default.
    To enable passing in the spike time(s) for output neuron(s), set
    ``target_is_time=True``.

    Note: After spike times with specified targets, no penalty is applied
    for subsequent spiking.
    To eliminate later spikes, an additional target should be applied.

    Example::

        import torch
        import snntorch.functional as SF

        # default takes in idx labels as targets
        # correct classes aimed to fire by default at t=0, incorrect at t=-1
        (final time step)
        loss_fn = mse_temporal_loss()
        loss = loss_fn(spk_out, targets)

        # as above, but correct class fire @ t=5, incorrect at t=100 with a
        tolerance of 2 steps
        loss_fn = mse_temporal_loss(on_target=5, off_target=100, tolerance=2)
        loss = loss_fn(spk_out, targets)

        # as above with multiple spike time targets
        on_target = torch.tensor(5, 10)
        off_target = torch.tensor(100, 105)
        loss_fn = mse_temporal_loss(on_target=on_target,
        off_target=off_target, tolerance=2)
        loss = loss_fn(spk_out, targets)

        # specify first spike time for 5 neurons individually, zero tolerance
        target = torch.tensor(5, 10, 15, 20, 25)
        loss_fn = mse_temporal_loss(target_is_time=True)
        loss = loss_fn(spk_out, target)


    :param target_is_time: Specify if target is specified as spike times
        (True) or as neuron indexes (False). Defaults to ``False``
    :type target_is_time: bool, optional

    :param on_target: Spike time for correct classes
        (only if target_is_time=False). Defaults to ``0``
    :type on_target: int
        (or interable over multiple int if ``multi_spike=True``), optional

    :param off_target: Spike time for incorrect classes
        (only if target_is_time=False).
        Defaults to ``-1``, i.e., final time step
    :type off_target: int (or interable over multiple int if
        ``multi_spike=True``), optional

    :param tolerance: If the distance between the spike time and target is
        less than the specified tolerance, then it does not contribute to the
        loss. Defaults to ``0``.
    :type tolerance: int, optional

    :param multi_spike: Specify if multiple spikes in target. Defaults to
        ``False``
    :type multi_spike: bool, optional

    :return: Loss
    :rtype: torch.Tensor (single element)

    """

    def __init__(
        self,
        target_is_time=False,
        on_target=0,
        off_target=-1,
        tolerance=0,
        multi_spike=False,
        reduction='mean',
        weight=None
    ):
        super().__init__()

        self.reduction = reduction
        self.weight = weight
        self.loss_fn = nn.MSELoss(reduction=('none' if self.weight is not None else self.reduction))
        self.spk_time_fn = SpikeTime(
            target_is_time, on_target, off_target, tolerance, multi_spike
        )
        self.__name__ = "mse_temporal_loss"

    def __call__(self, spk_rec, targets):
        spk_time, target_time = self.spk_time_fn(
            spk_rec, targets
        )  # return encoded targets
        loss = self.loss_fn(
            spk_time / spk_rec.size(0), target_time / spk_rec.size(0)
        )  # spk_time_final: num_spikes x B x Nc. # Same with targets.

        if self.weight is not None:
            loss = loss * self.weight[targets]
            if self.reduction == 'mean':
                loss = loss.mean()

        return loss


class ce_temporal_loss:
    """Cross Entropy Temporal Loss.

    The cross entropy loss of an 'inverted' first spike time of each output
    neuron [batch_size x num_outputs] is calculated.
    The 'inversion' is applied such that maximizing the value of the correct
    class decreases the first spike time (i.e., earlier spike).

    Options for inversion include: ``inverse='negate'`` which applies
    (-1 * output), or ``inverse='reciprocal'`` which takes (1/output).

    Note that the derivative of each spike time with respect to the spike
    df/dU is non-differentiable for most neuron classes, and is set to a
    sign estimator of -1.
    I.e., increasing membrane potential causes a proportionately earlier
    firing time.

    Index labels are passed as the target. To specify the exact spike time,
    use ``mse_temporal_loss`` instead.

    Note: After spike times with specified targets, no penalty is applied
    for subsequent spiking.

    Example::

        import torch
        import snntorch.functional as SF

        # correct classes aimed to fire by default at t=0, incorrect at
        final step
        loss_fn = ce_temporal_loss()
        loss = loss_fn(spk_out, targets)

    :param inverse: Specify how to invert output before taking cross
        entropy. Either scale by (-1 * x) with ``inverse='negate'`` or take the
        reciprocal (1/x) with ``inverse='reciprocal'``. Defaults to ``negate``
    :type inverse: str, optional

    :return: Loss
    :rtype: torch.Tensor (single element)


    """

    def __init__(self, inverse="negate", reduction='mean', weight=None):
        super().__init__()

        self.reduction = reduction
        self.weight = weight
        self.loss_fn = nn.CrossEntropyLoss(reduction=self.reduction, weight=self.weight)
        self.spk_time_fn = SpikeTime(target_is_time=False)
        self.inverse = inverse
        self._ce_temporal_cases()

        self.__name__ = "ce_temporal_loss"

    def __call__(self, spk_rec, targets):
        spk_time, _ = self.spk_time_fn(
            spk_rec, targets
        )  # return encoded targets
        if self.inverse == "negate":
            spk_time = -spk_time
        if self.inverse == "reciprocal":
            spk_time = 1 / (spk_time + 1)

        # loss = self.loss_fn(
        #     spk_time / spk_rec.size(0), targets / spk_rec.size(0)
        # )  # spk_time_final: num_spikes x B x Nc. # Same with targets.

        loss = self.loss_fn(
            spk_time, targets
        )  # spk_time_final: num_spikes x B x Nc. # Same with targets.

        return loss

    def _ce_temporal_cases(self):
        if self.inverse != "negate" and self.inverse != "reciprocal":
            raise ValueError(
                '`inverse` must be of type string containing either "negate" '
                'or "reciprocal".'
            )

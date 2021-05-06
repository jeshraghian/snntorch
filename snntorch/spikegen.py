import torch

dtype = torch.float


def rate(
    data, num_steps=False, gain=1, offset=0, first_spike_time=0, time_var_input=False
):

    """Spike rate encoding of input data. Convert tensor into Poisson spike trains using the features as the mean of a
    binomial distribution. If `num_steps` is specified, then the data will be first repeated in the first dimension
    before rate encoding.

    If data is time-varying, tensor dimensions use time first.

    Example::

        # 100% chance of spike generation
        a = torch.Tensor([1, 1, 1, 1])
        spikegen.rate(a)
        >>> tensor([1., 1., 1., 1.])

        # 0% chance of spike generation
        b = torch.Tensor([0, 0, 0, 0])
        spikegen.rate(b)
        >>> tensor([0., 0., 0., 0.])

        # 50% chance of spike generation per time step
        c = torch.Tensor([0.5, 0.5, 0.5, 0.5])
        spikegen.rate(c)
        >>> tensor([0., 1., 0., 1.])

        # Specifying num_steps treats each input element as a separate neuron
        d = spikegen.rate(torch.Tensor([0.5, 0.5, 0.5, 0.5]), num_steps = 2)
        print(d.size())
        >>> torch.Size([2, 4])

        print(c.size())
        >>> torch.Size([4])

    :param data: Data tensor for a single batch of shape [batch x input_size]
    :type data: torch.Tensor

    :param targets: Target tensor for a single batch, defaults to ``False``
    :type targets: torch.Tensor, optional

    :param num_outputs: Number of outputs, defaults to ``None``
    :type num_outputs: int, optional

    :param num_steps: Number of time steps. Only specify if input data does not already have time dimension, defaults to ``False``
    :type num_steps: int, optional

    :param gain: Scale input features by the gain, defaults to ``1``
    :type gain: float, optional

    :param offset: Shift input features by the offset, defaults to ``0``
    :type offset: torch.optim, optional

    :param first_spike_time: Time to first spike, defaults to ``0``.
    :type first_spike_time: int, optional

    :param time_var_input: Set to ``True`` if input tensor is time-varying. Otherwise, `first_spike_time!=0` will modify the wrong dimension. Defaults to ``False``
    :type time_var_input: bool, optional

    :return: rate encoding spike train of input features of shape [num_steps x batch x input_size]
    :rtype: torch.Tensor

    """

    if first_spike_time < 0 or num_steps < 0:
        raise Exception("``first_spike_time`` and ``num_steps`` cannot be negative.")

    if first_spike_time > (num_steps - 1):
        if num_steps:
            raise Exception(
                f"first_spike_time ({first_spike_time}) must be equal to or less than num_steps-1 ({num_steps-1})."
            )
        if not time_var_input:
            raise Exception(
                "If the input data is time-varying, set ``time_var_input=True``.\n If the input data is not time-varying, ensure ``num_steps > 0``."
            )

    if first_spike_time > 0 and not time_var_input and not num_steps:
        raise Exception(
            "``num_steps`` must be specified if both the input is not time-varying and ``first_spike_time`` is greater than 0."
        )

    if time_var_input and num_steps:
        raise Exception(
            "``num_steps`` should not be specified if input is time-varying, i.e., ``time_var_input=True``.\n The first dimension of the input data + ``first_spike_time`` will determine ``num_steps``."
        )

    device = torch.device("cuda") if data.is_cuda else torch.device("cpu")

    # intended for time-varying input data
    if time_var_input:
        spike_data = rate_conv(data)

        # zeros are added directly to the start of 0th (time) dimension
        if first_spike_time > 0:
            spike_data = torch.cat(
                (
                    torch.zeros(
                        tuple([first_spike_time] + list(spike_data[0].size())),
                        device=device,
                        dtype=dtype,
                    ),
                    spike_data,
                )
            )

    # intended for time-static input data
    else:

        # Generate a tuple: (num_steps, 1..., 1) where the number of 1's = number of dimensions in the original data.
        # Multiply by gain and add offset.
        time_data = (
            data.repeat(
                tuple([num_steps] + torch.ones(len(data.size()), dtype=int).tolist())
            )
            * gain
            + offset
        )

        spike_data = rate_conv(time_data)

        # zeros are multiplied by the start of the 0th (time) dimension
        if first_spike_time > 0:
            spike_data[0:first_spike_time] = 0

    return spike_data


def latency(
    data,
    num_steps=1,
    threshold=0.01,
    epsilon=1e-7,
    tau=1,
    first_spike_time=0,
    clip=False,
    normalize=False,
    linear=False,
):
    """Latency encoding of input data. Use input features to determine time-to-first spike. Assume a LIF neuron model
    that charges up with time constant tau. Tensor dimensions use time first.

    Example::

        a = torch.Tensor([0.02, 0.5, 1])
        spikegen.latency(a, num_steps=5, normalize=True, linear=True)
        >>> tensor([[0., 0., 1.],
                    [0., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 0.],
                    [1., 0., 0.]])

    :param data: Data tensor for a single batch of shape [batch x input_size]
    :type data: torch.Tensor

    :param num_steps: Number of time steps, defaults to ``1``
    :type num_steps: int, optional

    :param threshold: Input features below the threhold will fire at the final time step unless ``clip=True`` in which case they will not fire at all, defaults to ``0.01``
    :type threshold: float, optional

    :param epsilon:  A tiny positive value to avoid dividing by zero in firing time calculations, defaults to ``1e-7``
    :type epsilon: float, optional

    :param tau:  RC Time constant for LIF model used to calculate firing time, defaults to ``1``
    :type tau: float, optional

    :param first_spike_time: Time to first spike, defaults to ``0``.
    :type first_spike_time: int, optional

    :param clip:  Option to remove spikes from features that fall below the threshold, defaults to ``False``
    :type clip: Bool, optional

    :param normalize:  Option to normalize the latency code such that the final spike(s) occur within num_steps, defaults to ``False``
    :type normalize: Bool, optional

    :param linear:  Apply a linear latency code rather than the default logarithmic code, defaults to ``False``
    :type linear: Bool, optional

    :return: latency encoding spike train of input features
    :rtype: torch.Tensor
    """

    spike_time, idx = latency_code(
        data,
        num_steps=num_steps,
        threshold=threshold,
        epsilon=epsilon,
        tau=tau,
        first_spike_time=first_spike_time,
        normalize=normalize,
        linear=linear,
    )

    device = torch.device("cuda") if data.is_cuda else torch.device("cpu")

    if not num_steps:
        num_steps = 1

    spike_data = torch.zeros(
        (tuple([num_steps] + list(spike_time.size()))), dtype=dtype, device=device
    )
    clamp_flag = 0
    print_flag = True

    while True:  # can remove while loop + break.
        try:
            spike_data = spike_data.scatter(
                0, torch.round(spike_time).long().unsqueeze(0), 1
            )
            break
        except RuntimeError:  # this block runs if indexes in spike_time exceed range in num_steps.
            if print_flag:
                print(
                    "Warning: the spikes outside of the range of num_steps have been clipped.\n "
                    "Setting ``normalize = True`` or increasing ``num_steps`` can fix this."
                )
                print_flag = False
            # spike_data = torch.clamp_max(spike_data, num_steps - 1)
            spike_time = torch.clamp_max(spike_time, num_steps - 1)
            clamp_flag = 1

    if clamp_flag == 1:
        spike_data[-1] = 0

    # Use idx to remove spikes below the threshold
    if clip:
        spike_data = spike_data * ~idx  # idx is broadcast in T direction

    return spike_data


def delta(
    data,
    threshold=0.1,
    padding=False,
    off_spike=False,
):
    """Generate spike only when the difference between two subsequent time steps meets a threshold.
    Optionally include off_spikes for negative changes.

    Example::

        a = torch.Tensor([1, 2, 2.9, 3, 3.9])
        spikegen.delta(a, threshold=1)
        >>> tensor([1., 1., 0., 1., 0.])

        spikegen.delta(a, threshold=1, padding=True)
        >>> tensor([0., 1., 0., 1., 0.])

        b = torch.Tensor([1, 2, 0, 2, 2.9])
        spikegen.delta(b, threshold=1, off_spike=True)
        >>> tensor([ 1.,  1., -1.,  1.,  0.])

        spikegen.delta(b, threshold=1, padding=True, off_spike=True)
        >>> tensor([ 0.,  1., -1.,  1.,  0.])

    :param data: Data tensor for a single batch of shape [num_steps x batch x input_size]
    :type data: torch.Tensor

    :param threshold: Input features with a change greater than the thresold across one timestep will generate a spike, defaults to ``0.1``
    :type thr: float, optional

    :param padding: Used to change how the first time step of spikes are measured. If ``True``, the first time step will be repeated with itself resulting in ``0``'s for the output spikes.
    If ``False``, the first time step will be padded with ``0``'s, defaults to ``False``
    :type padding: bool, optional

    :param off_spike: If ``True``, negative spikes for changes less than ``-threshold``, defaults to ``False``
    :type off_spike: bool, optional
    """

    if padding:
        data_offset = torch.cat((data[0].unsqueeze(0), data))[
            :-1
        ]  # duplicate first time step, remove final step
    else:
        data_offset = torch.cat((torch.zeros_like(data[0]).unsqueeze(0), data))[
            :-1
        ]  # add 0's to first step, remove final step

    if not off_spike:
        return torch.ones_like(data) * ((data - data_offset) >= threshold)

    else:
        on_spk = torch.ones_like(data) * ((data - data_offset) >= threshold)
        off_spk = -torch.ones_like(data) * ((data - data_offset) <= -threshold)
        return on_spk + off_spk


def rate_conv(data):
    """Convert tensor into Poisson spike trains using the features as the mean of a binomial distribution.

        Example::

            # 100% chance of spike generation
            a = torch.Tensor([1, 1, 1, 1])
            spikegen.rate(a)
            >>> tensor([1., 1., 1., 1.])

            # 0% chance of spike generation
            b = torch.Tensor([0, 0, 0, 0])
            spikegen.rate(b)
            >>> tensor([0., 0., 0., 0.])

            # 50% chance of spike generation per time step
            c = torch.Tensor([0.5, 0.5, 0.5, 0.5])
            spikegen.rate(c)
            >>> tensor([0., 1., 0., 1.])

    :param data: Data tensor for a single batch of shape [batch x input_size]
    :type data: torch.Tensor

    :return: rate encoding spike train of input features of shape [num_steps x batch x input_size]
    :rtype: torch.Tensor
    """

    # Clip all features between 0 and 1 so they can be used as probabilities.
    clipped_data = torch.clamp(data, min=0, max=1)

    # pass time_data matrix into bernoulli function.
    spike_data = torch.bernoulli(clipped_data)

    return spike_data


def latency_code(
    data,
    num_steps=False,
    threshold=0.01,
    epsilon=1e-7,
    tau=1,
    first_spike_time=0,
    normalize=False,
    linear=False,
):
    """Latency encoding of input data. Convert input features to spike times. Assumes a LIF neuron model
    that charges up with time constant tau by default.

    Example::

        a = torch.Tensor([0.02, 0.5, 1])
        spikegen.latency_code(a, num_steps=5, normalize=True, linear=True)
        >>> (tensor([3.9200, 2.0000, -0.0000]), tensor([False, False, False]))

    :param data: Data tensor for a single batch of shape [batch x input_size]
    :type data: torch.Tensor

    :param num_steps: Number of time steps. Explicitly needed if ``normalize=True``, defaults to ``False`` (then changed to ``1`` if ``normalize=False``)
    :type num_steps: int, optional

    :param threshold: Input features below the threhold will fire at the final time step unless ``clip=True`` in which case they will not fire at all, defaults to ``0.01``
    :type threshold: float, optional

    :param epsilon:  A tiny positive value to avoid dividing by zero in firing time calculations, defaults to ``1e-7``
    :type epsilon: float, optional

    :param tau:  RC Time constant for LIF model used to calculate firing time, defaults to ``1``
    :type tau: float, optional

    :param first_spike_time: Time to first spike, defaults to ``0``.
    :type first_spike_time: int, optional

    :param normalize:  Option to normalize the latency code such that the final spike(s) occur within num_steps, defaults to ``False``
    :type normalize: Bool, optional

    :param linear:  Apply a linear latency code rather than the default logarithmic code, defaults to ``False``
    :type linear: Bool, optional

    :return: latency encoding spike times of input features
    :rtype: torch.Tensor

    :return: Tensor of Boolean values which correspond to the latency encoding elements that fall below the threshold. Used in ``latency_conv`` to clip saturated spikes.
    :rtype: torch.Tensor
    """

    if (
        threshold <= 0 or threshold >= 1
    ):  # double check if this can just be threshold < 0 instead.
        raise Exception("Threshold must be between 0 and 1.")

    if tau <= 0:  # double check if this can just be threshold < 0 instead.
        raise Exception("``tau`` must be greater than 0.")

    if normalize and not num_steps:
        raise Exception("`num_steps` should not be empty if normalize is set to True.")

    if not num_steps:
        num_steps = 1

    if first_spike_time > (num_steps - 1):
        raise Exception(
            f"first_spike_time ({first_spike_time}) must be equal to or less than num_steps-1 ({num_steps-1})."
        )

    if first_spike_time and torch.max(data) > 1 and torch.min(data) < 0:
        raise Exception(
            "`first_spike_time` can only be applied to data between `0` and `1`."
        )

    if first_spike_time < 0:
        raise Exception("``first_spike_time`` cannot be negative.")

    idx = data < threshold

    if not linear:
        data = torch.clamp(
            data, threshold + epsilon
        )  # saturates all values below threshold.
        spike_time = tau * torch.log(data / (data - threshold))
        if (
            first_spike_time > 0
        ):  # linearly shift spike times s.t. first spike occurs at first_spike_time.
            spike_time = spike_time - first_spike_time * (
                torch.min(spike_time) / first_spike_time - 1
            )
        if normalize:
            # tmax = torch.Tensor([(threshold + epsilon) / epsilon])
            # spike_time = spike_time * (num_steps - 1) / (tau * torch.log(tmax))
            spike_time = (spike_time - first_spike_time) * (
                num_steps - first_spike_time - 1
            ) / torch.max(spike_time - first_spike_time) + first_spike_time

    elif linear:
        if torch.max(data) > 1 and not normalize:
            raise Exception(
                "For ``linear=True``, data > 1 results in negative spike times.\n"
                "Either set ``normalize=True``, or ensure all values in data <= 1."
            )
        if normalize:
            tau = num_steps - 1 - first_spike_time
        spike_time = (
            torch.clamp_max((-tau * (data - 1)), -tau * (threshold - 1))
            + first_spike_time
        )

        #  if data has values > 1, negative spike_time values will exist. Normalize btwn 0 & 1, then renormalize 1 --> num_steps.
        if torch.min(spike_time) < 0 and normalize:
            spike_time = (
                (spike_time - torch.min(spike_time))
                * (1 / (torch.max(spike_time) - torch.min(spike_time)))
                * (num_steps - 1)
            )

    return spike_time, idx


def targets_conv(
    targets,
    num_classes,
    code="rate",
    num_steps=False,
    first_spike_time=0,
    correct_rate=1,
    incorrect_rate=0,
    on_target=1,
    off_target=0,
    firing_pattern="regular",
    interpolate=False,
    smoothing=0,
):
    """Spike encoding of targets. Expected input is a 1-D tensor with index of targets.
    If the output tensor is time-varying, the returned tensor will have time in the first dimension.
    If it is not time-varying, then the returned tensor will omit the time dimension and use batch first.

    The following arguments will necessarily incur a time-varying output:
        ``code='latency'``, ``first_spike_time!=0``, ``correct_rate!=1``, or ``incorrect_rate!=0``

    The following arguments will produce an output tensor that may sensibly be applied as a target to either the output spike or the membrane potential, as the output will consistently be either a `1` or `0`:
        ``on_target=1``, ``off_target=0``, and ``interpolate=False``

    If any of the above 3 conditions do not hold, then the target is better suited for the output membrane potential, as the output will likely include values other than `1` and `0`.

    :param targets: Target tensor for a single batch. The target should be a class index in the range [0, C-1] where C=number of classes.
    :type targets: torch.Tensor

    :param num_classes:  Number of outputs.
    :type num_classes: int

    :param code:  Encoding scheme. Options of ``'rate'`` or ``'latency'``, defaults to ``'rate'``
    :type code: string, optional

    :param num_steps: Number of time steps, defaults to ``False``
    :type num_steps: int, optional

    :param first_spike_time: Time step for first spike to occur, defaults to ``0``
    :type first_spike_time: int, optional

    :param correct_rate: Firing frequency of correct class as a ratio, e.g., ``1`` enables firing at every step; ``0.5`` enables firing at 50% of steps, ``0`` means no firing, defaults to ``1``
    :type correct_rate: float, optional

    :param incorrect_rate: Firing frequency of incorrect class(es), e.g., ``1`` enables firing at every step; ``0.5`` enables firing at 50% of steps, ``0`` means no firing, defaults to ``0``
    :type incorrect_rate: float, optional

    :param on_target: Target at spike times, defaults to ``1``
    :type on_target: float, optional

    :param off_target: Target during refractory period, defaults to ``0``
    :type off_target: float, optional

    :param firing_pattern: Firing pattern of correct and incorrect classes. ``'regular'`` enables periodic firing, ``'uniform'`` samples spike times from a uniform distributions (duplicates are removed), ``'poisson'`` samples from a binomial distribution at each step where each probability is the firing frequency, defaults to ``'regular'``
    :type firing_pattern: string, optional

    :param interpolate: Applies linear interpolation such that there is a gradually increasing target up to each spike, defaults to ``False``
    :type interpolate: Bool, optional

    :param smoothing: Determines how gradually linear interpolation is applied for a latency coded spike in terms of number of steps, defaults to ``0``
    :type smoothing: int, optional

    :return: spike coded target of output neurons. If targets are time-varying, the output tensor will use time-first dimensions. Otherwise, time is omitted from the tensor.
    :rtype: torch.Tensor

    """

    # raise exceptions if num_steps is not supplied, and rates have been specified, or if latency is specified.

    if code == "rate":
        return targets_rate(
            targets=targets,
            num_classes=num_classes,
            num_steps=num_steps,
            first_spike_time=first_spike_time,
            correct_rate=correct_rate,
            incorrect_rate=incorrect_rate,
            on_target=on_target,
            off_target=off_target,
            firing_pattern=firing_pattern,
            interpolate=interpolate,
        )

    # uncomment this soon
    # if code is "latency":
    #     if not num_steps:
    #         raise Exception("``num_steps`` must be passed if latency coding is used.")
    #     return targets_latency(target=targets, num_classes=num_classes, num_steps=num_steps, first_spike_time = 0, on_target=on_target, off_target=off_target, interpolate=interpolate, smoothing=smoothing)
    # q: how do we use interp for latency?
    # if code is "latency":
    #     return targets_latency(targets=targets, num_classes=num_classes, num_steps=num_steps, on_target=on_target, off_target=off_target, interpolate=interpolate, smoothing=smoothing)
    # targets = to_one_hot(targets, num_classes, on_target, off_target).repeat(tuple([num_steps] + torch.ones(len(targets.size()), dtype=int).tolist()))


def targets_rate(
    targets,
    num_classes,
    num_steps=False,
    first_spike_time=0,
    correct_rate=1,
    incorrect_rate=0,
    on_target=1,
    off_target=0,
    firing_pattern="regular",
    interpolate=False,
):

    """Spike rate encoding of targets. Input tensor must be one-dimensional with target indexes.
    If the output tensor is time-varying, the returned tensor will have time in the first dimension.
    If it is not time-varying, then the returned tensor will omit the time dimension and use batch first.
    If ``first_spike_time!=0``, ``correct_rate!=1``, or ``incorrect_rate!=0``, the output tensor will be time-varying.

    If ``on_target=1``, ``off_target=0``, and ``interpolate=False``, then the target may sensibly be applied as a target for the output spike.
    IF any of the above 3 conditions do not hold, then the target would be better suited for the output membrane potential.

    :param targets: Target tensor for a single batch. The target should be a class index in the range [0, C-1] where C=number of classes.
    :type targets: torch.Tensor

    :param num_classes: Number of outputs.
    :type num_classes: int

    :param num_steps: Number of time steps, defaults to ``False``
    :type num_steps: int, optional

    :param first_spike_time: Time step for first spike to occur, defaults to ``0``
    :type first_spike_time: int, optional

    :param correct_rate: Firing frequency of correct class as a ratio, e.g., ``1`` enables firing at every step; ``0.5`` enables firing at 50% of steps, ``0`` means no firing, defaults to ``1``
    :type correct_rate: float, optional

    :param incorrect_rate: Firing frequency of incorrect class(es), e.g., ``1`` enables firing at every step; ``0.5`` enables firing at 50% of steps, ``0`` means no firing, defaults to ``0``
    :type incorrect_rate: float, optional

    :param on_target: Target at spike times, defaults to ``1``
    :type on_target: float, optional

    :param off_target: Target during refractory period, defaults to ``0``
    :type off_target: float, optional

    :param firing_pattern: Firing pattern of correct and incorrect classes. ``'regular'`` enables periodic firing, ``'uniform'`` samples spike times from a uniform distributions (duplicates are removed), ``'poisson'`` samples from a binomial distribution at each step where each probability is the firing frequency, defaults to ``'regular'``
    :type firing_pattern: string, optional

    :param interpolate: Applies linear interpolation such that there is a gradually increasing target up to each spike, defaults to ``False``
    :type interpolate: Bool, optional

    :return: rate coded target of output neurons. If targets are time-varying, the output tensor will use time-first dimensions. Otherwise, time is omitted from the tensor.
    :rtype: torch.Tensor
    """

    if not 0 < correct_rate < 1 or not 0 < incorrect_rate < 1:
        raise Exception(
            f"``correct_rate``{correct_rate} and ``incorrect_rate``{incorrect_rate} must be between 0 and 1."
        )

    if not num_steps and (correct_rate != 1 or incorrect_rate != 0):
        raise Exception(
            "``num_steps`` must be passed if correct_rate is not 1 or incorrect_rate is not 0."
        )

    if incorrect_rate > correct_rate:
        raise Exception("``correct_rate`` must be greater than ``incorrect_rate``.")

    if firing_pattern.lower() not in ["regular", "uniform", "poisson"]:
        raise Exception(
            "``firing_pattern`` must be either 'regular', 'uniform' or 'poisson'."
        )

    device = torch.device("cuda") if targets.is_cuda else torch.device("cpu")

    # return a non time-varying tensor
    if correct_rate == 1 and incorrect_rate == 0:
        if first_spike_time == 0:
            return torch.clamp(to_one_hot(targets, num_classes) * on_target, off_target)

        # return time-varying tensor: off up to first_spike_time, then correct classes are on after
        if first_spike_time > 0:
            spike_targets = torch.clamp(
                to_one_hot(targets, num_classes) * on_target, off_target
            )
            spike_targets = spike_targets.repeat(
                tuple(
                    [num_steps]
                    + torch.ones(len(spike_targets.size()), dtype=int).tolist()
                )
            )
            spike_targets[0:first_spike_time] = off_target
            return spike_targets

            # executes if on/off firing rates are not 100% / 0%
    else:
        one_hot_targets = to_one_hot(targets, num_classes)
        one_hot_inverse = to_one_hot_inverse(one_hot_targets)

        # project one-hot-encodings along the time-axis (0th dim)
        one_hot_targets = one_hot_targets.repeat(
            tuple(
                [num_steps]
                + torch.ones(len(one_hot_targets.size()), dtype=int).tolist()
            )
        )
        one_hot_inverse = one_hot_inverse.repeat(
            tuple(
                [num_steps]
                + torch.ones(len(one_hot_inverse.size()), dtype=int).tolist()
            )
        )

        # create tensor of spike_targets for correct class
        correct_spike_targets, correct_spike_times = target_rate_code(
            num_steps=num_steps,
            first_spike_time=first_spike_time,
            rate=correct_rate,
            firing_pattern=firing_pattern,
        )
        correct_spikes_one_hot = one_hot_targets * correct_spike_targets.to(
            device
        ).unsqueeze(-1).unsqueeze(
            -1
        )  # the two unsquezes make the dims of correct_spikes num_steps x 1 x 1, s.t. time is broadcast in every other direction

        # create tensor of spike targets for incorrect class
        incorrect_spike_targets, incorrect_spike_times = target_rate_code(
            num_steps=num_steps,
            first_spike_time=first_spike_time,
            rate=incorrect_rate,
            firing_pattern=firing_pattern,
        )
        incorrect_spikes_one_hot = one_hot_inverse * incorrect_spike_targets.to(
            device
        ).unsqueeze(-1).unsqueeze(
            -1
        )  # the two unsquezes make the dims of correct_spikes num_steps x 1 x 1, s.t. time is broadcasted in every other direction

        # merge the incorrect and correct tensors
        if not interpolate:
            return torch.clamp(
                (
                    incorrect_spikes_one_hot.to(device)
                    + correct_spikes_one_hot.to(device)
                )
                * on_target,
                off_target,
            )

        # interpolate values between spikes
        else:
            correct_spike_targets = one_hot_targets * (
                rate_interpolate(
                    correct_spike_times,
                    num_steps=num_steps,
                    on_target=on_target,
                    off_target=off_target,
                )
                .to(device)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )  # the two unsquezes make the dims of correct_spikes num_steps x 1 x 1, s.t. the time is broadcasted in every other direction
            incorrect_spike_targets = one_hot_inverse * (
                rate_interpolate(
                    incorrect_spike_times,
                    num_steps=num_steps,
                    on_target=on_target,
                    off_target=off_target,
                )
                .to(device)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            return correct_spike_targets + incorrect_spike_targets


def target_rate_code(num_steps, first_spike_time=0, rate=1, firing_pattern="regular"):
    """
    Rate coding a single output neuron of tensor of length ``num_steps`` containing spikes, and another tensor containing the spike times.

    :param num_steps: Number of time steps, defaults to ``False``
    :type num_steps: int, optional

    :param first_spike_time: Time step for first spike to occur, defaults to ``0``
    :type first_spike_time: int, optional

    :param rate: Firing frequency as a ratio, e.g., ``1`` enables firing at every step; ``0.5`` enables firing at 50% of steps, ``0`` means no firing, defaults to ``1``
    :type rate: float, optional

    :param firing_pattern: Firing pattern of correct and incorrect classes. ``'regular'`` enables periodic firing, ``'uniform'`` samples spike times from a uniform distributions (duplicates are removed), ``'poisson'`` samples from a binomial distribution at each step where each probability is the firing frequency, defaults to ``'regular'``
    :type firing_pattern: string, optional

    :return: rate coded target of single neuron class of length ``num_steps``
    :rtype: torch.Tensor

    :return: rate coded spike times in terms of steps
    :rtype: torch.Tensor
    """

    if not 0 < rate < 1:
        raise Exception(f"``rate``{rate} must be between 0 and 1.")

    if first_spike_time > num_steps:
        raise Exception(
            f"``first_spike_time {first_spike_time} must be less than num_steps {num_steps}."
        )

    if rate == 0:
        return torch.zeros(num_steps), torch.Tensor()

    if firing_pattern.lower() == "regular":
        spike_times = torch.arange(first_spike_time, num_steps, 1 / rate)
        return (
            torch.zeros(num_steps).scatter(0, spike_times.long(), 1),
            spike_times.long(),
        )

    elif firing_pattern.lower() == "uniform":
        spike_times = (
            torch.rand(len(torch.arange(first_spike_time, num_steps, 1 / rate)))
            * (num_steps - first_spike_time)
            + first_spike_time
        )
        return (
            torch.zeros(num_steps).scatter(0, spike_times.long(), 1),
            spike_times.long(),
        )

    elif firing_pattern.lower() == "poisson":
        spike_targets = torch.bernoulli(
            torch.cat(
                (
                    # torch.zeros((first_spike_time), device=device),
                    # torch.ones((num_steps - first_spike_time), device=device) * rate,
                    torch.zeros((first_spike_time)),
                    torch.ones((num_steps - first_spike_time)) * rate,
                )
            )
        )
        return spike_targets, torch.where(spike_targets == 1)[0]


def rate_interpolate(spike_times, num_steps, on_target=1, off_target=0, epsilon=1e-7):
    """Apply linear interpolation to a tensor of target spike times to enable gradual increasing membrane.

    :param spike_times: spike time targets in terms of steps
    :type targets: torch.Tensor

    :param num_steps: Number of time steps, defaults to ``False``
    :type num_steps: int, optional

    :param on_target: Target at spike times, defaults to ``1``
    :type on_target: float, optional

    :param off_target: Target during refractory period, defaults to ``0``
    :type off_target: float, optional

    :param epsilon:  A tiny positive value to avoid rounding errors when using torch.arange, defaults to ``1e-7``
    :type epsilon: float, optional

    :return: interpolated target of output neurons. Output tensor will use time-first dimensions.
    :rtype: torch.Tensor

    """

    # if no spikes
    if not spike_times.numel():
        return torch.ones((num_steps)) * off_target

    current_time = -1

    interpolated_targets = torch.Tensor([])

    for step in range(num_steps):
        if step in spike_times:
            if step == (current_time + 1):
                interpolated_targets = torch.cat(
                    (interpolated_targets, torch.Tensor([on_target]))
                )
            else:
                interpolated_targets = torch.cat(
                    (
                        interpolated_targets,
                        torch.arange(
                            off_target,
                            on_target + epsilon,
                            (on_target - off_target) / (step - current_time - 1),
                        ),
                    )
                )
            current_time = step

    if torch.max(spike_times) < num_steps - 1:
        for step in range(int(torch.max(spike_times).item()), num_steps - 1):
            interpolated_targets = torch.cat(
                (interpolated_targets, torch.Tensor([off_target]))
            )
    return interpolated_targets


def to_one_hot_inverse(one_hot_targets):
    """Boolean inversion of a matrix of 1's and 0's.
    Used to merge the targets of correct and incorrect neuron classes in ``targets_rate``."""

    one_hot_inverse = one_hot_targets.clone()
    one_hot_inverse[one_hot_targets == 0] = 1
    one_hot_inverse[one_hot_targets != 0] = 0

    return one_hot_inverse


def to_one_hot(targets, num_classes):
    """One hot encoding of target labels.

    Example::

        targets = torch.tensor([0, 1, 2, 3])
        spikegen.targets_to_spikes(targets, num_classes=4)
        >>> tensor([[1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]])

    :param targets: Target tensor for a single batch
    :type targets: torch.Tensor

    :param num_classes: Number of classes
    :type num_classes: int

    :return: one-hot encoding of targets of shape [batch x num_classes]
    :rtype: torch.Tensor
    """

    device = torch.device("cuda") if targets.is_cuda else torch.device("cpu")

    # Initialize zeros. E.g, for MNIST: (batch_size, 10).
    one_hot = torch.zeros([len(targets), num_classes], device=device, dtype=dtype)

    # Unsqueeze converts dims of [100] to [100, 1]
    one_hot = one_hot.scatter(1, targets.type(torch.int64).unsqueeze(-1), 1)

    return one_hot


def from_one_hot(one_hot_label):
    """Convert one-hot encoding back into an integer

    Example::

        one_hot_label = torch.tensor([[1., 0., 0., 0.],
                                      [0., 1., 0., 0.],
                                      [0., 0., 1., 0.],
                                      [0., 0., 0., 1.]])
        spikegen.from_one_hot(one_hot_label)
        >>> tensor([0, 1, 2, 3])

    :param targets: one-hot label vector
    :type targets: torch.Tensor

    :return: targets
    :rtype: torch.Tensor
    """

    # one_hot_label = torch.where(one_hot_label == 1)[0][0]
    # return int(one_hot_label)

    one_hot_label = torch.where(one_hot_label == 1)[0]
    return one_hot_label

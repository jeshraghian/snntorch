import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float


def rate(
    data,
    targets=False,
    num_outputs=None,
    num_steps=1,
    gain=1,
    offset=0,
    one_hot=False,
    time_varying_targets=False,
):

    """Spike rate encoding of input data. Convert tensor into Poisson spike trains using the features as the mean of a
    binomial distribution.
    Optionally convert targets into temporal one-hot spike trains. Tensor dimensions use time first.

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

    :param targets: Target tensor for a single batch, defaults to ``False``
    :type targets: torch.Tensor, optional

    :param num_outputs: Number of outputs, defaults to ``None``
    :type num_outputs: int, optional

    :param num_steps: Number of time steps, defaults to ``1``
    :type num_steps: int, optional

    :param gain: Scale input features by the gain, defaults to ``1``
    :type gain: float, optional

    :param offset: Shift input features by the offset, defaults to ``0``
    :type offset: torch.optim

    :param one_hot: Convert targets to one-hot-representation if True, defaults to ``False``
    :type one_hot: Bool, optional

    :param time_varying_targets: Repeat targets along the time-axis if True, defaults to ``False``
    :type time_varying_targets: Bool, optional

    :return: rate encoding spike train of input features of shape [num_steps x batch x input_size]
    :rtype: torch.Tensor

    :return: optionally return targets with one-hot-encoding or temporal conversion, with time in the first dimension of shape [num_steps x batch x num_outputs]
    :rtype: torch.Tensor

    """

    # Generate a tuple: (1, num_steps, 1..., 1) where the number of 1's = number of dimensions in the original data.
    # Multiply by gain and add offset.
    time_data = (
        data.repeat(
            tuple([num_steps] + torch.ones(len(data.size()), dtype=int).tolist())
        )
        * gain
        + offset
    )

    spike_data = rate_conv(time_data)

    if targets is not False:
        return spike_data, target_handling(
            targets, num_outputs, num_steps, one_hot, time_varying_targets
        )

    else:
        return spike_data


def latency(
    data,
    targets=False,
    num_outputs=None,
    num_steps=1,
    threshold=0.01,
    epsilon=1e-7,
    tau=1,
    clip=False,
    normalize=False,
    linear=False,
    one_hot=False,
    time_varying_targets=False,
):
    """Latency encoding of input data. Use input features to determine time-to-first spike. Assume a LIF neuron model
    that charges up with time constant tau.
    Optionally convert targets into temporal one-hot spike trains. Tensor dimensions use time first.

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

    :param targets: Target tensor for a single batch, defaults to ``False``
    :type targets: torch.Tensor, optional

    :param num_outputs: Number of outputs, defaults to ``None``
    :type num_outputs: int, optional

    :param num_steps: Number of time steps. Only needed if ``normalize=True``, defaults to ``1``
    :type num_steps: int, optional

    :param threshold: Input features below the threhold will fire at the final time step unless ``clip=True`` in which case they will not fire at all, defaults to ``0.01``
    :type threshold: float, optional

    :param epsilon:  A tiny positive value to avoid dividing by zero in firing time calculations, defaults to ``1e-7``
    :type epsilon: float, optional

    :param tau:  RC Time constant for LIF model used to calculate firing time, defaults to ``1``
    :type tau: float, optional

    :param clip:  Option to remove spikes from features that fall below the threshold, defaults to ``False``
    :type clip: Bool, optional

    :param normalize:  Option to normalize the latency code such that the final spike(s) occur within num_steps, defaults to ``False``
    :type normalize: Bool, optional

    :param linear:  Apply a linear latency code rather than the default logarithmic code, defaults to ``False``
    :type linear: Bool, optional

    :param one_hot: Convert targets to one-hot-representation if True, defaults to ``False``
    :type one_hot: Bool, optional

    :param time_varying_targets: Repeat targets along the time-axis if True, defaults to ``False``
    :type time_varying_targets: Bool, optional

    :return: latency encoding spike train of input features
    :rtype: torch.Tensor

    :return: optionally return one-hot encoding of targets with time in the first dimension of shape [time x batch x num_outputs]
    :rtype: torch.Tensor
    """

    spike_data = latency_conv(
        data,
        num_steps=num_steps,
        threshold=threshold,
        epsilon=epsilon,
        tau=tau,
        clip=clip,
        normalize=normalize,
        linear=linear,
    )

    if targets is not False:
        return spike_data, target_handling(
            targets, num_outputs, num_steps, one_hot, time_varying_targets
        )

    else:
        return spike_data


def delta(
    data,
    targets=False,
    threshold=0.1,
    padding=False,
    off_spike=False,
    num_outputs=None,
    num_steps=False,
    time_varying_targets=False,
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

    :param targets: Target tensor for a single batch, defaults to ``False``
    :type targets: torch.Tensor, optional

    :param threshold: Input features with a change greater than the thresold across one timestep will generate a spike, defaults to ``0.1``
    :type thr: float, optional

    :param padding: Used to change how the first time step of spikes are measured. If ``True``, the first time step will be repeated with itself resulting in ``0``'s for the output spikes.
    If ``False``, the first time step will be padded with ``0``'s, defaults to ``False``
    :type padding: bool, optional

    :param off_spike: If ``True``, negative spikes for changes less than ``-threshold``, defaults to ``False``
    :type off_spike: bool, optional

    :param num_outputs: Number of outputs, defaults to ``None``
    :type num_outputs: int, optional

    :param num_steps: Number of time steps. Only needed if ``targets`` and ``time_varying_targets`` are ``True``, defaults to ``False``
    :type num_steps: bool, optional

    :param time_varying_targets: Repeat targets along the time-axis if ``True``, defaults to ``False``
    :type time_varying_targets: bool, optional
    """

    spike_data = delta_conv(data, threshold, padding, off_spike)

    if targets is not False:
        return spike_data, target_handling(
            targets, num_outputs, num_steps, time_varying_targets
        )

    else:
        return spike_data


def delta_conv(data, threshold=0.1, padding=False, off_spike=False):

    if padding:
        data_offset = torch.cat((data[0].unsqueeze(0), data))[
            :-1
        ]  # duplicate first time step, remove final step
    else:
        data_offset = torch.cat((torch.zeros_like(data[0]).unsqueeze(0), data))[
            :-1
        ]  # add 0's to first step, remove final step

    if not off_spike:
        spike_data = torch.ones_like(data) * ((data - data_offset) >= threshold)

    else:
        on_spk = torch.ones_like(data) * ((data - data_offset) >= threshold)
        off_spk = -torch.ones_like(data) * ((data - data_offset) <= -threshold)
        spike_data = on_spk + off_spk

    return spike_data


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


def latency_conv(
    data,
    num_steps=False,
    threshold=0.01,
    epsilon=1e-7,
    tau=1,
    clip=False,
    normalize=False,
    linear=False,
):
    """Latency encoding of input data. Convert input features to spikes that fire according to the latency code.
    Assumes a LIF neuron model that charges up with time constant tau by default.

    Example::

        a = torch.Tensor([0.02, 0.5, 1])
        spikegen.latency_conv(a, num_steps=5, normalize=True, linear=True)
        >>> tensor([[0., 0., 1.],
                    [0., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 0.],
                    [1., 0., 0.]])

    :param data: Data tensor for a single batch of shape [batch x input_size]
    :type data: torch.Tensor

    :param num_steps: Number of time steps. Only needed if ``normalize=True``, defaults to ``False``
    :type num_steps: int, optional

    :param threshold: Input features below the threhold will fire at the final time step unless ``clip=True`` in which case they will not fire at all, defaults to ``0.01``
    :type threshold: float, optional

    :param epsilon:  A tiny positive value to avoid dividing by zero in firing time calculations, defaults to ``1e-7``
    :type epsilon: float, optional

    :param tau:  RC Time constant for LIF model used to calculate firing time, defaults to ``1``
    :type tau: float, optional

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
        normalize=normalize,
        linear=linear,
    )

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


def latency_code(
    data,
    num_steps=False,
    threshold=0.01,
    epsilon=1e-7,
    tau=1,
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

    :param num_steps: Number of time steps. Only needed if ``normalize=True``, defaults to ``False``
    :type num_steps: int, optional

    :param threshold: Input features below the threhold will fire at the final time step unless ``clip=True`` in which case they will not fire at all, defaults to ``0.01``
    :type threshold: float, optional

    :param epsilon:  A tiny positive value to avoid dividing by zero in firing time calculations, defaults to ``1e-7``
    :type epsilon: float, optional

    :param tau:  RC Time constant for LIF model used to calculate firing time, defaults to ``1``
    :type tau: float, optional

    :param normalize:  Option to normalize the latency code such that the final spike(s) occur within num_steps, defaults to ``False``
    :type normalize: Bool, optional

    :param linear:  Apply a linear latency code rather than the default logarithmic code, defaults to ``False``
    :type linear: Bool, optional

    :return: latency encoding spike times of input features
    :rtype: torch.Tensor

    :return: Tensor of Boolean values which correspond to the latency encoding elements that are under the threshold. Used in ``latency_conv`` to clip saturated spikes.
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

    idx = data < threshold

    if not linear:
        data = torch.clamp(
            data, threshold + epsilon
        )  # saturates all values below threshold.
        spike_time = tau * torch.log(data / (data - threshold))
        if normalize:
            tmax = torch.Tensor([(threshold + epsilon) / epsilon])
            spike_time = spike_time * (num_steps - 1) / (tau * torch.log(tmax))

    elif linear:
        if torch.max(data) > 1 and not normalize:
            raise Exception(
                "For ``linear=True``, data > 1 results in negative spike times.\n"
                "Either set ``normalize=True``, or ensure all values in data <= 1."
            )
        if normalize:
            tau = num_steps - 1
        spike_time = -tau * (data - 1)
        spike_time = torch.clamp_max(spike_time, (-tau * (threshold - 1)))

        #  if data has values > 1, negative spike_time values will exist. Normalize btwn 0 & 1, then renormalize 1 --> num_steps.
        if torch.min(spike_time) < 0 and normalize:
            spike_time = (
                (spike_time - torch.min(spike_time))
                * (1 / (torch.max(spike_time) - torch.min(spike_time)))
                * (num_steps - 1)
            )

    return spike_time, idx


def target_handling(
    targets,
    num_outputs=None,
    num_steps=False,
    one_hot=False,
    time_varying_targets=False,
):
    """Options to convert targets into temporal targets and/or one-hot-encodings.

    :param targets: Target tensor for a single batch
    :type targets: torch.Tensor

    :param num_outputs:  Number of outputs, defaults to ``None``
    :type num_outputs: int, optional

    :param num_steps: Number of time steps, defaults to ``False``
    :type num_steps: int, optional

    :param one_hot: Convert targets to one-hot-representation if True, defaults to ``False``
    :type one_hot: Bool, optional

    :param time_varying_targets: Repeat targets along the time-axis if True, defaults to ``False``
    :type time_varying_targets: Bool, optional

    :return: targets with optional one-hot-encoding or temporal conversion with time in the first dimension of shape [num_steps x batch x num_outputs]
    :rtype: torch.Tensor
    """
    if one_hot:
        # One-hot encoding of targets. If time_varying_targets is True, repeat it along the first dimension, i.e., time first
        return targets_to_spikes(targets, num_outputs, num_steps, time_varying_targets)

    elif not one_hot and time_varying_targets:
        # Repeat target tensor in first dimension without converting targets to one-hot
        return targets.repeat(
            tuple([num_steps] + torch.ones(len(targets.size()), dtype=int).tolist())
        )

    else:
        return targets


def targets_to_spikes(
    targets, num_outputs=None, num_steps=False, time_varying_targets=False
):
    """Convert targets to one-hot encodings in the time-domain.

    Example::

        targets = torch.tensor([0, 1, 2, 3])
        spikegen.targets_to_spikes(targets, num_outputs=4)
        >>> tensor([[1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]])

    :param targets: Target tensor for a single batch
    :type targets: torch.Tensor

    :param num_outputs: Number of outputs, defaults to ``None``
    :type num_outputs: int, optional

    :param num_steps: Number of time steps, defaults to ``1``
    :type num_steps: int, optional

    :param time_varying_targets: Repeat targets along the time-axis if True, defaults to ``False``
    :type time_varying_targets: Bool, optional

    :return: latency encoding spike train of input features
    :rtype: torch.Tensor

    :return: one-hot encoding of targets with time in the first dimension of shape [time x batch x num_outputs]
    :rtype: torch.Tensor
    """

    # Autocalc num_outputs if not provided
    if num_outputs is None:
        print(
            "Warning: num_outputs will automatically be calculated using the number of unique values in "
            "targets.\n"
            "It is recommended to explicitly pass num_steps as an argument instead."
        )
        num_outputs = len(targets.unique())
        print(f"num_outputs has been calculated to be {num_outputs}.")

    targets_1h = to_one_hot(targets, num_outputs)

    if time_varying_targets:
        if not num_steps:
            raise Exception(
                "num_steps must be specified if time_varying_targets is True."
            )
        # Extend one-hot targets in time dimension. Create a new axis in the second dimension.
        # Allocate first dim to batch size, and subtract it off len(targets_1h.size())
        spike_targets = targets_1h.repeat(
            tuple([num_steps] + torch.ones(len(targets_1h.size()), dtype=int).tolist())
        )

    else:
        spike_targets = targets_1h

    return spike_targets


def to_one_hot(targets, num_outputs):
    """One hot encoding of target labels.

    Example::

        targets = torch.tensor([0, 1, 2, 3])
        spikegen.targets_to_spikes(targets, num_outputs=4)
        >>> tensor([[1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]])

    :param targets: Target tensor for a single batch
    :type targets: torch.Tensor

    :param num_outputs: Number of outputs
    :type num_outputs: int

    :return: one-hot encoding of targets of shape [batch x num_outputs]
    :rtype: torch.Tensor
    """
    # Initialize zeros. E.g, for MNIST: (batch_size, 10).
    one_hot = torch.zeros([len(targets), num_outputs], device=device, dtype=dtype)

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

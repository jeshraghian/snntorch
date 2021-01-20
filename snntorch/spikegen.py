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
    convert_targets=False,
    temporal_targets=False,
):
    """Spike rate encoding of input data. Convert tensor into Poisson spike trains using the features as the mean of a
    binomial distribution.
    Optionally convert targets into temporal one-hot spike trains. Tensor dimensions use time first.

               Parameters
               ----------
               data : torch tensor
                   Input of shape (batch, input_size).
               targets : torch tensor, optional
                   Target tensor for a single batch (default: ``False``).
               num_outputs : int, optional
                   Number of outputs (default: ``False``).
               num_steps : int, optional
                   Number of time steps (default: ``1``).
               gain: int, optional
                    Scale input features by the gain (default: ``1``).
               offset : float, optional
                    Shift input features by the offset (default: ``0``).
               convert_targets : Bool, optional
                    Convert targets to one-hot-representation if True (default: ``False``).
               temporal_targets : Bool, optional
                    Repeat targets along the time-axis if True (default: ``False``).

                Returns
                -------
                torch.Tensor
                    rate encoding spike train of input features.
                torch.Tensor
                    one-hot encoding of targets with time optionally in the first dimension.
    """

    if convert_targets:
        # One-hot encoding of targets and repeat it along the first dimension, i.e., time first.
        spike_targets = targets_to_spikes(
            targets, num_outputs, num_steps, temporal_targets
        )

    elif not convert_targets and temporal_targets:
        # Repeat tensor in the first dimension without converting targets to one-hot.
        spike_targets = targets.repeat(
            tuple([num_steps] + torch.ones(len(targets.size()), dtype=int).tolist())
        )

    else:
        spike_targets = targets

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

    return spike_data, spike_targets


def latency(
    data,
    targets=False,
    num_outputs=None,
    convert_targets=False,
    temporal_targets=False,
    num_steps=1,
    threshold=0.01,
    epsilon=1e-7,
    tau=1,
    clip=False,
    normalize=False,
    linear=False,
):
    """Latency encoding of input data. Use input features to determine time-to-first spike. Assume a LIF neuron model
    that charges up with time constant tau.
    Optionally convert targets into temporal one-hot spike trains. Tensor dimensions use time first.

               Parameters
               ----------
               data : torch tensor
                   Input of shape (batch, input_size).
               targets : torch tensor, optional
                   Target tensor for a single batch (default: ``False``).
               num_outputs : int, optional
                   Number of outputs (default: ``False``).
               num_steps : int, optional
                   Number of time steps (default: ``1``).
               convert_targets : Bool, optional
                    Convert targets to one-hot-representation if True (default: ``False``).
               temporal_targets : Bool, optional
                    Repeat targets along the time-axis if True (default: ``False``).
               num_steps : int, optional
                    Number of time steps. Only needed if normalizing latency code (default: ``False``).
               threshold : float, optional
                    Value below which features will fire at time tmax (default: ``0.01``).
               epsilon : float, optional
                    A tiny positive value to avoid dividing by zero in firing time calculations (default: ``1e-7``).
               tau : float, optional
                    RC Time constant for LIF model used to calculate firing time (default: ``1``).
               clip : Bool, optional
                    Option to remove spikes from features that fall below the threshold (default: ``False``).
               normalize : Bool, optional
                    Option to normalize the latency code such that the final spike(s) occur within num_steps (default: ``False``).
               linear : Bool, optional
                    Apply a linear latency code rather than the default logarithmic code (default: ``False``).

                Returns
                -------
                torch.Tensor
                    latency encoding spike train of input features.
                torch.Tensor
                    one-hot encoding of targets with time optionally in the first dimension.
    """

    if convert_targets:
        # One-hot encoding of targets and repeat it along the first dimension, i.e., time first.
        spike_targets = targets_to_spikes(
            targets, num_outputs, num_steps, temporal_targets
        )

    elif not convert_targets and temporal_targets:
        # Repeat tensor in the first dimension without converting targets to one-hot.
        spike_targets = targets.repeat(
            tuple([num_steps] + torch.ones(len(targets.size()), dtype=int).tolist())
        )

    else:
        spike_targets = targets

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

    return spike_data, spike_targets


def rate_conv(data):
    """Convert tensor into Poisson spike trains using the features as the mean of a binomial distribution.

    Parameters
    ----------
    data : torch tensor
         Input features e.g., [num_steps x batch size x channels x width x height].

     Returns
     -------
     torch.Tensor
         spike train corresponding to input features.
    """

    # Clip all features between 0 and 1 so they can be used as probabilities.
    clipped_data = torch.clamp(data, min=0, max=1)

    # pass time_data matrix into bernoulli function.
    spike_data = torch.bernoulli(clipped_data)

    return spike_data


def latency_conv(
    data,
    num_steps=False,
    threshold=0,
    epsilon=1e-7,
    tau=1,
    clip=False,
    normalize=False,
    linear=False,
):
    """Latency encoding of input data. Convert input features to spikes that fire according to the latency code.
    Assumes a LIF neuron model that charges up with time constant tau by default.

               Parameters
               ----------
               data : torch tensor
                    Input features e.g., [num_steps x batch size x channels x width x height].
               num_steps : int, optional
                    Number of time steps. Only needed if normalizing latency code (default: ``False``).
               threshold : float, optional
                    Value below which features will fire at time tmax (default: ``0``).
               epsilon : float, optional
                    A tiny positive value to avoid dividing by zero in firing time calculations (default: ``1e-7``).
               tau : float, optional
                    RC Time constant for LIF model used to calculate firing time (default: ``1e-7``).
               clip : Bool, optional
                    Option to remove spikes from features that fall below the threshold (default: ``False``).
               normalize : Bool, optional
                    Option to normalize the latency code such that the final spike(s) occur within num_steps (default: ``False``).
               linear : Bool, optional
                    Apply a linear latency code rather than the default logarithmic code (default: ``False``).


               Returns
               -------
                torch.Tensor
                    latency encoding spike train of input features.
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

    while True:
        try:
            spike_data = spike_data.scatter(
                0, torch.round(spike_time).long().unsqueeze(0), 1
            )
            break
        except RuntimeError:  # this block runs if indexes in spike_time exceed range in num_steps.
            if print_flag:
                print(
                    "Warning: the spikes outside of the range of num_steps have been clipped.\n "
                    "Setting ``normalize`` to ``True`` or increasing ``num_steps`` can fix this."
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

               Parameters
               ----------
               data : torch tensor
                    Input features e.g., [num_steps x batch size x channels x width x height].
               num_steps : int , optional
                    Number of time steps. Only needed if normalizing latency code (default: ``False``).
               threshold : float, optional
                    Value below which features will fire at some saturating time (default: ``0.01``).
               epsilon : float, optional
                    A tiny positive value to avoid dividing by zero in firing time calculations (default: ``1e-7``).
               tau : float, optional
                    RC Time constant for LIF model used to calculate firing time (default: ``1``).
               normalize : Bool, optional
                    Option to normalize the latency code such that the latest spike occurs within num_steps (default: ``False``).
               linear : Bool, optional
                    Apply a linear latency code rather than the default logarithmic code (default: ``False``).

               Returns
                -------
                torch.Tensor
                    Latency encoding spike times of input features.

                torch.Tensor
                    Tensor of Boolean values which correspond to the latency encoding elements that are under the
                    threshold. Used in latency_conv to clip saturated spikes.
    """

    if (
        threshold <= 0 or threshold >= 1
    ):  # double check if this can just be threshold < 0 instead.
        raise Exception("Threshold must be between 0 and 1.")

    if tau <= 0:  # double check if this can just be threshold < 0 instead.
        raise Exception("``tau`` must be greater than 0.")

    if normalize and not num_steps:
        raise Exception("`num_steps` should not be empty if normalize is set to True.")

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
        if normalize:
            tau = num_steps - 1
        spike_time = -tau * (data - 1)
        spike_time = torch.clamp_max(spike_time, (-tau * (threshold - 1)))

    return spike_time, idx


def targets_to_spikes(targets, num_outputs=None, num_steps=1, temporal_targets=False):
    """Convert targets to one-hot encodings in the time-domain.

    Parameters
    ----------
    targets : torch tensor
        Target tensor for a single minibatch.
    num_outputs : int
        Number of outputs (default: ``None``).
    num_steps : int, optional
        Number of time steps (default: ``1``).
    temporal_targets : Bool, optional
        Repeat targets along the time-axis if True (default: ``False``).

    Returns
    -------
    torch.Tensor
        one hot encoding of targets with time in the first dimension.
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

    if temporal_targets:
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

    Parameters
    ----------
    targets : torch tensor
        Target tensor for a single minibatch.
    num_outputs : int
        Number of outputs.

     Returns
     -------
     torch.Tensor
         one hot encoding of targets
    """
    # Initialize zeros. E.g, for MNIST: (batch_size, 10).
    one_hot = torch.zeros([len(targets), num_outputs], device=device)

    # Unsqueeze converts dims of [100] to [100, 1]
    one_hot = one_hot.scatter(1, targets.unsqueeze(-1), 1)

    return one_hot


def from_one_hot(one_hot_label):
    """Convert one-hot encoding back into an integer

    Parameters
       ----------
       one_hot_label : torch tensor
           A single one-hot label vector

        Returns
        -------
        integer
            target.
    """
    one_hot_label = torch.where(one_hot_label == 1)[0][0]
    return int(one_hot_label)

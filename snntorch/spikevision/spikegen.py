import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Explore the use of yield for this?
# Or if it's best to iterate outside the function


def spike_conversion(data, targets=False, num_outputs=None, num_steps=1, gain=1, offset=0, convert_targets=True,
                     temporal_targets=False):
    """Convert tensor into Poisson spike trains using the features as the mean of a binomial distribution.
    Optionally convert targets into temporal one-hot spike trains.

               Parameters
               ----------
               data : torch tensor
                   Input features e.g., [batch size x channels x width x height].
               targets : torch tensor, optional
                   Target tensor for a single minibatch (default: ``False``).
               num_outputs : int, optional
                   Number of outputs (default: ``False``).
               num_steps : int, optional
                   Number of time steps (default: ``1``).
               gain: int, optional
                    Scale input features by the gain (default: ``1``).
               offset : float, optional
                    Shift input features by the offset (default: ``0``).
               convert_targets : Bool, optional
                    Convert targets to one-hot-representation if True (default: ``True``).
               temporal_targets : Bool, optional
                    Repeat targets along the time-axis if True (default: ``False``).

                Returns
                -------
                torch.Tensor
                    spike train corresponding to input features.
                torch.Tensor
                    one-hot encoding of targets with time optionally in the first dimension.
               """

    if convert_targets is True:
        # Perform one-hot encoding of targets and repeat it along the time axis
        spike_targets = targets_to_spikes(targets, num_outputs, num_steps, temporal_targets)

    else:
        spike_targets = targets

    # Generate a tuple: (num_steps, 1, 1..., 1) where the number of 1's = number of dimensions in the original data.
    # Multiply by gain and add offset.
    time_data = data.repeat(tuple([num_steps]+torch.ones(len(data.size()), dtype=int).tolist()))*gain+offset

    spike_data = spike_train(time_data)

    return spike_data, spike_targets


# Use this function in spike_conversion
def spike_train(data):
    """Convert tensor into Poisson spike trains using the features as the mean of a binomial distribution.

               Parameters
               ----------
               data : torch tensor
                    Input features e.g., [batch size x channels x width x height]."""
    # Time x number of neurons
    # time_data = torch.ones([data_config.T, N_in], device=device)*rate

    # Clip all gain between 0 and 1: these are treated as probabilities in the next line.
    clipped_data = torch.clamp(data, min=0, max=1)

    # pass that entire time_data matrix into bernoulli.
    spike_data = torch.bernoulli(clipped_data)

    return spike_data


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
        print("Warning: num_outputs will automatically be calculated using the number of unique values in "
              "targets.\n"
              "It is recommended to explicitly pass num_steps as an argument instead.")
        num_outputs = len(targets.unique())
        print(f"num_outputs has been calculated to be {num_outputs}.")

    targets_1h = to_one_hot(targets, num_outputs)

    if temporal_targets is not False:
        # Extend one-hot targets in time dimension. Create a new axis in the first dimension.
        spike_targets = targets_1h.repeat(tuple([num_steps] + torch.ones(len(targets_1h.size()), dtype=int).tolist()))

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


# def spike_train(N_in, data_config, rate):
#
#     # if rate is a constant then just that number in the bernoulli dist
#     if not hasattr(rate, '__iter__'):
#         spike_in = torch.zeros([N_in])
#
#         # If rate is a scalar, fill all elements of spike_in with the rate
#         spike_in.fill_(rate)
#
#         # Time x N
#         spike_in = spike_in.repeat(data_config.T, 1)
#
#     # if rate defines one neurons behaviour: T x 1
#     elif rate[0] == data_config.T:
#         spike_in = rate.repeat(data_config.T, 1)
#
#     # Rate: T x N
#     else:
#         spike_in = rate
#
#     # Clip all gain between 0 and 1: these are treated as probabilities in the next line.
#     spike_in = torch.clamp(spike_in, min=0, max=1)
#
#     # pass that entire time_data matrix into bernoulli.
#     spike_in = torch.bernoulli(spike_in)
#
#     return spike_in

import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def spike_conversion(data, targets, data_config, gain=1, convert_targets=True):
    """Convert images into Poisson spike trains and their targets into one-hot spike trains. Input pixels are treated as
    the probability for a spike to occur.

               Parameters
               ----------
               data : torch tensor
                   Input image data [batch size x channels x width x height].
               targets : torch tensor
                   Target tensor for a single minibatch.
               data_config : snntorch configuration
                   Configuration class.
               gain: int, optional
                    Factor to multiply the input tensor of pixel values for the Bernoulli distribution generator
                    (default: ``1``).
               convert_targets: Bool, optional
                    Convert targets to time-domain one-hot-representation if True (default: ``True``).
                Returns
                -------
                torch.Tensor
                    spike train corresponding to input pixels.
                torch.Tensor
                    one hot encoding of targets with time in the first dimension.
               """

    # Perform one-hot encoding of targets and repeat it along the time axis
    if convert_targets is True:
        spike_targets = targets_to_spikes(data_config, targets)

    else:
        spike_targets = targets

    # Time x num_batches x channels x width x height
    # This is only set-up for MNIST/single channel data. To-fix.
    time_data = data.repeat(data_config.T, 1, 1, 1, 1)*gain

    # Clip all gain between 0 and 1: these are treated as probabilities in the next line.
    time_data = torch.clamp(time_data, min=0, max=1)

    # pass that entire time_data matrix into bernoulli.
    spike_data = torch.bernoulli(time_data)

    return spike_data, spike_targets


# Use this function in spike_conversion
def spike_train(N_in, data_config, rate):

    # Time x number of neurons
    time_data = torch.ones([data_config.T, N_in], device=device)*rate

    # Clip all gain between 0 and 1: these are treated as probabilities in the next line.
    time_data = torch.clamp(time_data, min=0, max=1)

    # pass that entire time_data matrix into bernoulli.
    spike_data = torch.bernoulli(time_data)

    return spike_data


def targets_to_spikes(data_config, targets):
    """Convert targets to one-hot encodings in the time-domain.

           Parameters
           ----------
           data_config : snntorch configuration
               Configuration class.
           targets : torch tensor
               Target tensor for a single minibatch.

            Returns
            -------
            torch.Tensor
                one hot encoding of targets with time in the first dimension.
           """
    targets_1h = to_one_hot(data_config, targets)

    # Extend one-hot targets in time dimension. Create a new axis in the first dimension.
    # E.g., turn 100x10 into 1000x100x10.
    spike_targets = targets_1h.repeat(data_config.T, 1, 1)
    return spike_targets


def to_one_hot(data_config, targets):
    """One hot encoding of target labels.

       Parameters
       ----------
       data_config : snntorch configuration
           Configuration class.
       targets : torch tensor
           Target tensor for a single minibatch.

        Returns
        -------
        torch.Tensor
            one hot encoding of targets
       """
    # Initialize zeros. E.g, for MNIST: (100,10).
    one_hot = torch.zeros([len(targets), data_config.num_classes], device=device)

    # unsqueeze converts dims of [100] to [100, 1]
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


def spike_train(N_in, data_config, rate):

    # if rate is a constant then just that number in the bernoulli dist
    if not hasattr(rate, '__iter__'):
        spike_in = torch.zeros([N_in])

        # If rate is a scalar, fill all elements of spike_in with the rate
        spike_in.fill_(rate)

        # Time x N
        spike_in = spike_in.repeat(data_config.T, 1)

    # if rate defines one neurons behaviour: T x 1
    elif rate[0] == data_config.T:
        spike_in = rate.repeat(data_config.T, 1)

    # Rate: T x N
    else:
        spike_in = rate

    # Clip all gain between 0 and 1: these are treated as probabilities in the next line.
    spike_in = torch.clamp(spike_in, min=0, max=1)

    # pass that entire time_data matrix into bernoulli.
    spike_in = torch.bernoulli(spike_in)

    return spike_in

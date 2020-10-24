import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def spike_conversion(data, targets, data_config, gain=1):
    # Perform one-hot encoding of targets and repeat it along the time axis
    spike_targets = targets_to_spikes(data_config, targets)

    # Time x num_batches x channels x width x height
    time_data = data.repeat(data_config.T, 1, 1, 1, 1)*gain

    # Clip all gain between 0 and 1: these are treated as probabilities in the next line.
    time_data = torch.clamp(time_data, min=0, max=1)

    # pass that entire time_data matrix into bernoulli.
    spike_data = torch.bernoulli(time_data)

    return spike_data, spike_targets


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

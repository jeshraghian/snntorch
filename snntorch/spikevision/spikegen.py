import torch
import numpy as np

# x is raw_input and y is raw_target from iterator
# i should change image2spiketrain to "spike_generator" or "input2spike" or "spikegen" or "convert2spike"
# consider implementing a 'gain' function.
def spike_conversion(data, targets, data_config, gain=1):
    # Perform one-hot encoding of targets and repeat it along the time axis
    spike_targets = targets_to_spikes(data_config, targets)

    # From my previous implementation. Let's do this on CUDA if avail?

    # repeat data input along time axis.
    time_data = np.repeat(data[np.newaxis, :, :], data_config.T, axis=0)*gain

    # Clip all gain between 0 and 1.
    time_data = torch.clamp(time_data, min=0, max=1)

    # pass that entire time_data matrix into bernoulli. do we need to keep FloatTensor?
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
    targets_1h = one_hot(data_config, targets)

    # Extend one-hot targets in time dimension. Create a new axis in the first position.
    # E.g., turn 100x10 into 1000x100x10.
    spike_targets = np.repeat(targets_1h[np.newaxis, :, :], data_config.T, axis=0)

    return spike_targets


def one_hot(data_config, targets):
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
    one_hot = torch.zeros([len(targets), data_config.num_classes])

    #unsqueeze converts dims of [100] to [100, 1]
    one_hot.scatter_(1, targets.unsqueeze(-1), 1)

    return one_hot

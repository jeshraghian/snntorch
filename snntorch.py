import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, models, transforms, utils
import torch
from torch.utils.data import random_split
import sys
import math

# dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Configuration:
    """" A class for holding data about the model and dataset."""
    # A class for holding data about the model and dataset.
    def __init__(self, input_size, channels=1, batch_size=100, split=0, subset=1, num_classes=False, T=500, data_path='/data'):
        """"
        :param input_size (list): input data dimensions [W x H] **extend this out to C x W x H **or** T x C x W x H.
        :param channels (int): number of channels (default: ``1``).
        :param batch_size (int, optional): batch size (default: ``100``).
        :param split (int, optional): size of validation set (default: ``0``).
        :param subset (int, optional): factor by which to divide size of train and test sets (default: ``1``).
        :param num_classes (int, optional): number of output classes. Automatically calculated if not provided
        (default: ``False``)
        :param T (int, optional): number of time steps (default: ``500``).
        :param data_path (string, optional): pathway to download dataset to (default: ``/data``).
        """
        self.input_size = input_size
        self.channels = channels
        self.batch_size = batch_size
        self.split = split
        self.subset = subset
        self.num_classes = num_classes
        self.T = T
        self.data_path = data_path

    #def _calc_input_size(self, ds_train):
    #     """Measure input dimension size and update config file."""
    #    self.input_size = list(np.array(ds_train[0][0]).shape)

    def _dataset_size(self, ds_valid, ds_test):
        """Measure validation and test set sizes and update config file. Called automatically from mnist_dataset()."""
        self.valid_size, self.test_size = ds_valid.__len__(), ds_test.__len__()

    def _calc_num_classes(self, ds_test, target_dim=-1):
        """Calculate number of output classes and update config file. Only invoked if num_classes is unspecified."""
        ds_targets = []
        for n in range(len(ds_test)):
            ds_targets.append(ds_test[n][target_dim])
        self.num_classes = len(set(ds_targets))


def spikes_to_evlist(spikes):
    t = np.tile(np.arange(spikes.shape[0]), [spikes.shape[1], 1])
    n = np.tile(np.arange(spikes.shape[1]), [spikes.shape[0], 1]).T
    return t[spikes.astype('bool').T], n[spikes.astype('bool').T]


def plotLIF(U, S, Vplot='all', staggering=1, ax1=None, ax2=None, **kwargs):
    '''
    This function plots the output of the function LIF.

    Inputs:
    *S*: an TxNnp.array, where T are time steps and N are the number of neurons
    *S*: an TxNnp.array of zeros and ones indicating spikes. This is the second
    output return by function LIF
    *Vplot*: A list indicating which neurons' membrane potentials should be
    plotted. If scalar, the list range(Vplot) are plotted. Default: 'all'
    *staggering*: the amount by which each V trace should be shifted. None

    Outputs the figure returned by figure().
    '''
    V = U
    spikes = S
    # Plot
    t, n = spikes_to_evlist(spikes)
    # f = plt.figure()
    if V is not None and ax1 is None:
        ax1 = plt.subplot(211)
    elif V is None:
        ax1 = plt.axes()
        ax2 = None
    ax1.plot(t, n, 'k|', **kwargs)
    ax1.set_ylim([-1, spikes.shape[1] + 1])
    ax1.set_xlim([0, spikes.shape[0]])

    if V is not None:
        if Vplot == 'all':
            Vplot = range(V.shape[1])
        elif not hasattr(Vplot, '__iter__'):
            Vplot = range(np.minimum(Vplot, V.shape[1]))

        if ax2 is None:
            ax2 = plt.subplot(212)

        if V.shape[1] > 1:
            for i, idx in enumerate(Vplot):
                ax2.plot(V[:, idx] + i * staggering, '-', **kwargs)
        else:
            ax2.plot(V[:, 0], '-', **kwargs)

        if staggering != 0:
            plt.yticks([])
        plt.xlabel('time [ms]')
        plt.ylabel('u [au]')

    ax1.set_ylabel('Neuron ')

    plt.xlim([0, spikes.shape[0]])
    plt.ion()
    plt.show()
    return ax1, ax2


def mnist_dataset(data_config, download=True, index=0, transform=0):
    """Download and initialize training, validation and test sets using torchvision.datasets.MNIST.
    :param data_config: instance of the Configuration class containing information about the dataset.
    :param download (bool, optional): If true, downloads the dataset from the internet. If dataset is already
    downloaded, it is not downloaded again (default: ``True``).
    :param index (int, optional): If data_config.subset>1, specifies which subset of the full dataset to use
    (``default: 0``).

    :return ds_train, ds_valid, ds_test."""
    # Download and initialize training, validation and test sets

    if transform == 0:
        transform = transforms.Compose([
            transforms.Resize(data_config.input_size),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.0,), (1.0,))])

    dataset = torchvision.datasets.MNIST(root=data_config.data_path, download=download, train=True, transform=transform)
    ds_test = torchvision.datasets.MNIST(root=data_config.data_path, download=download, train=False, transform=transform)

    ds_train, ds_valid = mnist_split(dataset, data_config)
    ds_train, ds_valid, ds_test = mnist_subset(ds_train, ds_valid, ds_test, data_config, index)

    # Update input_size, dataset sizes, and num_classes where not provided
    _config_update(data_config, ds_train, ds_valid, ds_test)

    return ds_train, ds_valid, ds_test


def _config_update(data_config, ds_train, ds_valid, ds_test):
    # Update config file with input_size if not provided.
    #if data_config.input_size is False:
    #    data_config._calc_image_size(ds_train)

    # Update config file with validation and test set sizes -- makes spike train conversion easier.
    data_config._dataset_size(ds_valid, ds_test)

    # Update config file with num_classes if num_classes has not been provided
    if data_config.num_classes is False:
        data_config._calc_num_classes(ds_test)


def mnist_split(dataset, data_config):
    # Random split train/validation sets

    ds_train, ds_valid = random_split(dataset, [(len(dataset) - data_config.split), data_config.split])
    return ds_train, ds_valid


def mnist_subset(ds_train, ds_valid, ds_test, data_config, index=0):
    # Take subsample of training and test sets

    train_sampler = list(range(index, (index + 1) * (len(ds_train)) // data_config.subset))
    test_sampler = list(range(index, (index + 1) * (len(ds_test)) // data_config.subset))

    ds_train = torch.utils.data.Subset(ds_train, train_sampler)
    ds_test = torch.utils.data.Subset(ds_test, test_sampler)

    return ds_train, ds_valid, ds_test


def time_series_data(dataset, data_config):

    # initialize data and targets [N x W x H] ---- we can consider expanding for channel size later
    # uint8 because when we broadcast takes up loads of memory. dtype = 'uint8'
    data = np.zeros([len(dataset), data_config.channels, *data_config.input_size])
    targets = np.zeros([len(dataset), 1])

    # Combine dataset into one huge tensor so spike train can be passed into Bernoulli function all in one hit.
    print("\nMerging dataset. This may take a while: ...")
    for idx in range(len(dataset)):
        data[idx] = dataset[idx][0]
        targets[idx] = dataset[idx][1]
        _update_progress(idx / len(dataset))
    print("\nMerging complete.")

    # repeat array T times in dim=0 for multiple timesteps. TO-DO: give option to load this on CUDA.
    print("\nBroadcasting data along the time axis. This may take a while...")
    data_time = np.repeat(data[np.newaxis, :, :], data_config.T, axis=0)
    print("Broadcasting complete.")

    return data_time, targets


def time_series_targets(targets, data_config):

    # targets needs to be (500x1). squeeze makes it (500). Unsqueeze makes it (500x1x1).
    # one_hot targets.
    targets = (torch.from_numpy(targets)).type(torch.int64)
    one_hot = torch.zeros([len(targets), (data_config.num_classes)])
    targets_one_hot = one_hot.scatter_(1, targets, 1)

    # extend targets_one_hot in the T direction
    st_targets = np.repeat(targets_one_hot[np.newaxis, :, :], data_config.T, axis=0)

    return st_targets


def spiketrain(data_time):
    # spike generation function. data_np_repeated forms the probabilities of the function.
    # I've removed the 'regular' mode of periodic firing. Only option provided is Poisson.
    # needs to be of type FloatTensor for training.
    print("\nGenerating Poisson spike train. This may take a while...")
    st_data = torch.bernoulli(torch.from_numpy(data_time)).type(torch.FloatTensor)
    print("Spike train generation complete.")
    return st_data


def target_convolve(tgt, alpha=8, alphas=5):
    max_duration = tgt.shape[0]
    kernel_alpha = np.exp(-np.linspace(0, 10 * alpha, dtype='float') / alpha)
    kernel_alpha /= kernel_alpha.sum()
    kernel_alphas = np.exp(-np.linspace(0, 10 * alphas, dtype='float') / alphas)
    kernel_alphas /= kernel_alphas.sum()
    tgt = tgt.copy()
    for i in range(tgt.shape[1]):
        for j in range(tgt.shape[2]):
            tmp = np.convolve(np.convolve(tgt[:, i, j], kernel_alpha), kernel_alphas)[:max_duration]
            tgt[:, i, j] = tmp
    return tgt / tgt.max()


def sequester(tensor):
    dtype = tensor.dtype
    return torch.tensor(tensor.detach().cpu().numpy(), dtype=dtype)


def pixel_permutation(d_size, r_pix=1.0, seed=0):
    import copy
    n_pix = int(r_pix * d_size)
    np.random.seed(seed * 1313)
    pix_sel = np.random.choice(d_size, n_pix, replace=False).astype(np.int32)
    pix_prm = np.copy(pix_sel)
    np.random.shuffle(pix_prm)
    perm_inds = np.arange(d_size)
    perm_inds[pix_sel] = perm_inds[pix_prm]
    return perm_inds


def permute_dataset(dataset, r_pix, seed):
    shape = dataset.data.shape[1:]
    datap = dataset.data.view(-1, np.prod(shape)).detach().numpy()
    perm = pixel_permutation(np.prod(shape), r_pix, seed=seed)
    return torch.FloatTensor(datap[:, perm].reshape(-1, *shape))


# update_progress() : Displays or updates a console progress bar
## Accepts a float between 0 and 1. Any int will be converted to a float.
## A value under 0 represents a 'halt'.
## A value at 1 or bigger represents 100%
def _update_progress(progress):
    barLength = 15 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), math.ceil(progress*100), status)
    sys.stdout.write(text)
    sys.stdout.flush()
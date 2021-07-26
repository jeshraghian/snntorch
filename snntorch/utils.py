# Note: need NumPy 1.17 or later for RNG functions
import numpy as np
import snntorch as snn


def data_subset(dataset, subset, idx=0):
    """Partition the dataset by a factor of ``1/subset`` without removing access to data and target attributes.

    Example::

        from snntorch import utils
        from torchvision import datasets

        data_path = "path/to/data"
        subset = 10

        #  Download MNIST training set
        mnist_train = datasets.MNIST(data_path, train=True, download=True)
        print(len(mnist_train))
        >>> 60000

        #  Reduce size of MNIST training set
        utils.data_subset(mnist_train, subset)
        print(len(mnist_train))
        >>> 6000

    :param dataset: Dataset
    :type dataset: torchvision dataset

    :param subset: Factor to reduce dataset by
    :type subset: int

    :param idx: Which subset of the train and test sets to index into, defaults to ``0``
    :type idx: int, optional

    :return: Partitioned dataset
    :rtype: list of torch.utils.data
    """

    if subset > 1:
        N = len(dataset.data)

        idx_range = np.arange(N, dtype="int")
        step = N // subset
        idx_range = idx_range[step * idx : step * (idx + 1)]

        data = dataset.data[idx_range]
        targets = dataset.targets[idx_range]

        dataset.data = data
        dataset.targets = targets

    return dataset


def valid_split(ds_train, ds_val, split, seed=0):
    """Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results. Operates similarly to ``random_split`` from
    ``torch.utils.data.dataset`` but retains data and target attributes.

    Example ::

        from snntorch import utils
        from torchvision import datasets

        data_path = "path/to/data"
        val_split = 0.1

        #  Download MNIST training set into mnist_val and mnist_train
        mnist_train = datasets.MNIST(data_path, train=True, download=True)
        mnist_val = datasets.MNIST(data_path, train=True, download=True)

        print(len(mnist_train))
        >>> 60000

        print(len(mnist_val))
        >>> 60000

        #  Validation split
        mnist_train, mnist_val = utils.valid_split(mnist_train, mnist_val, val_split)

        print(len(mnist_train))
        >>> 54000

        print(len(mnist_val))
        >>> 6000

    :param ds_train: Training set
    :type ds_train: torchvision dataset

    :param ds_val: Validation set
    :type ds_val: torchvision dataset

    :param split: Proportion of samples assigned to the validation set from the training set
    :type split: Float

    :param seed: Fix to generate reproducible results, defaults to ``0``
    :type seed: int, optional

    :return: Randomly split train and validation sets
    :rtype: list of torch.utils.data
    """

    n = len(ds_train)
    n_val = int(n * split)
    n_train = n - n_val

    # Create an index list of length n_train, containing non-repeating values from 0 to n-1
    rng = np.random.default_rng(seed=seed)
    train_idx = rng.choice(n, size=n_train, replace=False)

    # create inverted index for validation from train
    val_idx = []
    for i in range(n):
        if i not in train_idx:
            val_idx.append(i)

    # Generate ds_val by indexing into ds_train
    vd = ds_train.data[val_idx]
    vt = ds_train.targets[val_idx]
    ds_val.data = vd
    ds_val.targets = vt

    # Recreate ds_train by indexing into the previous ds_train
    td = ds_train.data[train_idx]
    tt = ds_train.targets[train_idx]
    ds_train.data = td
    ds_train.targets = tt

    return ds_train, ds_val


def reset(net):
    """Check for the types of LIF neurons contained in net.
    Reset their hidden parameters to zero and detach them
    from the current computation graph."""

    global is_stein
    global is_alpha
    global is_leaky
    global is_lapicque
    global is_synaptic

    is_stein = False
    is_alpha = False
    is_leaky = False
    is_synaptic = False
    is_lapicque = False

    _layer_check(net=net)

    _layer_reset()


def _layer_check(net):
    """Check for the types of LIF neurons contained in net."""

    global is_leaky
    global is_lapicque
    global is_synaptic
    global is_stein
    global is_alpha

    for idx in range(len(list(net._modules.values()))):
        if isinstance(list(net._modules.values())[idx], snn.Lapicque):
            is_lapicque = True
        if isinstance(list(net._modules.values())[idx], snn.Synaptic):
            is_synaptic = True
        if isinstance(list(net._modules.values())[idx], snn.Leaky):
            is_leaky = True
        if isinstance(list(net._modules.values())[idx], snn.Stein):
            is_stein = True
        if isinstance(list(net._modules.values())[idx], snn.Alpha):
            is_alpha = True


def _layer_reset():
    """Reset hidden parameters to zero and detach them from the current computation graph."""

    if is_lapicque:
        snn.Lapicque.reset_hidden()  # reset hidden state to 0's
        snn.Lapicque.detach_hidden()
    if is_synaptic:
        snn.Synaptic.reset_hidden()  # reset hidden state to 0's
        snn.Synaptic.detach_hidden()
    if is_leaky:
        snn.Leaky.reset_hidden()  # reset hidden state to 0's
        snn.Leaky.detach_hidden()
    if is_stein:
        snn.Stein.reset_hidden()  # reset hidden state to 0's
        snn.Stein.detach_hidden()
    if is_alpha:
        snn.Alpha.reset_hidden()  # reset hidden state to 0's
        snn.Alpha.detach_hidden()


def _final_layer_check(net):
    """Check class of final layer and return the number of outputs."""

    if isinstance(list(net._modules.values())[-1], snn.Lapicque):
        return 2
    if isinstance(list(net._modules.values())[-1], snn.Synaptic):
        return 3
    if isinstance(list(net._modules.values())[-1], snn.Leaky):
        return 2
    if isinstance(list(net._modules.values())[-1], snn.Stein):
        return 3
    if isinstance(list(net._modules.values())[-1], snn.Alpha):
        return 4
    else:  # if not from snn, assume from nn with 1 return
        return 1

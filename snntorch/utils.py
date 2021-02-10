# Note: need NumPy 1.17 or later for RNG functions
import numpy as np


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

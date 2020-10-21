import torch
import numpy as np


def data_subset(dataset, data_config, idx=0):
    """Partition the dataset by a factor of 1/subset without removing access to data and target attributes.

       Parameters
       ----------
       dataset : torchvision dataset
           Dataset.
       data_config : snntorch configuration
           Configuration class.
       idx : int, optional
           Which subset of the train and test sets to index into (default: ``0``).

        Returns
        -------
        list of torch.utils.data
            Partitioned dataset.
       """
    if data_config.subset > 1:
        N = len(dataset.data)

        idx_range = np.arange(N, dtype='int')
        step = (N // data_config.subset)
        idx_range = idx_range[step * idx:step * (idx + 1)]

        data = dataset.data[idx_range]
        targets = dataset.targets[idx_range]

        dataset.data = data
        dataset.targets = targets

    return dataset

def valid_split(ds_train, ds_val, data_config, seed=0):
    """Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results. Operates similarly to random_split from
    torch.utils.data.dataset but retains data and target attributes.

           Parameters
           ----------
           ds_train : torchvision dataset
               Training set.
           ds_val : torchvision dataset
               Validation set.
           data_config : snntorch configuration
               Configuration class.
           seed : int, optional
               Fix to generate reproducible results (default: ``0``).

            Returns
            -------
            list of torch.utils.data
                Randomly split train and validation sets.
           """
    n = len(ds_train)
    n_val = int(n * data_config.split)
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
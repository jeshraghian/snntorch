# Spiking Heidelberg Digits (SHD) citation: Cramer, B., Stradmann, Y., Schemmel, J., and Zenke, F. (2020). The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks. IEEE Transactions on Neural Networks and Learning Systems 1–14.
# Dataloader adapted from https://github.com/nmi-lab/torchneuromorphic by Emre Neftci


import struct
import time
import numpy as np
import scipy.misc
import h5py
import torch.utils.data
from ..neuromorphic_dataset import NeuromorphicDataset
from ..events_timeslices import *
from .._transforms import *
import os

mapping = {
    0: "E0",
    1: "E1",
    2: "E2",
    3: "E3",
    4: "E4",
    5: "E5",
    6: "E6",
    7: "E7",
    8: "E8",
    9: "E9",
    10: "G0",
    11: "G1",
    12: "G2",
    13: "G3",
    14: "G4",
    15: "G5",
    16: "G6",
    17: "G7",
    18: "G8",
    19: "G9",
}


def one_hot1d(mbt, num_classes):
    out = np.zeros([num_classes], dtype="float32")
    out[int(mbt)] = 1
    return out


def create_events_hdf5(directory, hdf5_filename):
    train_evs, train_labels_isolated = load_shd_hdf5(directory + "/shd_train.h5")
    test_evs, test_labels_isolated = load_shd_hdf5(directory + "/shd_test.h5")
    border = len(train_labels_isolated)

    tmad = train_evs + test_evs
    labels = train_labels_isolated + test_labels_isolated
    test_keys = []
    train_keys = []

    with h5py.File(hdf5_filename, "w") as f:
        f.clear()
        key = 0
        metas = []
        data_grp = f.create_group("data")
        extra_grp = f.create_group("extra")
        print("Creating shd.hdf5...")
        for i, data in enumerate(tmad):
            times = data[:, 0]
            addrs = data[:, 1:]
            label = labels[i]
            out = []
            istrain = i < border
            if istrain:
                train_keys.append(key)
            else:
                test_keys.append(key)
            metas.append({"key": str(key), "training sample": istrain})
            subgrp = data_grp.create_group(str(key))
            tm_dset = subgrp.create_dataset("times", data=times, dtype=np.uint32)
            ad_dset = subgrp.create_dataset("addrs", data=addrs, dtype=np.uint16)
            lbl_dset = subgrp.create_dataset("labels", data=label, dtype=np.uint8)
            subgrp.attrs["meta_info"] = str(metas[-1])
            assert label in mapping
            key += 1
        extra_grp.create_dataset("train_keys", data=train_keys)
        extra_grp.create_dataset("test_keys", data=test_keys)
        extra_grp.attrs["N"] = len(train_keys) + len(test_keys)
        extra_grp.attrs["Ntrain"] = len(train_keys)
        extra_grp.attrs["Ntest"] = len(test_keys)
        print("shd.hdf5 was created successfully.")


def load_shd_hdf5(filename, train=True):
    with h5py.File(filename, "r", swmr=True, libver="latest") as f:
        evs = []
        labels = []
        for i, tl in enumerate(f["labels"]):
            label_ = tl
            digit = tl
            digit = int(digit)
            labels.append(digit)
            tm = np.int32(f["spikes"]["times"][i][:] * 1e6)
            ad = np.int32(f["spikes"]["units"][i][:])
            evs.append(np.column_stack([tm, ad]))
        return evs, labels


class SHD(NeuromorphicDataset):

    """`Spiking Heidelberg Digits <https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/>`_ Dataset.

    Spikes in 700 input channels were generated using an artificial cochlea model listening to studio recordings of spoken digits from 0 to 9 in both German and English languages.

    **Number of classes:** 20

    **Number of train samples:** 8156

    **Number of test samples:** 2264

    **Dimensions:** ``[num_steps x 700]``

    * **num_steps:** time-dimension of audio channels
    * **700:** number of channels in cochlea model

    For further reading, see:

        *Cramer, B., Stradmann, Y., Schemmel, J., and Zenke, F. (2020). The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks. IEEE Transactions on Neural Networks and Learning Systems 1–14.*




    Example::

        from snntorch.spikevision import spikedata

        train_ds = spikedata.SHD("data/shd", train=True)
        test_ds = spikedata.SHD("data/shd", train=False)

        # by default, each time step is integrated over 1ms, or dt=1000 microseconds
        # dt can be changed to integrate events over a varying number of time steps
        # Note that num_steps should be scaled inversely by the same factor

        train_ds = spikedata.SHD("data/shd", train=True, num_steps=500, dt=2000)
        test_ds = spikedata.SHD("data/shd", train=False, num_steps=500, dt=2000)

    The dataset can also be manually downloaded, extracted and placed into ``root`` which will allow the dataloader to bypass straight to the generation of a hdf5 file.

    **Direct Download Links:**

        `CompNeuro Train Set Link <https://compneuro.net/datasets/shd_train.h5.gz>`_

        `CompNeuro Test Set Link <https://compneuro.net/datasets/shd_test.h5.gz>`_

    :param root: Root directory of dataset.
    :type root: string

    :param train: If True, creates dataset from training set of dvsgesture, otherwise test set.
    :type train: bool, optional

    :param transform: A function/transform that takes in a PIL image and returns a transforms version. By default, a pre-defined set of transforms are applied to all samples to convert them into a time-first tensor with correct orientation.
    :type transform: callable, optional

    :param target_transform: A function/transform that takes in the target and transforms it.
    :type target_transform: callable, optional

    :param download_and_create: If True, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
    :type download_and_create: bool, optional

    :param num_steps: Number of time steps, defaults to ``1000``
    :type num_steps: int, optional

    :param dt: The number of time stamps integrated in microseconds, defaults to ``1000``
    :type dt: int, optional

    :param ds: Rescaling factor, defaults to ``1``.
    :type ds: int, optional


    Dataloader adapted from `torchneuromorphic <https://github.com/nmi-lab/torchneuromorphic>`_ originally by Emre Neftci.

    The dataset is released under a Creative Commons Attribution 4.0 International License. All rights remain with the original authors.

    """

    _resources_url = [
        ["https://compneuro.net/datasets/shd_test.h5.gz", None, "shd_test.h5.gz"],
        ["https://compneuro.net/datasets/shd_train.h5.gz", None, "shd_train.h5.gz"],
    ]

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download_and_create=True,
        num_steps=1000,
        ds=1,
        dt=1000,
    ):

        self.n = 0
        self.root = root
        self.download_and_create = download_and_create
        self.train = train
        self.num_steps = num_steps
        self.dt = dt
        self.ds = ds
        self.resources_local = [
            self.root + "/shd_train.hdf5",
            self.root + "/shd_test.hdf5",
        ]
        self.directory = self.root

        size = [700 // self.ds]

        if transform is None:
            transform = Compose(
                [
                    Downsample(factor=[self.dt, self.ds]),
                    ToChannelHeightWidth(),
                    ToCountFrame(T=self.num_steps, size=size),
                    ToTensor(),
                    hflip(),
                ]
            )

        if target_transform is not None:
            target_transform = Compose([Repeat(num_steps), toOneHot(len(mapping))])

        super(SHD, self).__init__(
            root=root + "/shd.hdf5",
            transform=transform,
            target_transform=target_transform,
        )

        with h5py.File(root + "/shd.hdf5", "r", swmr=True, libver="latest") as f:
            if train:
                self.n = f["extra"].attrs["Ntrain"]
                self.keys = f["extra"]["train_keys"]
            else:
                self.n = f["extra"].attrs["Ntest"]
                self.keys = f["extra"]["test_keys"]

    def _download(self):
        isexisting = super(SHD, self)._download()

    def _create_hdf5(self):
        create_events_hdf5(self.directory, self.root)

    def __len__(self):
        return self.n

    def __getitem__(self, key):
        # Important to open and close in getitem to enable num_workers>0
        with h5py.File(self.root, "r", swmr=True, libver="latest") as f:
            if not self.train:
                key = key + f["extra"].attrs["Ntrain"]
            data, target = sample(f, key, T=self.num_steps, shuffle=self.train)

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target


def sample(hdf5_file, key, T=500, shuffle=False):
    dset = hdf5_file["data"][str(key)]
    label = dset["labels"][()]
    tend = dset["times"][-1]
    start_time = 0

    tmad = get_tmad_slice(dset["times"][()], dset["addrs"][()], start_time, T * 1000)
    tmad[:, 0] -= tmad[0, 0]
    return tmad, label

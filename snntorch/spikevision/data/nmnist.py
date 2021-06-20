# Adapted from https://github.com/nmi-lab/torchneuromorphic by Emre Neftci and Clemens Schaefer

import struct
import time, copy
import numpy as np
import scipy.misc
import h5py
import torch.utils.data
from ..neuromorphic_dataset import NeuromorphicDataset 
from ..events_timeslices import *
from .._transforms import *
import os
from .._utils import load_ATIS_bin
from tqdm import tqdm
import glob

mapping = { 0 :'0',
            1 :'1',
            2 :'2',
            3 :'3',
            4 :'4',
            5 :'5',
            6 :'6',
            7 :'7',
            8 :'8',
            9 :'9'}

class NMNIST(NeuromorphicDataset):

    """`NMNIST <https://www.garrickorchard.com/datasets/n-mnist>`_ Dataset.
   
    Example::

        from snntorch import spikevision.data

        train_ds = spikevision.data.NMNIST("data/nmnist", train=True, num_steps=300)
        test_ds = spikevision.data.NMNIST("data/nmnist", train=False, num_steps=300)
        

    Args:
        :param root: Root directory of dataset where ``Train.zip`` and  ``Test.zip`` exist.
        :type root: string

        :param train: If True, creates dataset from ``Train.zip``, otherwise from ``Test.zip``
        :type train: bool, optional

        :param transform: A function/transform that takes in a PIL image and returns a transforms version. By default, a pre-defined set of transforms are applied to NMNIST to convert them into a time-first tensor.
        :type transform: callable, optional

        :param target_transform: A function/transform that takes in the target and transforms it.
        :type target_transform: callable, optional

        :param download_and_create: If True, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
        :type download_and_create: bool, optional

        :param num_steps: Number of time steps, defaults to ``300``
        :type num_steps: int, optional

        :param dt: Duration of each time step in microseconds, defaults to ``1000``
        :type dt: int, optional
    
    Adapted from `torchneuromorphic <https://github.com/nmi-lab/torchneuromorphic>`_ originally by Emre Neftci and Clemens Schaefer.
        
    """


    resources_url = [['https://www.dropbox.com/sh/tg2ljlbmtzygrag/AABlMOuR15ugeOxMCX0Pvoxga/Train.zip?dl=1', None, 'Train.zip'],
                     ['https://www.dropbox.com/sh/tg2ljlbmtzygrag/AADSKgJ2CjaBWh75HnTNZyhca/Test.zip?dl=1', None, 'Test.zip']]

    def __init__(
            self, 
            root,
            train=True,
            transform=None,
            target_transform=None,
            download_and_create=True,
            num_steps = 300,
            dt = 1000):

        self.n = 0
        self.nclasses = self.num_classes = 10
        self.download_and_create = download_and_create
        self.train = train 
        self.dt = dt
        self.num_steps = num_steps
        self.directory = root.split('n_mnist.hdf5')[0]
        self.resources_local = [self.directory + '/Train.zip', self.directory + '/Test.zip']  # debug: to-do: check if forward-slash holds up in colab
        
        size = [2, 32, 32]  # 32//ds

        if transform is None:
            transform = Compose([
                CropDims(low_crop=[0, 0], high_crop=[32, 32], dims=[2, 3]),
                Downsample(factor=[dt, 1, 1, 1]),
                ToCountFrame(T = num_steps, size = size),
                ToTensor(), 
                nmnist_permute()])

        if target_transform is not None:
            target_transform = Compose([Repeat(num_steps), toOneHot(10)])
        
        super(NMNIST, self).__init__(
                root = root + "/n_mnist.hdf5",
                transform = transform,
                target_transform = target_transform)

        with h5py.File(self.root, 'r', swmr=True, libver="latest") as f: 
            try:
                if train:
                    self.n = f['extra'].attrs['Ntrain']
                    self.keys = f['extra']['train_keys'][()]
                    self.keys_by_label = f['extra']['train_keys_by_label'][()]
                else:
                    self.n = f['extra'].attrs['Ntest']
                    self.keys = f['extra']['test_keys'][()]
                    self.keys_by_label = f['extra']['test_keys_by_label'][()]
                    self.keys_by_label[:,:] -= self.keys_by_label[0,0]  # normalize
            except (AttributeError, KeyError) as e:
                file_name = "/n_mnist.hdf5"
                print(f"Attribute not found in hdf5 file. You may be using an old hdf5 build. Delete {root + file_name} and run again.")
                raise


    def download(self):
        isexisting = super(NMNIST, self).download()

    def create_hdf5(self):
        create_events_hdf5(self.directory, self.root)


    def __len__(self):
        return self.n
        
    def __getitem__(self, key):
        #Important to open and close in getitem to enable num_workers>0
        with h5py.File(self.root, 'r', swmr=True, libver="latest") as f:
            if self.train:
                key = f['extra']['train_keys'][key]
            else:
                key = f['extra']['test_keys'][key]
            data, target = sample(
                    f,
                    key,
                    T = self.num_steps*self.dt)

        if self.transform is not None:
            data = self.transform(data)

        target = self.target_transform(target)

        return data, target


def create_events_hdf5(directory, hdf5_filename):
    fns_train, fns_test = nmnist_get_file_names(directory)
    fns_train = [val for sublist in fns_train for val in sublist]
    fns_test = [val for sublist in fns_test for val in sublist]
    test_keys = []
    train_keys = []
    train_label_list = [[] for i in range(10)]
    test_label_list = [[] for i in range(10)]
    

    with h5py.File(hdf5_filename, 'w') as f:
        f.clear()
        key = 0
        metas = []
        data_grp = f.create_group('data')
        extra_grp = f.create_group('extra')
        print(f"Creating n_mnist.hdf5...")
        for file_d in tqdm(fns_train+fns_test):
            istrain = file_d in fns_train
            data = nmnist_load_events_from_bin(file_d)
            times = data[:,0]
            addrs = data[:,1:]
            label = int(file_d.replace('\\', '/').split("/")[-2]) # \\ for binder/colab
            out = []

            if istrain: 
                train_keys.append(key)
                train_label_list[label].append(key)
            else:
                test_keys.append(key)
                test_label_list[label].append(key)
            metas.append({'key':str(key), 'training sample':istrain}) 
            subgrp = data_grp.create_group(str(key))
            tm_dset = subgrp.create_dataset('times' , data=times, dtype = np.uint32)
            ad_dset = subgrp.create_dataset('addrs' , data=addrs, dtype = np.uint8)
            lbl_dset= subgrp.create_dataset('labels', data=label, dtype = np.uint8)
            subgrp.attrs['meta_info']= str(metas[-1])
            assert label in range(10)
            key += 1
        extra_grp.create_dataset('train_keys', data = train_keys)
        extra_grp.create_dataset('train_keys_by_label', data = train_label_list)
        extra_grp.create_dataset('test_keys_by_label', data = test_label_list)
        extra_grp.create_dataset('test_keys', data = test_keys)
        extra_grp.attrs['N'] = len(train_keys) + len(test_keys)
        extra_grp.attrs['Ntrain'] = len(train_keys)
        extra_grp.attrs['Ntest'] = len(test_keys)
        print(f"n_mnist.hdf5 was created successfully.")


def nmnist_load_events_from_bin(file_path, max_duration=None):
    timestamps, xaddr, yaddr, pol = load_ATIS_bin(file_path)
    return np.column_stack([
        np.array(timestamps, dtype=np.uint32),
        np.array(pol, dtype=np.uint8),
        np.array(xaddr, dtype=np.uint16),
        np.array(yaddr, dtype=np.uint16)])


def nmnist_get_file_names(dataset_path):
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError("N-MNIST Dataset not found, looked at: {}".format(dataset_path))

    train_files = []
    test_files = []
    for digit in range(10):
        digit_train = glob.glob(os.path.join(dataset_path, 'Train/{}/*.bin'.format(digit)))
        digit_test = glob.glob(os.path.join(dataset_path, 'Test/{}/*.bin'.format(digit)))
        train_files.append(digit_train)
        test_files.append(digit_test)

    # We need the same number of train and test samples for each digit, let's compute the minimum
    max_n_train = min(map(lambda l: len(l), train_files))
    max_n_test = min(map(lambda l: len(l), test_files))
    n_train = max_n_train # we could take max_n_train, but my memory on the shared drive is full
    n_test = max_n_test # we test on the whole test set - lets only take 100*10 samples
    assert((n_train <= max_n_train) and (n_test <= max_n_test)), 'Requested more samples than present in dataset'

    print(f"\nN-MNIST: {n_train*10} train samples and {n_test*10} test samples")
    # Crop extra samples of each digits
    train_files = map(lambda l: l[:n_train], train_files)
    test_files = map(lambda l: l[:n_test], test_files)

    return list(train_files), list(test_files)


def sample(hdf5_file,
        key,
        T = 300):
    dset = hdf5_file['data'][str(key)]
    label = dset['labels'][()]
    tend = dset['times'][-1] 
    start_time = 0
    ha = dset['times'][()]

    tmad = get_tmad_slice(dset['times'][()], dset['addrs'][()], start_time, T*1000)
    tmad[:,0]-=tmad[0,0]
    return tmad, label


# def create_datasets(
#         root = 'data/nmnist/n_mnist.hdf5',
#         batch_size = 72 ,
#         num_steps_train = 300,
#         num_steps_test = 300,
#         ds = 1,
#         dt = 1000,
#         transform_train = None,
#         transform_test = None,
#         target_transform_train = None,
#         target_transform_test = None):

#     size = [2, 32//ds, 32//ds]

#     if transform_train is None:
#         transform_train = Compose([
#             CropDims(low_crop=[0,0], high_crop=[32,32], dims=[2,3]),
#             Downsample(factor=[dt,1,1,1]),
#             ToCountFrame(T = num_steps_train, size = size),
#             ToTensor()])
#     if transform_test is None:
#         transform_test = Compose([
#             CropDims(low_crop=[0,0], high_crop=[32,32], dims=[2,3]),
#             Downsample(factor=[dt,1,1,1]),
#             ToCountFrame(T = num_steps_test, size = size),
#             ToTensor()])

#     if target_transform_train is None:
#         target_transform_train =Compose([Repeat(num_steps_train), toOneHot(10)])
#     if target_transform_test is None:
#         target_transform_test = Compose([Repeat(num_steps_test), toOneHot(10)])

#     train_ds = NMNIST(root,train=True,
#                                  transform = transform_train, 
#                                  target_transform = target_transform_train, 
#                                  num_steps = num_steps_train,
#                                  dt = dt)

#     test_ds = NMNIST(root, transform = transform_test, 
#                                  target_transform = target_transform_test, 
#                                  train=False,
#                                  num_steps = num_steps_test,
#                                  dt = dt)

#     return train_ds, test_ds


# def create_dataloader(
#         root = 'data/nmnist/n_mnist.hdf5',
#         batch_size = 72 ,
#         num_steps_train = 300,
#         num_steps_test = 300,
#         ds = 1,
#         dt = 1000,
#         transform_train = None,
#         transform_test = None,
#         target_transform_train = None,
#         target_transform_test = None,
#         **dl_kwargs):

#     train_d, test_d = create_datasets(
#         root = root,
#         batch_size = batch_size,
#         num_steps_train = num_steps_train,
#         num_steps_test = num_steps_test,
#         ds = ds,
#         dt = dt,
#         transform_train = transform_train,
#         transform_test = transform_test,
#         target_transform_train = target_transform_train,
#         target_transform_test = target_transform_test)


#     train_dl = torch.utils.data.DataLoader(train_d, shuffle=True, batch_size=batch_size, **dl_kwargs)
#     test_dl = torch.utils.data.DataLoader(test_d, shuffle=False, batch_size=batch_size, **dl_kwargs)

#     return train_dl, test_dl
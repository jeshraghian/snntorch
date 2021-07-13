# DVS Gesture citation: A. Amir, B. Taba, D. Berg, T. Melano, J. McKinstry, C. Di Nolfo, T. Nayak, A. Andreopoulos, G. Garreau, M. Mendoza, J. Kusnitz, M. Debole, S. Esser, T. Delbruck, M. Flickner, and D. Modha, "A Low Power, Fully Event-Based Gesture Recognition System," 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, 2017.
# Dataloader adapted from https://github.com/nmi-lab/torchneuromorphic by Emre Neftci and Clemens Schaefer

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
from tqdm import tqdm
import glob
from .._utils import *


mapping = { 0 :'Hand Clapping'  ,
            1 :'Right Hand Wave',
            2 :'Left Hand Wave' ,
            3 :'Right Arm CW'   ,
            4 :'Right Arm CCW'  ,
            5 :'Left Arm CW'    ,
            6 :'Left Arm CCW'   ,
            7 :'Arm Roll'       ,
            8 :'Air Drums'      ,
            9 :'Air Guitar'     ,
            10:'Other'}

class DVSGesture(NeuromorphicDataset):

    """`DVS Gesture <https://www.research.ibm.com/dvsgesture/>`_ Dataset.

    The data was recorded using a DVS128. The dataset contains 11 hand gestures from 29 subjects under 3 illumination conditions.

    **Number of classes:** 11

    **Number of train samples:**  1176
    
    **Number of test samples:**  288

    **Dimensions:** ``[num_steps x 2 x 128 x 128]``

    * **num_steps:** time-dimension of event-based footage
    * **2:** number of channels (on-spikes for luminance increasing; off-spikes for luminance decreasing)
    * **128x128:** W x H spatial dimensions of event-based footage

    For further reading, see:

        *A. Amir, B. Taba, D. Berg, T. Melano, J. McKinstry, C. Di Nolfo, T. Nayak, A. Andreopoulos, G. Garreau, M. Mendoza, J. Kusnitz, M. Debole, S. Esser, T. Delbruck, M. Flickner, and D. Modha, "A Low Power, Fully Event-Based Gesture Recognition System," 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, 2017.*
   



    Example::

        from snntorch.spikevision import spikedata

        train_ds = spikedata.DVSGesture("data/dvsgesture", train=True, num_steps=500, dt=1000)
        test_ds = spikedata.DVSGesture("data/dvsgesture", train=False, num_steps=1800, dt=1000)

        # by default, each time step is integrated over 1ms, or dt=1000 microseconds
        # dt can be changed to integrate events over a varying number of time steps
        # Note that num_steps should be scaled inversely by the same factor

        train_ds = spikedata.DVSGesture("data/dvsgesture", train=True, num_steps=250, dt=2000)
        test_ds = spikedata.DVSGesture("data/dvsgesture", train=False, num_steps=900, dt=2000)


    The dataset can also be manually downloaded, extracted and placed into ``root`` which will allow the dataloader to bypass straight to the generation of a hdf5 file.
    
    **Direct Download Links:**

        `IBM Box Link <https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794>`_
    
        `Dropbox Link <https://www.dropbox.com/s/cct5kyilhtsliup/DvsGesture.tar.gz?dl=0>`_


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

    :param num_steps: Number of time steps, defaults to ``500`` for train set, or ``1800`` for test set
    :type num_steps: int, optional

    :param dt: The number of time stamps integrated in microseconds, defaults to ``1000``
    :type dt: int, optional

    :param ds: Rescaling factor, defaults to ``1``.
    :type ds: int, optional

    :return_meta: Option to return metadata, defaults to ``False``
    :type return_meta: bool, optional

    :time_shuffle: Option to randomize start time of dataset, defaults to ``False``
    :type time_shuffle: bool, optional
    
    Dataloader adapted from `torchneuromorphic <https://github.com/nmi-lab/torchneuromorphic>`_ originally by Emre Neftci and Clemens Schaefer.
    
    The dataset is released under a Creative Commons Attribution 4.0 license. All rights remain with the original authors.
    """
    # _resources_url = [['Manually Download dataset here: https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/file/211521748942?sb=/details and place under {0}'.format(directory),None, 'DvsGesture.tar.gz']]
    
    _resources_url = [["https://www.dropbox.com/s/cct5kyilhtsliup/DvsGesture.tar.gz?dl=1", None, 'DvsGesture.tar.gz']]
    # directory = 'data/dvsgesture/'

    def __init__(
            self,
            root,
            train=True,
            transform=None,
            target_transform=None,
            download_and_create=True,
            num_steps = None,
            dt = 1000,
            ds = None,
            return_meta = False,
            time_shuffle=False):

        self.n = 0
        self.download_and_create = download_and_create
        self.root = root
        self.train = train
        self.dt = dt
        self.return_meta = return_meta
        self.time_shuffle = time_shuffle
        self.hdf5_name = "dvs_gesture.hdf5"
        self.directory = root.split(self.hdf5_name)[0]
        self.resources_local = [self.directory + "/DvsGesture.tar.gz"]
        self.resources_local_extracted = [self.directory + "/DvsGesture/"]

        if ds is None:
            ds = 1
        if isinstance(ds, int):
            ds = [ds, ds]


        size = [2, 128//ds[0], 128//ds[1]]  # 128//ds[0], 128//ds[1]

        if num_steps is None:
            if self.train:
                self.num_steps = 500
            else:
                self.num_steps = 1800
        else:
            self.num_steps = num_steps

        if transform is None:
            transform = Compose([
            Downsample(factor=[self.dt, 1, ds[0], ds[1]]),
            ToCountFrame(T = self.num_steps, size = size),
            ToTensor(),
            dvs_permute()
            ])
        
        if target_transform is not None:
            target_transform = Compose([Repeat(num_steps), toOneHot(11)])

        super(DVSGesture, self).__init__(
                root = root + "/" + self.hdf5_name,
                transform=transform,
                target_transform_train=target_transform)

        with h5py.File(self.root, 'r', swmr=True, libver="latest") as f:
            if train:
                self.n = f['extra'].attrs['Ntrain']
                self.keys = f['extra']['train_keys'][()]
            else:
                self.n = f['extra'].attrs['Ntest']
                self.keys = f['extra']['test_keys'][()]

    def _download(self):
        isexisting = super(DVSGesture, self)._download()

    def _create_hdf5(self):
        create_events_hdf5(self.directory, self.resources_local_extracted[0], self.directory + "/" + self.hdf5_name)

    def __len__(self):
        return self.n
        
    def __getitem__(self, key):
        #Important to open and close in getitem to enable num_workers>0
        with h5py.File(self.root, 'r', swmr=True, libver="latest") as f:
            if not self.train:
                key = key + f['extra'].attrs['Ntrain'] 
            assert key in self.keys
            data, target, meta_info_light, meta_info_user = sample(
                    f,
                    key,
                    T = self.num_steps,
                    shuffle=self.time_shuffle,
                    train=self.train)

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)


        if self.return_meta is True:
            return data, target, meta_info_light, meta_info_user
        else:
            return data, target


def sample(hdf5_file,
        key,
        T = 500,
        shuffle = False,
        train=True):
    if train:  # test
        T_default = 500
    else:
        T_default = 1800
    dset = hdf5_file['data'][str(key)]
    label = dset['labels'][()]
    tbegin = dset['times'][0]
    tend = np.maximum(0,dset['times'][-1]- 2*T*1000 )
    start_time = np.random.randint(tbegin, tend+1) if shuffle else 0
    #print(start_time)
    # tmad = get_tmad_slice(dset['times'][()], dset['addrs'][()], start_time, T*1000)
    tmad = get_tmad_slice(dset['times'][()], dset['addrs'][()], start_time, T_default*1000)
    tmad[:,0] -= tmad[0,0]
    meta = eval(dset.attrs['meta_info'])
    return tmad[:, [0,3,1,2]], label, meta['light condition'], meta['subject']


def create_events_hdf5(directory, extracted_directory, hdf5_filename):
    fns_train = gather_aedat(directory, extracted_directory, 1, 24)
    fns_test = gather_aedat(directory, extracted_directory, 24, 30)
    test_keys = []
    train_keys = []

    assert len(fns_train) == 98

    with h5py.File(hdf5_filename, 'w') as f:
        f.clear()

        key = 0
        metas = []
        data_grp = f.create_group('data')
        extra_grp = f.create_group('extra')
        print(f"\nCreating dvs_gesture.hdf5...")
        for file_d in tqdm(fns_train+fns_test):
            istrain = file_d in fns_train
            data, labels_starttime = aedat_to_events(file_d)
            tms = data[:,0]
            ads = data[:,1:]
            lbls = labels_starttime[:,0]
            start_tms = labels_starttime[:,1]
            end_tms = labels_starttime[:,2]
            out = []

            for i, v in enumerate(lbls):
                if istrain: 
                    train_keys.append(key)
                else:
                    test_keys.append(key)
                s_ = get_slice(tms, ads, start_tms[i], end_tms[i])
                times = s_[0]
                addrs = s_[1]
                 # subj, light = file_d.replace('\\', '/').split('/')[-1].split('.')[0].split('_')[:2]  # this line throws an error in get_slice, because idx_beg = idx_end --> empty batch
                subj, light = file_d.split('/')[-1].split('.')[0].split('_')[:2]
                metas.append({'key':str(key), 'subject':subj,'light condition':light, 'training sample':istrain}) 
                subgrp = data_grp.create_group(str(key))
                tm_dset = subgrp.create_dataset('times' , data=times, dtype=np.uint32)
                ad_dset = subgrp.create_dataset('addrs' , data=addrs, dtype=np.uint8)
                lbl_dset= subgrp.create_dataset('labels', data=lbls[i]-1, dtype=np.uint8)
                subgrp.attrs['meta_info']= str(metas[-1])
                assert lbls[i]-1 in range(11)
                key += 1
        extra_grp.create_dataset('train_keys', data=train_keys)
        extra_grp.create_dataset('test_keys', data=test_keys)
        extra_grp.attrs['N'] = len(train_keys) + len(test_keys)
        extra_grp.attrs['Ntrain'] = len(train_keys)
        extra_grp.attrs['Ntest'] = len(test_keys)

        print(f"dvs_gesture.hdf5 was created successfully.")
            
def gather_aedat(directory, extracted_directory, start_id, end_id, filename_prefix = 'user'):
    if not os.path.isdir(directory):
        raise FileNotFoundError("DVS Gestures Dataset not found, looked at: {}".format(directory))

    fns = []
    for i in range(start_id, end_id):
        search_mask = extracted_directory + '/' + filename_prefix + "{0:02d}".format(i) + '*.aedat'
        glob_out = glob.glob(search_mask)
        if len(glob_out)>0:
            fns+=glob_out
    return fns

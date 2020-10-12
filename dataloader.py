from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
#from torch import utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class CustomDataset(Dataset):
    """Custom dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #modify here if needed
        img_name = os.path.join(self.root_dir,
                                self.data.iloc[idx, 0])
        image = io.imread(img_name)
        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample

#modify transforms if needed
data_transform = transforms.Compose([transforms.RandomSizedCrop(size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=mean_tuple,std=std_tuple)])

custom_dataset = CustomDataset(csv_file=file_path,
                               root_dir=dir_path,
                               transform=data_transform)

dataloader = DataLoader(custom_dataset, batch_size=8, shuffle=True, num_workers=0)

import snntorch as snn
import torch
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from collections import namedtuple

#datasetConfig = namedtuple('config', ['image_size', 'batch_size', 'data_path'])
#batch_size = 1
#path = '/data/mnist'

#config = datasetConfig(image_size=[28, 28], batch_size=batch_size, data_path=path)

dl_train, dl_valid, dl_test = snn.mnist_dataloader(subset=10)
#help(snn.mnist_dataloader)

print("Training DL length: {}".format(len(dl_train.dataset)))
print("Validation DL length: {}".format(len(dl_valid.dataset)))
print("Test DL length: {}".format(len(dl_test.dataset)))

#print("Training set length: {}".format(len(ds_train)))
#print("Validation set length: {}".format(len(ds_valid)))
#print("Test set length: {}".format(len(ds_test)))

# add one hot encoding function


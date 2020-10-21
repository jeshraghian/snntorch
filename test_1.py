import snntorch as snn
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
import numpy as np
from snntorch.spikevision import datamod, spikegen

### visualization?
import matplotlib.pyplot as plt


config = snn.utils.Configuration([28,28], channels=1, batch_size=100, split=0.1, subset=100, num_classes=10, T=1000,
                           data_path='/data/mnist')

transform = transforms.Compose([
            transforms.Resize(config.input_size),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(config.data_path, train=True, download=True, transform=transform)
mnist_val = datasets.MNIST(config.data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(config.data_path, train=False, download=True, transform=transform)

# Create train-valid split
mnist_train, mnist_val = datamod.valid_split(mnist_train, mnist_val, config, seed=0)

# Create subset of data
mnist_train = datamod.data_subset(mnist_train, config)
mnist_val = datamod.data_subset(mnist_val, config)
mnist_test = datamod.data_subset(mnist_test, config)

# create dataloaders
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=config.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(mnist_val, batch_size=config.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=config.batch_size, shuffle=True)

# iterate through dataloader
train_iterator = iter(train_loader)
data_it, targets_it = next(train_iterator)

#############spike_gen of targets and data. Let's write code to run this ## VISUALIZE SPIKE DATA.
spike_data, spike_targets = spikegen.spike_conversion(data_it, targets_it, config)
print(f"Size of spike_Data is {spike_data.size()}")
print((spike_data == 1).sum())

print(f"Type of spike_data is {type(spike_data)}")
print(f"Type of spike_targets is {type(spike_targets)}")

# Let's try visualizing the data?
#matrix=np.genfromtxt(path,delimiter=',') # Read the numpy matrix with images in the rows
#c=matrix[0]
#c=c.reshape(120, 165) # this is the size of my pictures
c = spike_data[0][0].reshape(28,28)
im=plt.imshow(c)
#for row in matrix:
#    row=row.reshape(28, 28) # this is the size of my pictures
#    im.set_data(row)
#    plt.pause(0.02)
#plt.show()
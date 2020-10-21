import snntorch as snn
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import numpy as np
from snntorch.spikevision import datamod, spikegen
import matplotlib; matplotlib.use("TkAgg")
from snntorch import spikeplot

### visualization
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
train_loader = DataLoader(mnist_train, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(mnist_val, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=config.batch_size, shuffle=True)

# iterate through dataloader
train_iterator = iter(train_loader)
data_it, targets_it = next(train_iterator)

# spike generator
spike_data, spike_targets = spikegen.spike_conversion(data_it, targets_it, config, gain=2)

# show figure animation
spike_data_visualizer = spike_data[:, 0, 0]
data_sample = spikeplot.spike_animator(spike_data_visualizer, 28, 28, T=100)
plt.show()

#print(spike_targets[0][:][0])
print(f"The target is: {np.argmax(spike_targets[0][:][0], axis=0)}")

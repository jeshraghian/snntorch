import snntorch as snn
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import numpy as np
from snntorch.spikevision import datamod, spikegen
# import matplotlib; matplotlib.use("TkAgg")
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
spike_data, spike_targets = spikegen.spike_conversion(data_it, targets_it, config, gain=1)

# Convert 1000x100x1x28x28 to 1000 x 784 for Visualization
spike_data_visualizer = spike_data[:, 0, 0]
spike_data_visualizer = spike_data_visualizer.reshape((config.T, -1))

# raster plot
fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)

# ax.scatter(*torch.where(spike_data_visualizer.cpu()), s=1, c='black', marker="o", linewidths="0")

spikeplot.raster(data=spike_data_visualizer, ax=ax, s=1, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()

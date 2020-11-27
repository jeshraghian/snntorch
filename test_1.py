import snntorch as snn
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import numpy as np
from snntorch.spikevision import datamod, spikegen
# import matplotlib; matplotlib.use("TkAgg")
from snntorch import spikeplot
import snntorch.neuron as neuron

### visualization
import matplotlib.pyplot as plt


#config = snn.utils.Configuration([28,28], channels=1, batch_size=100, split=0.1, subset=100, num_classes=10, T=1000,
#                           data_path='/data/mnist')

# configuration
num_inputs = 28*28
num_hidden = 100
num_outputs = 10
num_classes = 10
batch_size=128
data_path='/data/mnist'

time_step = 1e-3
num_steps = 100

transform = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_val = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# Create train-valid split
mnist_train, mnist_val = datamod.valid_split(mnist_train, mnist_val, split=0.1, seed=0)

# Create subset of data
mnist_train = datamod.data_subset(mnist_train, subset=100)
mnist_val = datamod.data_subset(mnist_val, subset=100)
mnist_test = datamod.data_subset(mnist_test, subset=100)

# create dataloaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(mnist_val, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

# iterate through dataloader
train_iterator = iter(train_loader)
data_it, targets_it = next(train_iterator)

# spike generator
spike_data, spike_targets = spikegen.spike_conversion(data_it, targets_it, num_steps, gain=1)

# Convert 1000x100x1x28x28 to 1000 x 784 for Visualization
spike_data_visualizer = spike_data[:, 0, 0]
spike_data_visualizer = spike_data_visualizer.reshape((num_steps, -1))

####################################################
####### Fix num_classes in to_one_hot
####### Maybe separate that function, as you don't need a full time-varying to-one-hot label?

# Create a random spike train
# N_in = 10
# Sin = torch.FloatTensor(spikegen.spike_train(N_in=N_in, data_config=config, rate=0.5))
#
# # Set up a fully-connected LIF network
# from collections import namedtuple
# import torch.nn as nn
#
#
# class Net(nn.Module):
#     NeuronState = namedtuple('NeuronState', ['U', 'I', 'S'])
#
#     def __init__(self, neuron_type, in_features, out_features, bias=True, alpha=.9, beta=.85):
#         super(Net, self).__init__()
#         self.neuron_type = neuron_type
#         self.fc_layer = nn.Linear(in_features, out_features)
#         self.in_channels = in_features
#         self.out_channels = out_features
#         self.alpha = alpha
#         self.beta = beta
#         self.state = self.NeuronState(U=torch.zeros(1, out_features),
#                                       I=torch.zeros(1, out_features),
#                                       S=torch.zeros(1, out_features))
#         self.fc_layer.weight.data.uniform_(-.3, .3)
#         self.fc_layer.bias.data.uniform_(-.01, .01)
#
#     def forward(self, Sin_t):
#         state = self.state
#         U = self.alpha * state.U + state.I - state.S
#         I = self.beta * state.I + self.fc_layer(Sin_t)
#         # update the neuronal state
#         S = (U > 0).float()
#         self.state = Net.NeuronState(U=U, I=I, S=S)
#         return self.state
#
# # create a population with 10 inputs and 10 outputs
#
#
# pop1 = Net(in_features=10, out_features=10)
#
# Uprobe = np.empty([1000,N_in])
# Iprobe = np.empty([1000,N_in])
# Sprobe = np.empty([1000,N_in])
# for n in range(1000):
#     state = pop1.forward(Sin[n].unsqueeze(0))
#     Uprobe[n] = state.U.data.numpy()
#     Iprobe[n] = state.I.data.numpy()
#     Sprobe[n] = state.S.data.numpy()

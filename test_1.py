import snntorch as snn
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import numpy as np
from snntorch.spikevision import datamod, spikegen
# import matplotlib; matplotlib.use("TkAgg")
from snntorch import spikeplot
import snntorch.neuron as neuron

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
            transforms.Resize([28, 28]),
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
# spike_data, spike_targets = spikegen.spike_conversion(data_it, targets_it, num_outputs=num_outputs, num_steps=num_steps,
#                                                       gain=1, offset=0, convert_targets=True, temporal_targets=False)

#############
#### To-do: Open this in colab and working on building a network piece by piece

########## Then add surrgradclass into package
##########

import torch.nn as nn
import torch.nn.functional as F


class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).
    """

    scale = 100.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        By F. Zenke.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        return grad


# here we overwrite our naive spike function by the "SurrGradSpike" nonlinearity which implements a surrogate gradient
spike_fn = SurrGradSpike.apply

dtype = torch.float

tau_mem = 10e-3
tau_syn = 5e-3

alpha = float(np.exp(-time_step/tau_syn))
beta = float(np.exp(-time_step/tau_mem))


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 10)
        # self.fc2 = nn.Linear(100, 10)

        self.syn = torch.zeros((batch_size, 10), device=device, dtype=dtype)
        self.mem = torch.zeros((batch_size, 10), device=device, dtype=dtype)

        self.mem_rec = [self.mem]
        self.spk_rec = [self.mem]

    # rewatch classes to see which needs 'self' attached to it
    def forward(self, x):
        for t in range(num_steps):
            mthr = self.mem - 1.0
            out = spike_fn(mthr)
            rst = torch.zeros_like(self.mem) ###surely there's a better way to just get rst = mthr > 0
            c = (mthr > 0)
            rst[c] = torch.ones_like(self.mem)[c]

            new_syn = alpha*self.syn + self.fc1(x[:, t])  # B x T x C x W x H --> want to iterate through T.
            new_mem = beta*self.mem + self.syn - rst

            self.mem = new_mem
            self.syn = new_syn

            self.mem_rec.append(self.mem)
            self.spk_rec.append(out)

            # x = F.relu(self.fc1(x))
            # x = self.fc2(x)
        return self.spk_rec, self.mem_rec


# spike_data, spike_targets = spikegen.spike_conversion(data_it, targets_it, num_outputs=num_outputs, num_steps=num_steps,
#                                                       gain=1, offset=0, convert_targets=True, temporal_targets=False)

net = Net()

###
optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999))

log_softmax_fn = nn.LogSoftmax(dim=1)
loss_fn = nn.NLLLoss()

loss_hist = []
# for e in range(100):
#     data_it, targets_it = next(train_iterator)
#     spike_data, spike_targets = spikegen.spike_conversion(data_it, targets_it, num_outputs=num_outputs, num_steps=num_steps,
#                                                           gain=1, offset=0, convert_targets=True, temporal_targets=False)
#     output, mem_rec = net(spike_data)
#     log_p_y = log_softmax_fn()

spike_data, spike_targets = spikegen.spike_conversion(data_it, targets_it, num_outputs=num_outputs, num_steps=num_steps,
                                                      gain=1, offset=0, convert_targets=True, temporal_targets=False)
output, mem_rec = net(spike_data)


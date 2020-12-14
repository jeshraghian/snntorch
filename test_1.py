import snntorch as snn
from snntorch import spikeplot
from snntorch.spikevision import datamod, spikegen
from snntorch.neuron import FastSimgoidSurrogate as FSS
from snntorch.neuron import LIF
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import itertools
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Network Architecture
num_inputs = 28*28
#num_hidden = 100
num_outputs = 10

# Training Parameters
batch_size=128
data_path='/data/mnist'
#val_split = 0.1
#subset = 100

# Temporal Dynamics
num_steps = 25
time_step = 1e-3
tau_mem = 10e-3
tau_syn = 5e-3
# explore the option of not using numpy here?
# alpha = float(np.exp(-time_step/tau_syn))
# beta = float(np.exp(-time_step/tau_mem))
alpha = 0.15
beta = 0.2

dtype = torch.float

# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# Create DataLoader
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

# Define a surrogate gradient function
# Problem: scale=10000 keyword is overridden by the default function.
spike_fn = FSS.apply
# snn.neuron.slope = 50

# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

    # initialize layers
        self.fc1 = nn.Linear(28*28, 1000)
        self.lif1 = LIF(spike_fn=spike_fn, alpha=alpha, beta=beta)
        self.fc2 = nn.Linear(1000, 10)
        self.lif2 = LIF(spike_fn=spike_fn, alpha=alpha, beta=beta)

    def forward(self, x):
        spk1, syn1, mem1 = self.lif1.init_hidden(batch_size, 1000)
        spk2, syn2, mem2 = self.lif2.init_hidden(batch_size, 10)

        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x[step])
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            cur2 = self.fc2(spk1)
            # print(f"cur2 size: {cur2.size()}") # 128 x 18
            # print(f"syn2 size: {syn2.size()}") # syn2 is 100x10???
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

net = Net().to(device)

# Helper function for accuracy


def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data.view(num_steps, batch_size, -1))
    _, am = output.sum(dim=0).max(1)
    acc = np.mean((targets == am). detach().cpu().numpy())

    if train is True:
        print(f"Train Set Accuracy: {acc}")
    else:
        print(f"Test Set Accuracy: {acc}")

# Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=2e-4, betas=(0.9, 0.999))
log_softmax_fn = nn.LogSoftmax(dim=-1)
loss_fn = nn.NLLLoss()

loss_hist = []
test_loss_hist = []
counter = 0

for epoch in range(5):
    minibatch_counter = 0
    data = iter(train_loader)

    for data_it, targets_it in data:
        data_it = data_it.to(device)
        targets_it = targets_it.to(device)

        spike_data, spike_targets = spikegen.spike_conversion(data_it, targets_it, num_outputs=num_outputs, num_steps=num_steps,
                                                              gain=1, offset=0, convert_targets=False, temporal_targets=False)

        output, mem_rec = net(spike_data.view(num_steps, batch_size, -1))

        log_p_y = log_softmax_fn(mem_rec)

        loss_val = torch.zeros((1), dtype=dtype, device=device)

        # Sum loss over time steps to perform BPTT
        for t in range(num_steps):
            loss_val += loss_fn(log_p_y[t], spike_targets)

        optimizer.zero_grad()
        loss_val.backward(retain_graph=True)

        nn.utils.clip_grad_norm_(net.parameters(), 1)

        optimizer.step()

        loss_hist.append(loss_val.item())

        # test set
        test_data = itertools.cycle(test_loader)
        testdata_it, testtargets_it = next(test_data)
        testdata_it = testdata_it.to(device)
        testtargets_it = testtargets_it.to(device)

        test_spike_data, test_spike_targets = spikegen.spike_conversion(testdata_it, testtargets_it, num_outputs=num_outputs,
                                                                        num_steps=num_steps, gain=1, offset=0, convert_targets=False, temporal_targets=False)
        test_output, test_mem_rec = net(test_spike_data.view(num_steps, batch_size, -1))

        log_p_ytest = log_softmax_fn(test_mem_rec)
        log_p_ytest = log_p_ytest.sum(dim=0)
        loss_val_test = loss_fn(log_p_ytest, test_spike_targets)
        test_loss_hist.append(loss_val_test.item())

        if counter % 25 == 0:
            print(f"Epoch {epoch}, Minibatch {minibatch_counter}")
            print(f"Train Set Loss: {loss_hist[counter]}")
            print(f"Test Set Loss: {test_loss_hist[counter]}")
            print_batch_accuracy(spike_data, spike_targets, train=True)
            print_batch_accuracy(test_spike_data, test_spike_targets, train=False)
            print("\n")
        minibatch_counter += 1
        counter += 1

loss_hist_true_grad = loss_hist
test_loss_hist_true_grad = test_loss_hist

# PLOT LOSS
fig = plt.figure(facecolor="w", figsize=(10, 5))

plt.plot(loss_hist)
plt.plot(test_loss_hist)
plt.legend(["Test Loss", "Train Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
from snntorch import spikegen, neuron
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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

# create iterator
data = iter(train_loader)
data_it, targets_it = next(data)

spike_train, spike_targets = spikegen.rate(data_it, targets_it, num_steps=50, num_outputs=10)

# test heaviside auto
lif = neuron.Stein(alpha=0.5, beta=0.5)
spk, syn, mem = neuron.LIF.init_hidden(1)

# generate input
spk_in = torch.zeros(num_steps, device=device, dtype=dtype)
spk_in[5:7] = 0.51
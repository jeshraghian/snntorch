import snntorch as snn
import torch.nn as nn
import numpy as np

# Network Architecture
num_inputs = 28*28
num_hidden = 1000
num_outputs = 10

# Training Parameters
batch_size = 128


# Temporal Dynamics
num_steps = 25
time_step = 1e-3
tau_mem = 3e-3
tau_syn = 2.2e-3
alpha = float(np.exp(-time_step/tau_syn))
beta = float(np.exp(-time_step/tau_mem))

# ============== Define Network ===================

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # initialize layers ---> now we've got num_inputs and hidden_init included.
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Stein(alpha=alpha, beta=beta, num_inputs=num_hidden, batch_size=batch_size, hidden_init=True)
        # self.lif1 = snn.Stein(alpha=alpha, beta=beta, num_inputs=(5,10,69), hidden_init=True) # convolutional layer
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Stein(alpha=alpha, beta=beta, num_inputs=num_outputs, batch_size=batch_size, hidden_init=True)

    def forward(self, x):
        # self.lif1.detach_hidden() # It would be good to combine these two into one single line tho
        # self.lif2.detach_hidden()
        snn.Stein.detach_hidden()

        cur1 = self.fc1(x)
        self.lif1.spk1, self.lif1.syn1, self.lif1.mem1 = self.lif1(cur1, self.lif1.syn, self.lif1.mem)
        cur2 = self.fc2(self.lif1.spk)
        self.lif2.spk, self.lif2.syn, self.lif2.mem = self.lif2(cur2, self.lif2.syn, self.lif2.mem)

        return self.lif2.spk, self.lif2.mem

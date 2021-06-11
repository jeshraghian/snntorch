import pytest
import snntorch as snn
import torch
import torch.nn as nn


# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()

#     # initialize layers
#         snn.LIF.clear_instances()  # boilerplate
#         self.lif1 = snn.Stein(alpha=0.5, beta=0.5, num_inputs=1, batch_size=1, hidden_init=True)
#         self.srm0 = snn.SRM0(alpha=0.5, beta=0.4, num_inputs=1, batch_size=1, hidden_init=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # initialize layers
        snn.LIF.clear_instances()  # boilerplate
        self.fc1 = nn.Linear(1, 1)
        self.lif1 = snn.Synaptic(
            alpha=0.5, beta=0.5, num_inputs=1, batch_size=1, hidden_init=True
        )
        self.lif2 = snn.SRM0(
            alpha=0.6, beta=0.5, num_inputs=1, batch_size=1, hidden_init=True
        )

    def forward(self, x):
        cur1 = self.fc1(x)
        self.lif1.spk1, self.lif1.syn1, self.lif1.mem1 = self.lif1(
            cur1, self.lif1.syn, self.lif1.mem
        )
        return self.lif1.spk, self.lif1.mem


# net = Net().to(device)

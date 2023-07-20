import pytest
import snntorch as snn
import torch
import torch.nn as nn


# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()

#     # initialize layers
#         snn.LIF.clear_instances()  # boilerplate
#         self.lif1 = snn.Stein(
#         alpha=0.5, beta=0.5, num_inputs=1, batch_size=1, init_hidden=True)
#         self.srm0 = snn.SRM0(
#         alpha=0.5, beta=0.4, num_inputs=1, batch_size=1, init_hidden=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # initialize layers
        snn.LIF.clear_instances()  # boilerplate
        self.fc1 = nn.Linear(1, 1)
        self.lif1 = snn.Synaptic(
            alpha=0.5, beta=0.5, num_inputs=1, batch_size=1, init_hidden=True
        )
        self.lif2 = snn.Alpha(
            alpha=0.6, beta=0.5, num_inputs=1, batch_size=1, hidden_init=True
        )

    def forward(self, x):
        cur1 = self.fc1(x)
        self.lif1.spk1, self.lif1.syn1, self.lif1.mem1 = self.lif1(
            cur1, self.lif1.syn, self.lif1.mem
        )
        return self.lif1.spk, self.lif1.mem


# net = Net().to(device)
class EnergyEfficiencyNetTest1(nn.Module):
    def __init__(self, beta: float, num_timesteps: int | None = None):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(1, 1)
        self.lif2 = snn.Leaky(beta=beta)

        self.num_timesteps = num_timesteps

    def forward(self, x: torch.Tensor, num_timesteps: int | None = None):
        assert num_timesteps is not None or self.num_timesteps is not None, "please specify the number of timestep for this testing network!"

        # specifying timesteps in forward will override the self.num_timesteps
        if num_timesteps is not None:
            timesteps = num_timesteps
        else:
            timesteps = self.num_timesteps

        assert x.size(0) == timesteps

        # Initialize the hidden states of LIFs
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # final layers values in time
        spk2_rec, mem2_rec = [], []

        for timestep in range(timesteps):
            cur1 = self.fc1(x[timestep])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


class EnergyEfficiencyNetTest2(nn.Module):
    def __init__(self, beta: float):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.lif1 = snn.Leaky(beta=beta, threshold=0.2)
        self.fc2 = nn.Linear(64, 10)
        self.lif2 = snn.Leaky(beta=beta, threshold=0.4)
        self.reset()

    def forward(self, x: torch.Tensor):
        cur1 = self.fc1(x)
        spk1, self.mem1 = self.lif1(cur1, self.mem1)
        cur2 = self.fc2(spk1)
        spk2, self.mem2 = self.lif2(cur2, self.mem2)

        return spk2, self.mem2

    def reset(self):
        # Initialize the hidden states of LIFs
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()


class EnergyEfficiencyNetTest3(nn.Module):
    def __init__(self, beta: float):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)
        self.lif1 = snn.Leaky(beta=beta, threshold=0.5)
        self.fc2 = nn.Linear(1, 1)
        self.lif2 = snn.Leaky(beta=beta, threshold=0.25)
        self.reset()

    def forward(self, x: torch.Tensor):
        cur1 = self.fc1(x)
        spk1, self.mem1 = self.lif1(cur1, self.mem1)
        cur2 = self.fc2(spk1)
        spk2, self.mem2 = self.lif2(cur2, self.mem2)

        return spk2, self.mem2

    def forward_full(self, x: torch.Tensor):
        cur1 = self.fc1(x)
        spk1, self.mem1 = self.lif1(cur1, self.mem1)
        cur2 = self.fc2(spk1)
        spk2, self.mem2 = self.lif2(cur2, self.mem2)

        return spk1, self.mem1, spk2, self.mem2

    def reset(self):
        # Initialize the hidden states of LIFs
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()

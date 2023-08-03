import pytest
import snntorch as snn
import torch
import torch.nn as nn
import math
from snntorch.energy_estimation.energy_estimation_network_interface import EnergyEstimationNetworkInterface


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


class EnergyEfficiencyNetTest3(EnergyEstimationNetworkInterface):
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


class EnergyEfficiencyNetTest4(EnergyEstimationNetworkInterface):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Conv2d(1, 1, kernel_size=3, padding=0)
        self.lif1 = snn.Leaky(beta=0.5, threshold=1)
        self.fc2 = nn.Linear(9, 1)
        self.lif2 = snn.Leaky(beta=0.25, threshold=1)
        self.reset()

    def forward(self, x: torch.Tensor):
        cur1 = self.cnn1(x)
        spk1, self.mem1 = self.lif1(cur1, self.mem1)

        # reshape into (time, batchsize)
        spk1 = spk1.reshape(spk1.shape[0], -1)
        cur2 = self.fc2(spk1)
        spk2, self.mem2 = self.lif2(cur2, self.mem2)

        return spk2, self.mem2

    def forward_full(self, x: torch.Tensor):
        cur1 = self.cnn1(x)
        spk1, self.mem1 = self.lif1(cur1, self.mem1)
        spk1 = spk1.reshape(spk1.shape[0], -1)
        cur2 = self.fc2(spk1)
        spk2, self.mem2 = self.lif2(cur2, self.mem2)
        return spk1, self.mem1, spk2, self.mem2

    def reset(self):
        # Initialize the hidden states of LIFs
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()


class EnergyEfficiencyNetTest5Large(EnergyEstimationNetworkInterface):
    def __init__(self):
        super().__init__()
        # added just to remove a warning about declaration of variables outsides __init__
        self.mem1 = self.mem2 = self.mem3 = self.mem4 = self.mem5 = None

        self.cnn1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(beta=0.5, threshold=1)

        self.cnn2 = nn.Conv2d(16, 16, kernel_size=5, dilation=2)
        self.lif2 = snn.Leaky(beta=0.5, threshold=1)

        self.cnn3 = nn.Conv2d(16, 16, kernel_size=7, dilation=3)
        self.lif3 = snn.Leaky(beta=0.5, threshold=1)

        self.fc1 = nn.Linear(576, 32)
        self.lif4 = snn.Leaky(beta=0.5, threshold=1)

        self.fc2 = nn.Linear(32, 10)
        self.lif5 = snn.Leaky(beta=0.5, threshold=1)
        self.reset()

    def forward(self, x: torch.Tensor):
        cur1 = self.cnn1(x)
        spk1, self.mem1 = self.lif1(cur1, self.mem1)

        cur2 = self.cnn2(spk1)
        spk2, self.mem2 = self.lif2(cur2, self.mem2)

        cur3 = self.cnn3(spk2)
        spk3, self.mem3 = self.lif3(cur3, self.mem3)

        # flatten it out
        spk3 = spk3.reshape(spk1.shape[0], 16 * 6 * 6)

        cur4 = self.fc1(spk3)
        spk4, self.mem4 = self.lif4(cur4, self.mem4)

        cur5 = self.fc2(spk4)
        spk5, self.mem5 = self.lif5(cur5, self.mem5)

        return spk5, self.mem5

    def reset(self):
        # Initialize the hidden states of LIFs
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()
        self.mem3 = self.lif3.init_leaky()
        self.mem4 = self.lif4.init_leaky()
        self.mem5 = self.lif5.init_leaky()


class EnergyEfficiencyNet1DConv(EnergyEstimationNetworkInterface):
    def __init__(self):
        super().__init__()
        # added just to remove a warning about declaration of variables outsides __init__
        self.mem1 = self.mem2 = self.mem3 = self.mem4 = None
        self.cnn1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(beta=0.5, threshold=1)

        self.cnn2 = nn.Conv1d(16, 16, kernel_size=5)
        self.lif2 = snn.Leaky(beta=0.5, threshold=1)

        self.reshape_val = 16 * 28
        self.fc1 = nn.Linear(self.reshape_val, 64)
        self.lif3 = snn.Leaky(beta=0.5, threshold=1)

        self.cls = nn.Linear(64, 1)
        self.cls_activ = snn.Leaky(beta=0.5, threshold=1)
        self.reset()

    def forward(self, x: torch.Tensor):
        cur1 = self.cnn1(x)
        spk1, self.mem1 = self.lif1(cur1, self.mem1)

        cur2 = self.cnn2(spk1)
        spk2, self.mem2 = self.lif2(cur2, self.mem2)
        spk2 = spk2.reshape(-1, self.reshape_val)

        cur3 = self.fc1(spk2)
        spk3, self.mem3 = self.lif3(cur3, self.mem3)

        cur4 = self.cls(spk3)
        spk4, self.mem4 = self.cls_activ(cur4, self.mem4)
        return spk4, self.mem4

    def reset(self):
        # Initialize the hidden states of LIFs
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()
        self.mem3 = self.lif3.init_leaky()
        self.mem4 = self.cls_activ.init_leaky()


class UnrecognizedLeafTestLayer(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(input, self.weight, self.bias)


class EnergyEfficiencyNetUnrecognizedLayers(EnergyEstimationNetworkInterface):
    def __init__(self):
        super().__init__()
        self.mem1 = self.mem2 = None
        self.fc1 = UnrecognizedLeafTestLayer(128, 32)
        self.lif1 = snn.Leaky(beta=0.5, threshold=1)
        self.fc2 = UnrecognizedLeafTestLayer(32, 16)
        self.lif2 = snn.Leaky(beta=0.5, threshold=1)
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


class NestedNetworkModule(nn.Module):
    def __init__(self, inp: int, out: int):
        super().__init__()
        self.fc1 = nn.Linear(inp, out)
        self.lif1 = snn.Leaky(beta=1)
        self.mem1 = None
        self.reset()

    def forward(self, x):
        x = self.fc1(x)
        spk1, self.mem1 = self.lif1(x, self.mem1)
        return spk1

    def reset(self):
        self.mem1 = self.lif1.init_leaky()


class NestedNetworkBlock(nn.Module):
    def __init__(self, inp, hidden1, hidden2, out):
        super().__init__()
        self.module1 = NestedNetworkModule(inp, hidden1)
        self.module2 = NestedNetworkModule(hidden1, hidden2)
        self.module3 = NestedNetworkModule(hidden2, out)

    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        return x

    def reset(self):
        self.module1.reset()
        self.module2.reset()
        self.module3.reset()


class EnergyEfficiencyNestedNetworkTest(EnergyEstimationNetworkInterface):
    def __init__(self):
        super().__init__()
        self.encoder = NestedNetworkBlock(128, 64, 32, 16)
        self.decoder = NestedNetworkBlock(16, 32, 64, 128)

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        return self.decoder(x)

    def reset(self):
        # Initialize the hidden states of LIFs
        self.encoder.reset()
        self.decoder.reset()


class EnergyEfficiencyNestedNetworkTestUnfolded(EnergyEstimationNetworkInterface):

    def __init__(self, Beta=0.9):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.lif1 = snn.Leaky(Beta)

        self.fc2 = nn.Linear(64, 32)
        self.lif2 = snn.Leaky(Beta)

        self.fc3 = nn.Linear(32, 16)
        self.lif3 = snn.Leaky(Beta)

        self.fc4 = nn.Linear(16, 32)
        self.lif4 = snn.Leaky(Beta)

        self.fc5 = nn.Linear(32, 64)
        self.lif5 = snn.Leaky(Beta)

        self.fc6 = nn.Linear(64, 128)
        self.lif6 = snn.Leaky(Beta)
        self.reset()

    def forward(self, x: torch.Tensor):
        spk, self.mem1 = self.lif1(self.fc1(x), self.mem1)
        spk, self.mem2 = self.lif2(self.fc2(spk), self.mem2)
        spk, self.mem3 = self.lif3(self.fc3(spk), self.mem3)
        spk, self.mem4 = self.lif4(self.fc4(spk), self.mem4)
        spk, self.mem5 = self.lif5(self.fc5(spk), self.mem5)
        return self.lif6(self.fc6(spk), self.mem6)

    def reset(self):
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif1.init_leaky()
        self.mem3 = self.lif1.init_leaky()
        self.mem4 = self.lif1.init_leaky()
        self.mem5 = self.lif1.init_leaky()
        self.mem6 = self.lif1.init_leaky()

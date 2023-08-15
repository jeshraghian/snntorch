import torch
from torch.nn import Module


class EnergyEstimationNetworkInterface(Module):
    """
        An interface which makes the estimate_energy easier, because it will use the .reset function which should
        reset the potentials on the membrane (or in general hidden states), therefore guaranteeing consistent results
        when doing the forward passes on given spike train (before doing forward passes, estimate_energy will reset
        the network).

        TODO : this is not a great solution but it currently works
    """

    def __init__(self):
        super().__init__()

    def forward(self):
        raise NotImplemented()

    def reset(self):
        raise NotImplemented()

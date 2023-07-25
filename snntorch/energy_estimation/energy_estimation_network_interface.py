from torch.nn import Module


class EnergyEstimationNetworkInterface(Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        raise NotImplemented()

    def reset(self):
        raise NotImplemented()

import torch


class GradedSpikes(torch.nn.Module):
    """Learnable spiking magnitude for spiking layers."""

    def __init__(self, size, constant_factor=None):
        super().__init__()
        self.size = size
        weights = torch.Tensor(size)
        self.weights = torch.nn.Parameter(weights)

        if constant_factor:
            torch.nn.init.ones_(tensor=self.weights) * constant_factor
        else:
            torch.nn.init.uniform_(tensor=self.weights, a=0.0, b=1.0)

    def forward(self, x):
        """Forward pass is simply: spikes 'x' * weights."""
        return torch.multiply(input=x, other=self.weights)

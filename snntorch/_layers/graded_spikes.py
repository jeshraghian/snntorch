import torch


class GradedSpikes(torch.nn.Module):
    """Learnable spiking magnitude for spiking layers."""

    def __init__(self, size, constant_factor):
        """

        :param size: The input size of the layer. Must be equal to the
            number of neurons in the preceeding layer.
        :type size: int
        :param constant_factor: If provided, the weights will be
            initialized with ones times the contant_factor. If
            not provided, the weights will be initialized using a uniform
            distribution U(0.5, 1.5).
        :type constant_factor: float

        """
        super().__init__()
        self.size = size

        if constant_factor:
            weights = torch.ones(size=[size, 1]) * constant_factor
            self.weights = torch.nn.Parameter(weights)
        else:
            weights = torch.rand(size=[size, 1]) + 0.5
            self.weights = torch.nn.Parameter(weights)

    def forward(self, x):
        """Forward pass is simply: spikes 'x' * weights."""
        return torch.multiply(input=x, other=self.weights)

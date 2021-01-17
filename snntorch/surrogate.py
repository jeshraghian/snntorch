import torch

# Spike-gradient functions

slope = 25


class FastSigmoid(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.
    Forward pass: Heaviside step function.
    Backward pass: Gradient of fast sigmoid function.

    f(x) = x / (1 + abs(x))
    f'(x) = 1 / ([1+slope*abs(x)]^2)

    Adapted from Zenke & Ganguli (2018).
    """

    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        out = torch.zeros_like(input_)
        out[input_ > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (slope * torch.abs(input_) + 1.0) ** 2
        return grad


class Sigmoid(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.
    Forward pass: Heaviside step function.
    Backward pass: Gradient of sigmoid function.

    f(x) = 1 / (1 + exp(-slope * x)
    f'(x) = slope*exp(-slope*x) / ((exp(-A*x)+1)^2)
    """

    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        out = torch.zeros_like(input_)
        out[input_ > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * slope * torch.exp(-slope*input_) / ((torch.exp(-slope*input_)+1) ** 2)
        return grad


class SpikeRateEscape(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.
    Forward pass: Boltzmann Distribution.
    Backward pass: Gradient of Boltzmann Distribution.

    f'(x) = slope * exp(-beta*abs(x - 1))

    Wulfram Gerstner and Werner M. Kistler, "Spiking neuron models: Single neurons, populations, plasticity."
    Cambridge University Press, 2002.
    """

    def __init__(self, beta=1):
        self.beta = beta

    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        out = torch.zeros_like(input_)
        out[input_ > 0] = 1.0
        return out

    def backward(self, ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * slope * torch.exp(-self.beta*torch.abs(input_ - 1))
        return grad

# piecewise linear func
# tanh surrogate func
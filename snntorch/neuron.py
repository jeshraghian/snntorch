import torch
import torch.nn as nn


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float
slope = 25


# LIF: don't forget to add new variables to LIF as new neuron classes are added.


class LIF(nn.Module):
    """Leaky Integrate and Fire Neuron."""
    def __init__(self, alpha, beta, threshold=1.0, spike_grad=None):
        super(LIF, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold

        if spike_grad is None:
            self.spike_grad = self.Heaviside.apply  # Heaviside will not be parsed unless it is above or within LIF class.
        else:
            self.spike_grad = spike_grad

    @staticmethod
    def init_hidden(batch_size, *args):
        """Used to initialize syn, mem and spk.
        *args are the input feature dimensions.
        E.g., batch_size=128 and input feature of size=1x28x28 would require init_hidden(128, 1, 28, 28)."""
        syn = torch.zeros((batch_size, *args), device=device, dtype=dtype)
        mem = torch.zeros((batch_size, *args), device=device, dtype=dtype)
        spk = torch.zeros((batch_size, *args), device=device, dtype=dtype)

        return spk, syn, mem

    @staticmethod
    class Heaviside(torch.autograd.Function):
        """Default and non-approximate spiking function for neuron.
        Forward pass: Heaviside step function.
        Backward pass: Dirac Delta clipped to 1 at x=0."""

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            out = torch.zeros_like(input)
            out[input > 0] = 1.0
            return out

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            if input == 0:
                grad = grad_input * 1
            else:
                grad = 0
            return grad


# Neuron Models

class Stein(LIF):
    def __init__(self, alpha, beta, threshold, spike_grad):
        super(Stein, self).__init__(alpha, beta, threshold, spike_grad)

    def forward(self, input, syn, mem):
        mem_shift = mem - self.threshold
        spk = self.spike_grad(mem_shift).to(device)
        reset = torch.zeros_like(mem)
        spk_idx = (mem_shift > 0)
        reset[spk_idx] = torch.ones_like(mem)[spk_idx]

        syn = self.alpha * syn + input
        mem = self.beta * mem + syn - reset

        return spk, syn, mem


# class SRM(LIF):
#     def __init__(self, alpha, beta, gamma, threshold, spike_grad):
#         super(SRM, self).__init__(alpha, beta, gamma, threshold, spike_grad) # add gamma and any other params to LIF
#
#     def forward(self, input, syn, mem): # modify this
#         mem_shift = mem - self.threshold
#         spk = self.spike_grad(mem_shift).to(device)
#         reset = torch.zeros_like(mem)
#         spk_idx = (mem_shift > 0)
#         reset[spk_idx] = torch.ones_like(mem)[spk_idx]
#
#         mem = (self.alpha / (self.alpha + self.beta))*(self.alpha * mem - self.beta * mem) + input - reset
#         # mem = (self.beta * mem + syn - reset)
#
#         return spk, mem  # don't need syn



# Spike-gradient functions


class FastSimgoidSurrogate(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.
    Forward pass: Heaviside step function.
    Backward pass: Gradient of fast sigmoid function.

    f(x) = x / (1 + abs(x))
    f'(x) = 1 / ([1+slope*abs(x)]^2)

    Adapted from Zenke & Ganguli (2018).
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (slope * torch.abs(input) + 1.0) ** 2
        return grad


class SigmoidSurrogate(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step funfction.
    Forward pass: Heaviside step function.
    Backward pass: Gradient of sigmoid function.

    f(x) = 1 / (1 + exp(-slope * x)
    f'(x) = slope*exp(-slope*x) / ((exp(-A*x)+1)^2)
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * slope * torch.exp(-slope*input) / ((torch.exp(-slope*input)+1) ** 2)
        return grad

# piecewise linear func
# tanh surrogate func
# boltzmann func
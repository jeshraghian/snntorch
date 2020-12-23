import torch
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float
slope = 25

class LIF(nn.Module):
    """Based on F. Zenke's code (2020)"""
    def __init__(self, spike_fn, alpha, beta):
        super(LIF, self).__init__()
        self.spike_fn = spike_fn
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, syn, mem):
        mthr = mem - 1.0
        spk = self.spike_fn(mthr).to(device)
        rst = torch.zeros_like(mem)
        c = (mthr > 0)
        rst[c] = torch.ones_like(mem)[c]

        syn = self.alpha*syn + input
        mem = self.beta*mem + syn - rst
        return spk, syn, mem

    @staticmethod
    def init_hidden(batch_size, *args):
        """Used to initialize syn, mem and spk. *args are the input feature dimensions.
        E.g., batch_size=128 and input feature of size=1x28x28 would call init_hidden(128, 1, 28, 28)."""
        syn = torch.zeros((batch_size, *args), device=device, dtype=dtype)
        mem = torch.zeros((batch_size, *args), device=device, dtype=dtype)
        spk = torch.zeros((batch_size, *args), device=device, dtype=dtype)

        return spk, syn, mem


class FastSimgoidSurrogate(torch.autograd.Function):
    """
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

import torch
import torch.nn as nn
import numpy as np


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float
slope = 25


class LIF(nn.Module):
    """Parent class for leaky integrate and fire neuron models."""
    def __init__(self, alpha, beta, threshold=1.0, spike_grad=None):
        super(LIF, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold

        if spike_grad is None:
            self.spike_grad = self.Heaviside.apply
        else:
            self.spike_grad = spike_grad

    @staticmethod
    def init_stein(batch_size, *args):
        """Used to initialize syn, mem and spk.
        *args are the input feature dimensions.
        E.g., batch_size=128 and input feature of size=1x28x28 would require init_hidden(128, 1, 28, 28)."""
        syn = torch.zeros((batch_size, *args), device=device, dtype=dtype)
        mem = torch.zeros((batch_size, *args), device=device, dtype=dtype)
        spk = torch.zeros((batch_size, *args), device=device, dtype=dtype)

        return spk, syn, mem

    @staticmethod
    def init_srm0(batch_size, *args):
        """Used to initialize syn_pre, syn_post, mem and spk.
        *args are the input feature dimensions.
        E.g., batch_size=128 and input feature of size=1x28x28 would require init_hidden(128, 1, 28, 28)."""
        syn_pre = torch.zeros((batch_size, *args), device=device, dtype=dtype)
        syn_post = torch.zeros((batch_size, *args), device=device, dtype=dtype)
        mem = torch.zeros((batch_size, *args), device=device, dtype=dtype)
        spk = torch.zeros((batch_size, *args), device=device, dtype=dtype)

        return spk, syn_pre, syn_post, mem

    @staticmethod
    class Heaviside(torch.autograd.Function):
        """Default and non-approximate spiking function for neuron.
        Forward pass: Heaviside step function.
        Backward pass: Dirac Delta clipped to 1 at x>0 instead of inf at x=1.
        This assumption holds true on the basis that a spike occurs as long as x>0 and the following time step incurs a reset."""

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
            grad_input[input_ < 0] = 0.0
            grad = grad_input
            return grad


# Neuron Models

class Stein(LIF):
    """
    Stein's model of the leaky integrate and fire neuron.
    The synaptic current jumps upon spike arrival, which causes a jump in membrane potential.
    Synaptic current and membrane potential decay exponentially with rates of alpha and beta, respectively.

    For further reading, see:
    R. B. Stein (1965) A theoretical analysis of neuron variability. Biophys. J. 5, pp. 173-194.
    R. B. Stein (1967) Some models of neuronal variability. Biophys. J. 7. pp. 37-68."""

    def __init__(self, alpha, beta, threshold=1.0, spike_grad=None):
        super(Stein, self).__init__(alpha, beta, threshold, spike_grad)

    def forward(self, input_, syn, mem):
        mem_shift = mem - self.threshold
        spk = self.spike_grad(mem_shift).to(device)
        reset = torch.zeros_like(mem)
        spk_idx = (mem_shift > 0)
        reset[spk_idx] = torch.ones_like(mem)[spk_idx]

        syn = self.alpha * syn + input_
        mem = self.beta * mem + syn - reset

        return spk, syn, mem


class SRM0(LIF):
    """
    Simplified Spike Response Model of the leaky integrate and fire neuron.
    The time course of the membrane potential response depends on a combination of exponentials.
    In this case, the change in post-synaptic potential experiences a delay.
    This can be interpreted as the input current taking on its own exponential shape as a result of an input spike train.
    For excitatory spiking, ensure alpha > beta.

    For further reading, see:
    R. Jovilet, J. Timothy, W. Gerstner (2003) The spike response model: A framework to predict neuronal spike trains. Artificial Neural Networks and Neural Information Processing, pp. 846-853.
    """
    def __init__(self, alpha, beta, threshold=1.0, spike_grad=None):
        super(SRM0, self).__init__(alpha, beta, threshold, spike_grad)
        self.tau_srm = np.log(self.alpha) / (np.log(self.beta) - np.log(self.alpha)) + 1
        if self.alpha <= self.beta:
            raise ValueError("alpha must be greater than beta.")

    def forward(self, input_, syn_pre, syn_post, mem):
        mem_shift = mem - self.threshold
        spk = self.spike_grad(mem_shift).to(device)
        reset = torch.zeros_like(mem)
        spk_idx = (mem_shift > 0)
        reset[spk_idx] = torch.ones_like(mem)[spk_idx]

        syn_pre = (self.alpha * syn_pre + input_) * (1 - reset)
        syn_post = (self.beta * syn_post - input_) * (1 - reset)
        mem = self.tau_srm * (syn_pre + syn_post)*(1-reset) + (mem*reset - reset)

    # cool forward function that resulted in firing bursts - worth exploring

    # def forward(self, input_, syn_pre, syn_post, mem):
    #     mem_shift = mem - self.threshold
    #     spk = self.spike_grad(mem_shift).to(device)
    #     reset = torch.zeros_like(mem)
    #     spk_idx = (mem_shift > 0)
    #     reset[spk_idx] = torch.ones_like(mem)[spk_idx]
    #
    #     syn_pre = self.alpha * syn_pre + input_
    #     syn_post = self.beta * syn_post - input_
    #     mem = self.tau_srm * (syn_pre + syn_post) - reset

        return spk, syn_pre, syn_post, mem


# Spike-gradient functions


class FastSigmoidSurrogate(torch.autograd.Function):
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


class SigmoidSurrogate(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step funfction.
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

# boltzmann func
# piecewise linear func
# tanh surrogate func
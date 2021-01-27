import torch
import torch.nn as nn
import numpy as np


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float
slope = 25


class LIF(nn.Module):
    """Parent class for leaky integrate and fire neuron models."""
    instances = []
    def __init__(self, alpha, beta, threshold=1.0, spike_grad=None):
        super(LIF, self).__init__()
        LIF.instances.append(self)

        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold

        if spike_grad is None:
            self.spike_grad = self.Heaviside.apply
        else:
            self.spike_grad = spike_grad

    def fire(self, mem):
        """Generates spike if mem > threshold.
        Returns spk and reset."""
        mem_shift = mem - self.threshold
        spk = self.spike_grad(mem_shift).to(device)
        reset = torch.zeros_like(mem)
        spk_idx = (mem_shift > 0)
        reset[spk_idx] = torch.ones_like(mem)[spk_idx]
        return spk, reset

    def fire_single(self, batch_size, mem):
        """Generates spike if mem > threshold.
        Returns spk and reset."""
        mem_shift = mem - self.threshold
        index = torch.argmax(mem_shift, dim=1)
        spk_tmp = self.spike_grad(mem_shift)

        mask_spk1 = torch.zeros_like(spk_tmp)
        mask_spk1[torch.arange(batch_size), index] = 1
        spk = spk_tmp * mask_spk1.to(device)

        reset = torch.zeros_like(mem)
        spk_idx = (mem_shift > 0)
        reset[spk_idx] = torch.ones_like(mem)[spk_idx]
        return spk, reset

    @classmethod
    def clear_instances(cls):
      cls.instances = []

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
    def detach(*args):
        """Used to detach input arguments from the current graph.
        Intended for use in truncated backpropagation through time where hidden state variables are global variables."""
        for state in args:
            state.detach_()

    @staticmethod
    def zeros(*args):
        """Used to clear hidden state variables to zero.
            Intended for use where hidden state variables are global variables."""
        for state in args:
            state = torch.zeros_like(state)

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
    For mem[T] > threshold, spk[T+1] = 0 to account for axonal delay.

    For further reading, see:
    R. B. Stein (1965) A theoretical analysis of neuron variability. Biophys. J. 5, pp. 173-194.
    R. B. Stein (1967) Some models of neuronal variability. Biophys. J. 7. pp. 37-68."""

    def __init__(self, alpha, beta, threshold=1.0, num_inputs=False, spike_grad=None, batch_size=False, hidden_init=False):
        super(Stein, self).__init__(alpha, beta, threshold, spike_grad)

        self.num_inputs = num_inputs
        self.batch_size = batch_size
        self.hidden_init = hidden_init

        if self.hidden_init:
            if not self.num_inputs:
                raise ValueError("num_inputs must be specified to initialize hidden states as instance variables.")
            elif not self.batch_size:
                raise ValueError("batch_size must be specified to initialize hidden states as instance variables.")
            elif hasattr(self.num_inputs, '__iter__'):
                self.spk, self.syn, self.mem = self.init_stein(self.batch_size, *(self.num_inputs)) # need to automatically call batch_size
            else:
                self.spk, self.syn, self.mem = self.init_stein(self.batch_size, self.num_inputs)

    def forward(self, input_, syn, mem):
        if not self.hidden_init:
            spk, reset = self.fire(mem)
            syn = self.alpha * syn + input_
            mem = self.beta * mem + syn - reset

            return spk, syn, mem

        # intended for truncated-BPTT where instance variables are hidden states
        if self.hidden_init:
            self.spk, self.reset = self.fire(self.mem)
            self.syn = self.alpha * self.syn + input_
            self.mem = self.beta * self.mem + self.syn - self.reset

            return self.spk, self.syn, self.mem

    @classmethod
    def detach_hidden(cls):
        """Used to detach hidden states from the current graph.
        Intended for use in truncated backpropagation through time where hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            cls.instances[layer].spk.detach_()
            cls.instances[layer].syn.detach_()
            cls.instances[layer].mem.detach_()

    @classmethod
    def zeros_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            cls.instances[layer].spk = torch.zeros_like(cls.instances[layer].spk)
            cls.instances[layer].syn = torch.zeros_like(cls.instances[layer].syn)
            cls.instances[layer].mem = torch.zeros_like(cls.instances[layer].mem)

class Stein_single(LIF):
    """
    Stein's model of the leaky integrate and fire neuron.
    The synaptic current jumps upon spike arrival, which causes a jump in membrane potential.
    Synaptic current and membrane potential decay exponentially with rates of alpha and beta, respectively.
    For mem[T] > threshold, spk[T+1] = 0 to account for axonal delay.

    For further reading, see:
    R. B. Stein (1965) A theoretical analysis of neuron variability. Biophys. J. 5, pp. 173-194.
    R. B. Stein (1967) Some models of neuronal variability. Biophys. J. 7. pp. 37-68."""

    def __init__(self, alpha, beta, threshold=1.0, num_inputs=False, spike_grad=None, batch_size=False, hidden_init=False):
        super(Stein, self).__init__(alpha, beta, threshold, spike_grad)

        self.num_inputs = num_inputs
        self.batch_size = batch_size
        self.hidden_init = hidden_init

        if self.hidden_init:
            if not self.num_inputs:
                raise ValueError("num_inputs must be specified to initialize hidden states as instance variables.")
            elif not self.batch_size:
                raise ValueError("batch_size must be specified to initialize hidden states as instance variables.")
            elif hasattr(self.num_inputs, '__iter__'):
                self.spk, self.syn, self.mem = self.init_stein(self.batch_size, *(self.num_inputs)) # need to automatically call batch_size
            else:
                self.spk, self.syn, self.mem = self.init_stein(self.batch_size, self.num_inputs)

    def forward(self, input_, syn, mem):
        if not self.hidden_init:
            spk, reset = self.fire_single(self.batch_size, mem)
            syn = self.alpha * syn + input_
            mem = self.beta * mem + syn - reset

            return spk, syn, mem

        # intended for truncated-BPTT where instance variables are hidden states
        if self.hidden_init:
            self.spk, self.reset = self.fire(self.mem)
            self.syn = self.alpha * self.syn + input_
            self.mem = self.beta * self.mem + self.syn - self.reset

            return self.spk, self.syn, self.mem

    @classmethod
    def detach_hidden(cls):
        """Used to detach hidden states from the current graph.
        Intended for use in truncated backpropagation through time where hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            cls.instances[layer].spk.detach_()
            cls.instances[layer].syn.detach_()
            cls.instances[layer].mem.detach_()

    @classmethod
    def zeros_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            cls.instances[layer].spk = torch.zeros_like(cls.instances[layer].spk)
            cls.instances[layer].syn = torch.zeros_like(cls.instances[layer].syn)
            cls.instances[layer].mem = torch.zeros_like(cls.instances[layer].mem)


class SRM0(LIF):
    """
    Simplified Spike Response Model of the leaky integrate and fire neuron.
    The time course of the membrane potential response depends on a combination of exponentials.
    In this case, the change in membrane potential experiences a delay.
    This can be interpreted as the input current taking on its own exponential shape as a result of an input spike train.
    For excitatory spiking, ensure alpha > beta.
    For mem[T] > threshold, spk[T+1] = 0 to account for axonal delay.

    For further reading, see:
    R. Jovilet, J. Timothy, W. Gerstner (2003) The spike response model: A framework to predict neuronal spike trains. Artificial Neural Networks and Neural Information Processing, pp. 846-853.
    """

    def __init__(self, alpha, beta, threshold=1.0, num_inputs=False, spike_grad=None, batch_size=False, hidden_init=False):
        super(SRM0, self).__init__(alpha, beta, threshold, spike_grad)

        self.num_inputs = num_inputs
        self.batch_size = batch_size
        self.hidden_init = hidden_init

        if self.hidden_init:
            if not self.num_inputs:
                raise ValueError("num_inputs must be specified to initialize hidden states as instance variables.")
            elif not self.batch_size:
                raise ValueError("batch_size must be specified to initialize hidden states as instance variables.")
            elif hasattr(self.num_inputs, '__iter__'):
                self.spk, self.syn_pre, self.syn_post, self.mem = self.init_srm0(batch_size=self.batch_size,
                                                                                 *(self.num_inputs))
            else:
                self.spk, self.syn_pre, self.syn_post, self.mem = self.init_srm0(batch_size, num_inputs)

        self.tau_srm = np.log(self.alpha) / (np.log(self.beta) - np.log(self.alpha)) + 1
        if self.alpha <= self.beta:
            raise ValueError("alpha must be greater than beta.")

    def forward(self, input_, syn_pre, syn_post, mem):
        # if hidden states are passed externally
        if not self.hidden_init:
            spk, reset = self.fire(mem)
            syn_pre = (self.alpha * syn_pre + input_) * (1 - reset)
            syn_post = (self.beta * syn_post - input_) * (1 - reset)
            mem = self.tau_srm * (syn_pre + syn_post)*(1-reset) + (mem*reset - reset)
            return spk, syn_pre, syn_post, mem

        # if hidden states and outputs are instance variables
        if self.hidden_init:
            self.spk, self.reset = self.fire(self.mem)
            self.syn_pre = (self.alpha * self.syn_pre + input_) * (1 - self.reset)
            self.syn_post = (self.beta * self.syn_post - input_) * (1 - self.reset)
            self.mem = self.tau_srm * (self.syn_pre + self.syn_post) * (1 - self.reset) + (self.mem * self.reset - self.reset)
            return self.spk, self.syn_pre, self.syn_post, self.mem

    # cool forward function that resulted in burst firing - worth exploring

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

        # return spk, syn_pre, syn_post, mem

    @classmethod
    def detach_hidden(cls):
        """Used to detach hidden states from the current graph.
        Intended for use in truncated backpropagation through time where hidden state variables are instance variables."""
        for layer in range(len(cls.instances)):
            cls.instances[layer].spk.detach_()
            cls.instances[layer].syn_pre.detach_()
            cls.instances[layer].syn_post.detach_()
            cls.instances[layer].mem.detach_()

    @classmethod
    def zeros_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables."""
        for layer in range(len(cls.instances)):
            cls.instances[layer].spk = torch.zeros_like(cls.instances[layer].spk)
            cls.instances[layer].syn_pre = torch.zeros_like(cls.instances[layer].syn_pre)
            cls.instances[layer].syn_post = torch.zeros_like(cls.instances[layer].syn_post)
            cls.instances[layer].mem = torch.zeros_like(cls.instances[layer].mem)


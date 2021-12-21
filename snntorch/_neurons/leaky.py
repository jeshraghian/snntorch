from .lif import *


class Leaky(LIF):
    """
    First-order leaky integrate-and-fire neuron model.
    Input is assumed to be a current injection.
    Membrane potential decays exponentially with rate beta.
    For :math:`U[T] > U_{\\rm thr} ⇒ S[T+1] = 1`.

    If `reset_mechanism = "subtract"`, then :math:`U[t+1]` will have `threshold` subtracted from it whenever the neuron emits a spike:

    .. math::

            U[t+1] = βU[t] + I_{\\rm in}[t+1] - RU_{\\rm thr}

    If `reset_mechanism = "zero"`, then :math:`U[t+1]` will be set to `0` whenever the neuron emits a spike:

    .. math::

            U[t+1] = βU[t] + I_{\\rm syn}[t+1] - R(βU[t] + I_{\\rm in}[t+1])

    * :math:`I_{\\rm in}` - Input current
    * :math:`U` - Membrane potential
    * :math:`U_{\\rm thr}` - Membrane threshold
    * :math:`R` - Reset mechanism: if active, :math:`R = 1`, otherwise :math:`R = 0`
    * :math:`β` - Membrane potential decay rate

    Example::

        import torch
        import torch.nn as nn
        import snntorch as snn

        alpha = 0.9
        beta = 0.5

        # Define Network
        class Net(nn.Module):
            def __init__(self):
                super().__init__()

                # initialize layers
                self.fc1 = nn.Linear(num_inputs, num_hidden)
                self.lif1 = snn.Leaky(beta=beta)
                self.fc2 = nn.Linear(num_hidden, num_outputs)
                self.lif2 = snn.Leaky(beta=beta)

            def forward(self, x, mem1, spk1, mem2):
                cur1 = self.fc1(x)
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                return mem1, spk1, mem2, spk2


    """

    def __init__(
        self,
        beta,
        threshold=1.0,
        spike_grad=None,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        reset_mechanism="subtract",
        output=False,
    ):
        super(Leaky, self).__init__(
            beta,
            threshold,
            spike_grad,
            init_hidden,
            inhibition,
            learn_beta,
            learn_threshold,
            reset_mechanism,
            output,
        )

        if self.init_hidden:
            self.mem = self.init_leaky()

    def forward(self, input_, mem=False):

        if hasattr(mem, "init_flag"):  # only triggered on first-pass
            mem = _SpikeTorchConv(mem, input_=input_)
        elif mem is False and hasattr(self.mem, "init_flag"):  # init_hidden case
            self.mem = _SpikeTorchConv(self.mem, input_=input_)

        # TODO: alternatively, we could do torch.exp(-1 / self.beta.clamp_min(0)),
        # giving actual time constants instead of values in [0, 1] as initial beta
        beta = self.beta.clamp(0, 1)

        if not self.init_hidden:
            reset = self.mem_reset(mem)
            if self.reset_mechanism == "subtract":
                mem = beta * mem + input_ - reset * self.threshold

            elif self.reset_mechanism == "zero":
                mem = beta * mem + input_ - reset * (beta * mem + input_)

            if self.inhibition:
                spk = self.fire_inhibition(mem.size(0), mem)  # batch_size
            else:
                spk = self.fire(mem)

            return spk, mem

        # intended for truncated-BPTT where instance variables are hidden states
        if self.init_hidden and not mem:
            self.reset = self.mem_reset(self.mem)
            if self.reset_mechanism == "subtract":
                self.mem = beta * self.mem + input_ - self.reset * self.threshold

            elif self.reset_mechanism == "zero":
                self.mem = (
                    beta * self.mem + input_ - self.reset * (beta * self.mem + input_)
                )
            if self.inhibition:
                self.spk = self.fire_inhibition(self.mem.size(0), self.mem)
            else:
                self.spk = self.fire(self.mem)

            if self.output:  # read-out layer returns output+states
                return self.spk, self.mem
            else:  # hidden layer e.g., in nn.Sequential, only returns output
                return self.spk

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Leaky):
                cls.instances[layer].mem.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables.
        Assumes hidden states have a batch dimension already."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Leaky):
                cls.instances[layer].mem = _SpikeTensor(init_flag=False)

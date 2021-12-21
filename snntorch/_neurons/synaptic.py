import torch
import torch.nn as nn
from .lif import *


class Synaptic(LIF):
    """
    2nd order leaky integrate and fire neuron model accounting for synaptic conductance.
    The synaptic current jumps upon spike arrival, which causes a jump in membrane potential.
    Synaptic current and membrane potential decay exponentially with rates of alpha and beta, respectively.
    For :math:`U[T] > U_{\\rm thr} ⇒ S[T+1] = 1`.

    If `reset_mechanism = "subtract"`, then :math:`U[t+1]` will have `threshold` subtracted from it whenever the neuron emits a spike:

    .. math::

            I_{\\rm syn}[t+1] = αI_{\\rm syn}[t] + I_{\\rm in}[t+1] \\\\
            U[t+1] = βU[t] + I_{\\rm syn}[t+1] - RU_{\\rm thr}

    If `reset_mechanism = "zero"`, then :math:`U[t+1]` will be set to `0` whenever the neuron emits a spike:

    .. math::

            I_{\\rm syn}[t+1] = αI_{\\rm syn}[t] + I_{\\rm in}[t+1] \\\\
            U[t+1] = βU[t] + I_{\\rm syn}[t+1] - R(βU[t] + I_{\\rm syn}[t+1])

    * :math:`I_{\\rm syn}` - Synaptic current
    * :math:`I_{\\rm in}` - Input current
    * :math:`U` - Membrane potential
    * :math:`U_{\\rm thr}` - Membrane threshold
    * :math:`R` - Reset mechanism: if active, :math:`R = 1`, otherwise :math:`R = 0`
    * :math:`α` - Synaptic current decay rate
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
                self.lif1 = snn.Synaptic(alpha=alpha, beta=beta)
                self.fc2 = nn.Linear(num_hidden, num_outputs)
                self.lif2 = snn.Synaptic(alpha=alpha, beta=beta)

            def forward(self, x, syn1, mem1, spk1, syn2, mem2):
                cur1 = self.fc1(x)
                spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
                cur2 = self.fc2(spk1)
                spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
                return syn1, mem1, spk1, syn2, mem2, spk2


    For further reading, see:

    *R. B. Stein (1965) A theoretical analysis of neuron variability. Biophys. J. 5, pp. 173-194.*

    *R. B. Stein (1967) Some models of neuronal variability. Biophys. J. 7. pp. 37-68.*"""

    def __init__(
        self,
        alpha,
        beta,
        threshold=1.0,
        spike_grad=None,
        init_hidden=False,
        inhibition=False,
        learn_alpha=False,
        learn_beta=False,
        learn_threshold=False,
        reset_mechanism="subtract",
        output=False,
    ):
        super(Synaptic, self).__init__(
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

        if not isinstance(alpha, torch.Tensor):
            alpha = torch.as_tensor(alpha)
        if learn_alpha:
            self.alpha = nn.Parameter(alpha)
        else:
            self.register_buffer("alpha", alpha)

        if self.init_hidden:
            self.syn, self.mem = self.init_synaptic()

    def forward(self, input_, syn=False, mem=False):

        if hasattr(syn, "init_flag") or hasattr(
            mem, "init_flag"
        ):  # only triggered on first-pass
            syn, mem = _SpikeTorchConv(syn, mem, input_=input_)
        elif mem is False and hasattr(self.mem, "init_flag"):  # init_hidden case
            self.syn, self.mem = _SpikeTorchConv(self.syn, self.mem, input_=input_)

        alpha = self.alpha.clamp(0, 1)
        beta = self.beta.clamp(0, 1)

        if not self.init_hidden:
            reset = self.mem_reset(mem)

            syn = alpha * syn + input_

            if self.reset_mechanism == "subtract":
                mem = beta * mem + syn - reset * self.threshold

            elif self.reset_mechanism == "zero":
                mem = beta * mem + syn - reset * (beta * mem + syn)

            if self.inhibition:
                spk = self.fire_inhibition(mem.size(0), mem)
            else:
                spk = self.fire(mem)

            return spk, syn, mem

        # intended for truncated-BPTT where instance variables are hidden states
        if self.init_hidden and not mem and not syn:
            self.reset = self.mem_reset(self.mem)

            self.syn = alpha * self.syn + input_

            if self.reset_mechanism == "subtract":
                self.mem = beta * self.mem + self.syn - self.reset * self.threshold

            elif self.reset_mechanism == "zero":
                self.mem = (
                    beta * self.mem
                    + self.syn
                    - self.reset * (beta * self.mem + self.syn)
                )

            if self.inhibition:
                self.spk = self.fire_inhibition(self.mem.size(0), self.mem)
            else:
                self.spk = self.fire(self.mem)

            if self.output:
                return self.spk, self.syn, self.mem
            else:
                return self.spk

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Synaptic):
                cls.instances[layer].syn.detach_()
                cls.instances[layer].mem.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Synaptic):
                cls.instances[layer].syn = _SpikeTensor(init_flag=False)
                cls.instances[layer].mem = _SpikeTensor(init_flag=False)

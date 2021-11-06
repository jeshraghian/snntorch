import torch
import torch.nn as nn
import numpy as np
from snntorch import utils
from ._version import __version__

#### Note: when adding new neurons, be sure to update neurons_dict in backprop.py

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float


class LIF(nn.Module):
    """Parent class for leaky integrate and fire neuron models."""

    instances = []
    """Each :mod:`snntorch.LIF` neuron (e.g., :mod:`snntorch.Synaptic`) will populate the :mod:`snntorch.LIF.instances` list with a new entry.
    The list is used to initialize and clear neuron states when the argument `init_hidden=True`."""

    def __init__(
        self,
        beta,
        threshold=1.0,
        spike_grad=None,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        reset_mechanism="subtract",
        output=False,
    ):
        super(LIF, self).__init__()
        LIF.instances.append(self)

        # self.threshold = threshold
        self.init_hidden = init_hidden
        self.inhibition = inhibition
        self.reset_mechanism = reset_mechanism
        self.output = output

        # TODO: this way, people can provide their own
        # 1) shape (one constant per layer or one per neuron)
        # 2) initial distribution
        if not isinstance(beta, torch.Tensor):
            beta = torch.as_tensor(beta)  # TODO: or .tensor() if no copy
        if learn_beta:
            self.beta = nn.Parameter(beta)
        else:
            self.register_buffer("beta", beta)

        if not isinstance(threshold, torch.Tensor):
            threshold = torch.as_tensor(threshold)  # TODO: or .tensor() if no copy
        self.register_buffer("threshold", threshold)

        if spike_grad is None:
            self.spike_grad = self.Heaviside.apply
        else:
            self.spike_grad = spike_grad

        if reset_mechanism != "subtract" and reset_mechanism != "zero":
            raise ValueError(
                "reset_mechanism must be set to either 'subtract' or 'zero'."
            )

    def fire(self, mem):
        """Generates spike if mem > threshold.
        Returns spk."""
        mem_shift = mem - self.threshold
        spk = self.spike_grad(mem_shift)

        return spk

    def fire_inhibition(self, batch_size, mem):
        """Generates spike if mem > threshold, only for the largest membrane. All others neurons will be inhibited for that time step.
        Returns spk."""
        mem_shift = mem - self.threshold
        index = torch.argmax(mem_shift, dim=1)
        spk_tmp = self.spike_grad(mem_shift)

        mask_spk1 = torch.zeros_like(spk_tmp)
        mask_spk1[torch.arange(batch_size), index] = 1
        spk = spk_tmp * mask_spk1
        # reset = spk.clone().detach()

        return spk

    def mem_reset(self, mem):
        """Generates detached reset signal if mem > threshold.
        Returns reset."""
        mem_shift = mem - self.threshold
        reset = self.spike_grad(mem_shift).clone().detach()

        return reset

    @classmethod
    def init(cls):
        """Removes all items from :mod:`snntorch.LIF.instances` when called."""
        cls.instances = []

    @staticmethod
    def init_leaky():
        """
        Used to initialize mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert the hidden states to the same as the input.
        """
        # print(f"init_leaky executing")
        mem = _SpikeTensor(init_flag=False)

        return mem

    @staticmethod
    def init_synaptic():
        """Used to initialize syn and mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert the hidden states to the same as the input.
        """

        syn = _SpikeTensor(init_flag=False)
        mem = _SpikeTensor(init_flag=False)

        return syn, mem

    @staticmethod
    def init_stein():
        """Used to initialize syn and mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert the hidden states to the same as the input.
        """
        return LIF.init_synaptic()

    @staticmethod
    def init_lapicque():
        """
        Used to initialize mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert the hidden states to the same as the input.
        """

        return LIF.init_leaky()

    @staticmethod
    def init_alpha():
        """Used to initialize syn_exc, syn_inh and mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert the hidden states to the same as the input.
        """
        syn_exc = _SpikeTensor(init_flag=False)
        syn_inh = _SpikeTensor(init_flag=False)
        mem = _SpikeTensor(init_flag=False)

        return syn_exc, syn_inh, mem

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
        """Default spiking function for neuron.

        **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

        **Backward pass:** Heaviside step function shifted.

        .. math::

            \\frac{∂S}{∂U}=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

        Although the backward pass is clearly not the analytical solution of the forward pass, this assumption holds true on the basis that a reset necessarily occurs after a spike is generated when :math:`U ≥ U_{\\rm thr}`."""

        @staticmethod
        def forward(ctx, input_):
            out = (input_ > 0).float()
            ctx.save_for_backward(out)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            (out,) = ctx.saved_tensors
            grad = grad_output * out
            return grad


class _SpikeTensor(torch.Tensor):
    """Inherits from torch.Tensor with additional attributes.
    ``init_flag`` is set at the time of initialization.
    When called in the forward function of any neuron, they are parsed and replaced with a torch.Tensor variable.
    """

    @staticmethod
    def __new__(cls, *args, init_flag=False, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(
        self,
        *args,
        init_flag=True,
    ):
        # super().__init__() # optional
        self.init_flag = init_flag


def _SpikeTorchConv(*args, input_):
    """Convert SpikeTensor to torch.Tensor of the same size as ``input_``."""

    states = []
    # if len(input_.size()) == 0:
    #     _batch_size = 1  # assume batch_size=1 if 1D input
    # else:
    #     _batch_size = input_.size(0)
    if (
        len(args) == 1 and type(args) is not tuple
    ):  # if only one hidden state, make it iterable
        args = (args,)
    for arg in args:
        arg = torch.Tensor(arg)  # wash away the SpikeTensor class
        arg = torch.zeros_like(input_, requires_grad=True)
        states.append(arg)
    if len(states) == 1:  # otherwise, list isn't unpacked
        return states[0]

    return states


# # Neuron Models
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


class Lapicque(LIF):
    """
    An extension of Lapicque's experimental comparison between extracellular nerve fibers and an RC circuit.
    It is qualitatively equivalent to :code:`Leaky` but defined using RC circuit parameters.
    Input stimulus is integrated by membrane potential which decays exponentially with a rate of beta.
    For :math:`U[T] > U_{\\rm thr} ⇒ S[T+1] = 1`.

    If `reset_mechanism = "subtract"`, then :math:`U[t+1]` will have `threshold` subtracted from it whenever the neuron emits a spike:

    .. math::

            U[t+1] = I_{\\rm in}[t+1] (\\frac{T}{C}) + (1- \\frac{T}{\\tau})U[t] - RU_{\\rm thr}

    If `reset_mechanism = "zero"`, then :math:`U[t+1]` will be set to `0` whenever the neuron emits a spike:

    .. math::

            U[t+1] = I_{\\rm in}[t+1] (\\frac{T}{\\tau}) + (1- \\frac{T}{\\tau})U[t] - R(I_{\\rm in}[t+1] (\\frac{T}{C}) + (1- \\frac{T}{\\tau})U[t])

    * :math:`I_{\\rm in}` - Input current
    * :math:`U` - Membrane potential
    * :math:`U_{\\rm thr}` - Membrane threshold
    * :math:`T`- duration of each time step
    * :math:`R` - Reset mechanism: if active, :math:`R = 1`, otherwise :math:`R = 0`
    * :math:`β` - Membrane potential decay rate

    Alternatively, the membrane potential decay rate β can be specified instead:

    .. math::

            β = e^{-1/RC}

    * :math:`β` - Membrane potential decay rate
    * :math:`R` - Parallel resistance of passive membrane (note: distinct from the reset :math:`R`)
    * :math:`C` - Parallel capacitance of passive membrane

    * If only β is defined, then R will default to 1, and C will be inferred.
    * If RC is defined, β will be automatically calculated.
    * If (β and R) or (β and C) are defined, the missing variable will be automatically calculated.

    * Note that β, R and C are treated as `hard-wired' physically plausible parameters, and are therefore not learnable. For a single-state neuron with a learnable decay rate β, use `snn.Leaky` instead.

    Example::

        import torch
        import torch.nn as nn
        import snntorch as snn

        beta = 0.5

        R = 1
        C = 1.44

        # Define Network
        class Net(nn.Module):
            def __init__(self):
                super().__init__()

                # initialize layers
                self.fc1 = nn.Linear(num_inputs, num_hidden)
                self.lif1 = snn.Lapicque(beta=beta)
                self.fc2 = nn.Linear(num_hidden, num_outputs)
                self.lif2 = snn.Lapicque(R=R, C=C)  # lif1 and lif2 are approximately equivalent

            def forward(self, x, mem1, spk1, mem2):
                cur1 = self.fc1(x)
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                return mem1, spk1, mem2, spk2


    For further reading, see:

    *L. Lapicque (1907) Recherches quantitatives sur l'excitation électrique des nerfs traitée comme une polarisation. J. Physiol. Pathol. Gen. 9, pp. 620-635. (French)*

    *N. Brunel and M. C. Van Rossum (2007) Lapicque's 1907 paper: From frogs to integrate-and-fire. Biol. Cybern. 97, pp. 337-339. (English)*

    Although Lapicque did not formally introduce this as an integrate-and-fire neuron model, we pay homage to his discovery of an RC circuit mimicking the dynamics of synaptic current."""

    def __init__(
        self,
        beta=False,
        R=False,
        C=False,
        time_step=1,
        threshold=1.0,
        spike_grad=None,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        reset_mechanism="subtract",
        output=False,
    ):
        super(Lapicque, self).__init__(
            beta,
            threshold,
            spike_grad,
            init_hidden,
            inhibition,
            learn_beta,
            reset_mechanism,
            output,
        )

        self._lapicque_cases(time_step, beta, R, C)
        if self.init_hidden:
            self.mem = self.init_lapicque()

    def forward(self, input_, mem=False):

        R = self.R
        C = self.C

        if hasattr(mem, "init_flag"):  # only triggered on first-pass
            mem = _SpikeTorchConv(mem, input_=input_)
        elif mem is False and hasattr(self.mem, "init_flag"):  # init_hidden case
            self.mem = _SpikeTorchConv(self.mem, input_=input_)

        if not self.init_hidden:
            reset = self.mem_reset(mem)

            if self.reset_mechanism == "subtract":
                mem = (
                    input_ * R * (1 / (R * C)) * self.time_step
                    + (1 - (self.time_step / (R * C))) * mem
                    - reset * self.threshold
                )

            elif self.reset_mechanism == "zero":
                mem = (
                    input_ * R * (1 / (R * C)) * self.time_step
                    + (1 - (self.time_step / (R * C))) * mem
                    - reset
                    * (
                        (
                            input_ * R * (1 / (R * C)) * self.time_step
                            + (1 - (self.time_step / (R * C))) * mem
                        )
                    )
                )

            if self.inhibition:
                spk = self.fire_inhibition(mem.size(0), mem)
            else:
                spk = self.fire(mem)

            return spk, mem

        # intended for truncated-BPTT where instance variables are hidden states
        if self.init_hidden and not mem:
            self.reset = self.mem_reset(self.mem)

            if self.reset_mechanism == "subtract":
                self.mem = (
                    input_ * R * (1 / (R * C)) * self.time_step
                    + (1 - (self.time_step / (R * C))) * self.mem
                    - self.reset * self.threshold
                )

            elif self.reset_mechanism == "zero":
                self.mem = (
                    input_ * R * (1 / (R * C)) * self.time_step
                    + (1 - (self.time_step / (R * C))) * self.mem
                    - self.reset
                    * (
                        (
                            input_ * R * (1 / (R * C)) * self.time_step
                            + (1 - (self.time_step / (R * C))) * self.mem
                        )
                    )
                )

            if self.inhibition:
                self.spk = self.fire_inhibition(self.mem.size(0), self.mem)
            else:
                self.spk = self.fire(self.mem)

            if self.output:
                return self.spk, self.mem
            else:
                return self.spk

    def _lapicque_cases(self, time_step, beta, R, C):
        if not isinstance(time_step, torch.Tensor):
            time_step = torch.as_tensor(time_step)
        self.register_buffer("time_step", time_step)

        if not self.beta and not (R and C):
            raise ValueError(
                "Either beta or 2 of beta, R and C must be specified as an input argument."
            )

        elif not self.beta and (bool(R) ^ bool(C)):
            raise ValueError(
                "Either beta or 2 of beta, R and C must be specified as an input argument."
            )

        elif (R and C) and not self.beta:
            beta = torch.exp(torch.ones(1) * (-self.time_step / (R * C)))

            self.register_buffer("beta", beta)

            if not isinstance(R, torch.Tensor):
                R = torch.as_tensor(R)
            self.register_buffer("R", R)
            if not isinstance(C, torch.Tensor):
                C = torch.as_tensor(C)
            self.register_buffer("C", C)

        elif self.beta and not (R or C):
            R = torch.as_tensor(1)
            self.register_buffer("R", R)
            C = self.time_step / (R * torch.log(1 / self.beta))
            self.register_buffer("C", C)
            if not isinstance(R, torch.Tensor):
                self.register_buffer("beta", self.beta)

        elif self.beta and R and not C:
            C = self.time_step / (R * torch.log(1 / self.beta))
            self.register_buffer("C", C)
            if not isinstance(R, torch.Tensor):
                R = torch.as_tensor(R)
            self.register_buffer("R", R)
            self.register_buffer("beta", self.beta)

        elif self.beta and C and not R:
            if not isinstance(C, torch.Tensor):
                C = torch.as_tensor(C)
            self.register_buffer("C", C)
            self.register_buffer("beta", self.beta)
            R = self.time_step / (C * torch.log(1 / self.beta))
            self.register_buffer("R", R)

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Lapicque):
                cls.instances[layer].mem.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Lapicque):
                cls.instances[layer].mem = _SpikeTensor(init_flag=False)


class Alpha(LIF):
    """
    A variant of the leaky integrate and fire neuron where membrane potential follows an alpha function.
    The time course of the membrane potential response depends on a combination of exponentials.
    In general, this causes the change in membrane potential to experience a delay with respect to an input spike.
    For :math:`U[T] > U_{\\rm thr} ⇒ S[T+1] = 1`.

    .. warning:: For a positive input current to induce a positive membrane response, ensure :math:`α > β`.

    If `reset_mechanism = "subtract"`, then :math:`I_{\\rm exc}, I_{\\rm inh}` will both have `threshold` subtracted from them whenever the neuron emits a spike:

    .. math::

            I_{\\rm exc}[t+1] = (αI_{\\rm exc}[t] + I_{\\rm in}[t+1]) - R(αI_{\\rm exc}[t] + I_{\\rm in}[t+1]) \\\\
            I_{\\rm inh}[t+1] = (βI_{\\rm inh}[t] - I_{\\rm in}[t+1]) - R(βI_{\\rm inh}[t] - I_{\\rm in}[t+1]) \\\\
            U[t+1] = τ_{\\rm SRM}(I_{\\rm exc}[t+1] + I_{\\rm inh}[t+1])

    If `reset_mechanism = "zero"`, then :math:`I_{\\rm exc}, I_{\\rm inh}` will both be set to `0` whenever the neuron emits a spike:

    .. math::

            I_{\\rm exc}[t+1] = (αI_{\\rm exc}[t] + I_{\\rm in}[t+1]) - R(αI_{\\rm exc}[t] + I_{\\rm in}[t+1]) \\\\
            I_{\\rm inh}[t+1] = (βI_{\\rm inh}[t] - I_{\\rm in}[t+1]) - R(βI_{\\rm inh}[t] - I_{\\rm in}[t+1]) \\\\
            U[t+1] = τ_{\\rm SRM}(I_{\\rm exc}[t+1] + I_{\\rm inh}[t+1])

    * :math:`I_{\\rm exc}` - Excitatory current
    * :math:`I_{\\rm inh}` - Inhibitory current
    * :math:`I_{\\rm in}` - Input current
    * :math:`U` - Membrane potential
    * :math:`U_{\\rm thr}` - Membrane threshold
    * :math:`R` - Reset mechanism, :math:`R = 1` if spike occurs, otherwise :math:`R = 0`
    * :math:`α` - Excitatory current decay rate
    * :math:`β` - Inhibitory current decay rate
    * :math:`τ_{\\rm SRM} = \\frac{log(α)}{log(β)} - log(α) + 1`

    Example::

        import torch
        import torch.nn as nn
        import snntorch as snn

        alpha = 0.9
        beta = 0.8

        # Define Network
        class Net(nn.Module):
            def __init__(self):
                super().__init__()

                # initialize layers
                self.fc1 = nn.Linear(num_inputs, num_hidden)
                self.lif1 = snn.Alpha(alpha=alpha, beta=beta)
                self.fc2 = nn.Linear(num_hidden, num_outputs)
                self.lif2 = snn.Alpha(alpha=alpha, beta=beta)

            def forward(self, x, syn_exc1, syn_inh1, mem1, spk1, syn_exc2, syn_inh2, mem2):
                cur1 = self.fc1(x)
                spk1, syn_exc1, syn_inh1, mem1 = self.lif1(cur1, syn_exc1, syn_inh1, mem1)
                cur2 = self.fc2(spk1)
                spk2, syn_exc2, syn_inh2, mem2 = self.lif2(cur2, syn_exc2, syn_inh2, mem2)
                return syn_exc1, syn_inh1, mem1, spk1, syn_exc2, syn_inh2, mem2, spk2


    """

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
        reset_mechanism="subtract",
        output=False,
    ):
        super(Alpha, self).__init__(
            beta,
            threshold,
            spike_grad,
            init_hidden,
            inhibition,
            learn_beta,
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
            self.syn_exc, self.syn_inh, self.mem = self.init_alpha()

        if (self.alpha <= self.beta).any():
            raise ValueError("alpha must be greater than beta.")

        if (self.beta == 1).any():
            raise ValueError(
                "beta cannot be '1' otherwise ZeroDivisionError occurs: tau_srm = log(alpha)/log(beta) - log(alpha) + 1"
            )

        # if reset_mechanism == "subtract":
        #     self.mem_residual = False

    def forward(self, input_, syn_exc=False, syn_inh=False, mem=False):

        if (
            hasattr(syn_exc, "init_flag")
            or hasattr(syn_inh, "init_flag")
            or hasattr(mem, "init_flag")
        ):  # only triggered on first-pass
            syn_exc, syn_inh, mem = _SpikeTorchConv(
                syn_exc, syn_inh, mem, input_=input_
            )
        elif mem is False and hasattr(self.mem, "init_flag"):  # init_hidden case
            self.syn_exc, self.syn_inh, self.mem = _SpikeTorchConv(
                self.syn_exc, self.syn_inh, self.mem, input_=input_
            )

        alpha = self.alpha.clamp(0, 1)
        beta = self.beta.clamp(0, 1)
        tau_srm = torch.log(alpha) / (torch.log(beta) - torch.log(alpha)) + 1

        # if hidden states are passed externally
        if not self.init_hidden:
            reset = self.mem_reset(mem)

            # if neuron fires, subtract threhsold from neuron
            if self.reset_mechanism == "subtract":

                # if self.mem_residual is False:
                #     self.mem_residual = torch.zeros_like(mem)

                syn_exc = (alpha * syn_exc + input_) - reset * (
                    alpha * syn_exc + input_
                )
                syn_inh = (beta * syn_inh - input_) - reset * (beta * syn_inh - input_)
                #  #The residual of (mem - threshold) decays separately
                # self.mem_residual = reset * (mem - self.threshold) + (
                #     self.mem_residual / self.tau_srm
                # )
                mem = tau_srm * (syn_exc + syn_inh)  # + self.mem_residual

            # if neuron fires, reset membrane to zero
            elif self.reset_mechanism == "zero":
                syn_exc = (alpha * syn_exc + input_) - reset * (
                    alpha * syn_exc + input_
                )
                syn_inh = (beta * syn_inh - input_) - reset * (beta * syn_inh - input_)
                mem = tau_srm * (syn_exc + syn_inh)

            if self.inhibition:
                spk = self.fire_inhibition(mem.size(0), mem)

            else:
                spk = self.fire(mem)

            return spk, syn_exc, syn_inh, mem

        # if hidden states and outputs are instance variables
        if self.init_hidden and not mem:

            self.reset = self.mem_reset(self.mem)

            # if neuron fires, subtract threhsold from neuron
            if self.reset_mechanism == "subtract":

                # if self.mem_residual is False:
                #     self.mem_residual = torch.zeros_like(self.mem)

                self.syn_exc = (alpha * self.syn_exc + input_) - self.reset * (
                    alpha * self.syn_exc + input_
                )
                syn_inh = (beta * self.syn_inh - input_) - self.reset * (
                    beta * self.syn_inh - input_
                )
                # # The residual of (mem - threshold) decays separately
                # self.mem_residual = self.reset * (self.mem - self.threshold) + (
                #     self.mem_residual / self.tau_srm
                # )
                self.mem = tau_srm * (
                    self.syn_exc + self.syn_inh
                )  # + self.mem_residual

            # if neuron fires, reset membrane to zero
            elif self.reset_mechanism == "zero":

                syn_exc = (alpha * syn_exc + input_) - self.reset * (
                    alpha * syn_exc + input_
                )
                syn_inh = (beta * syn_inh - input_) - self.reset * (
                    beta * syn_inh - input_
                )
                self.mem = tau_srm * (syn_exc + syn_inh)

            if self.inhibition:
                self.spk = self.fire_inhibition(self.mem.size(0), self.mem)

            else:
                self.spk = self.fire(self.mem)

            if self.output:
                return self.spk, self.syn_exc, self.syn_inh, self.mem
            else:
                return self.spk

    # cool forward function that resulted in burst firing - worth exploring

    # def forward(self, input_, syn_exc, syn_inh, mem):
    #     mem_shift = mem - self.threshold
    #     spk = self.spike_grad(mem_shift).to(device)
    #     reset = torch.zeros_like(mem)
    #     spk_idx = (mem_shift > 0)
    #     reset[spk_idx] = torch.ones_like(mem)[spk_idx]
    #
    #     syn_exc = self.alpha * syn_exc + input_
    #     syn_inh = self.beta * syn_inh - input_
    #     mem = self.tau_srm * (syn_exc + syn_inh) - reset

    # return spk, syn_exc, syn_inh, mem

    @classmethod
    def detach_hidden(cls):
        """Used to detach hidden states from the current graph.
        Intended for use in truncated backpropagation through
        time where hidden state variables are instance variables."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Alpha):
                cls.instances[layer].syn_exc.detach_()
                cls.instances[layer].syn_inh.detach_()
                cls.instances[layer].mem.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Alpha):
                cls.instances[layer].syn_exc = _SpikeTensor(init_flag=False)
                cls.instances[layer].syn_inh = _SpikeTensor(init_flag=False)
                cls.instances[layer].mem = _SpikeTensor(init_flag=False)


####### Deprecated / Renamed
class Stein(LIF):  # see: Synaptic(LIF)
    """
    **Note: this neuron model has been deprecated. Please use `Synaptic(LIF) instead.
    Stein's model of the leaky integrate and fire neuron.
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
                self.lif1 = snn.Stein(alpha=alpha, beta=beta)
                self.fc2 = nn.Linear(num_hidden, num_outputs)
                self.lif2 = snn.Stein(alpha=alpha, beta=beta)

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
        reset_mechanism="subtract",
        output=False,
    ):
        super(Stein, self).__init__(
            beta,
            threshold,
            spike_grad,
            init_hidden,
            inhibition,
            learn_beta,
            reset_mechanism,
            output,
        )

        print(
            "`Stein` has been deprecated and will be removed in a future version. Use `Synaptic` instead."
        )

        self.alpha = alpha

        if self.init_hidden:
            self.syn, self.mem = self.init_stein()

    def forward(self, input_, syn, mem=False):

        if hasattr(syn, "init_flag") or hasattr(
            mem, "init_flag"
        ):  # only triggered on first-pass
            syn, mem = _SpikeTorchConv(syn, mem, input_=input_)

        if not self.init_hidden:
            if self.inhibition:
                spk, reset = self.fire_inhibition(mem.size(0), mem)
            else:
                spk, reset = self.fire(mem)
            syn = self.alpha * syn + input_

            if self.reset_mechanism == "subtract":
                mem = self.beta * mem + syn - reset * self.threshold

            elif self.reset_mechanism == "zero":
                mem = self.beta * mem + syn - reset * (self.beta * mem + syn)

            return spk, syn, mem

        # intended for truncated-BPTT where instance variables are hidden states
        if self.init_hidden and not mem:
            if self.inhibition:
                self.spk, self.reset = self.fire_inhibition(self.mem.size(0), self.mem)
            else:
                self.spk, self.reset = self.fire(self.mem)

            self.syn = self.alpha * self.syn + input_

            if self.reset_mechanism == "subtract":
                self.mem = self.beta * self.mem + self.syn - self.reset * self.threshold

            elif self.reset_mechanism == "zero":
                self.mem = (
                    self.beta * self.mem
                    + self.syn
                    - self.reset * (self.beta * self.mem + self.syn)
                )

            if self.output:  # read-out layer returns output+states
                return self.spk, self.syn, self.mem
            else:
                return self.spk

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states and detaches them from the current graph.
        Intended for use in truncated backpropagation through time where hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Stein):
                cls.instances[layer].syn.detach_()
                cls.instances[layer].mem.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Stein):
                cls.instances[layer].syn = _SpikeTensor(init_flag=False)
                cls.instances[layer].mem = _SpikeTensor(init_flag=False)

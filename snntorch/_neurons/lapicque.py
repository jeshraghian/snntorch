import torch
from .neurons import *


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

    Although Lapicque did not formally introduce this as an integrate-and-fire neuron model, we pay homage to his discovery of an RC circuit mimicking the dynamics of synaptic current.



    :param beta: RC potential decay rate. Clipped between 0 and 1 during the forward-pass. May be a single-valued tensor (i.e., equal decay rate for all neurons in a layer), or multi-valued (one weight per neuron).
    :type beta: float or torch.tensor, Optional

    :param R: Resistance of RC circuit
    :type R: int or torch.tensor, Optional

    :param C: Capacitance of RC circuit
    :type C: int or torch.tensor, Optional

    :param time_step: time step precision. Defaults to 1
    :type time_step: float, Optional

    :param threshold: Threshold for :math:`mem` to reach in order to generate a spike `S=1`. Defaults to 1
    :type threshold: float, optional

    :param spike_grad: Surrogate gradient for the term dS/dU. Defaults to None (corresponds to Heaviside surrogate gradient. See `snntorch.surrogate` for more options)
    :type spike_grad: surrogate gradient function from snntorch.surrogate, optional

    :param init_hidden: Instantiates state variables as instance variables. Defaults to False
    :type init_hidden: bool, optional

    :param inhibition: If `True`, suppresses all spiking other than the neuron with the highest state. Defaults to False
    :type inhibition: bool, optional

    :param learn_beta: Option to enable learnable beta. Defaults to False
    :type learn_beta: bool, optional

    :param learn_threshold: Option to enable learnable threshold. Defaults to False
    :type learn_threshold: bool, optional

    :param reset_mechanism: Defines the reset mechanism applied to :math:`mem` each time the threshold is met. Reset-by-subtraction: "subtract", reset-to-zero: "zero, none: "none". Defaults to "none"
    :type reset_mechanism: str, optional

    :param state_quant: If specified, hidden state :math:`mem` is quantized to a valid state for the forward pass. Defaults to False
    :type state_quant: quantization function from snntorch.quant, optional

    :param output: If `True` as well as `init_hidden=True`, states are returned when neuron is called. Defaults to False
    :type output: bool, optional


    Inputs: \\input_, mem_0
        - **input_** of shape `(batch, input_size)`: tensor containing input features
        - **mem_0** of shape `(batch, input_size)`: tensor containing the initial membrane potential for each element in the batch.

    Outputs: spk, mem_1
        - **spk** of shape `(batch, input_size)`: tensor containing the output spikes.
        - **mem_1** of shape `(batch, input_size)`: tensor containing the next membrane potential for each element in the batch

    Learnable Parameters:
        - **Lapcique.beta** (torch.Tensor) - optional learnable weights must be manually passed in, of shape `1` or (input_size).
        - **Lapcique.threshold** (torch.Tensor) - optional learnable thresholds must be manually passed in, of shape `1` or`` (input_size).

    """

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
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
    ):
        super(Lapicque, self).__init__(
            beta,
            threshold,
            spike_grad,
            init_hidden,
            inhibition,
            learn_beta,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
        )

        self._lapicque_cases(time_step, beta, R, C)

        if self.init_hidden:
            self.mem = self.init_lapicque()
            self.state_fn = self._build_state_function_hidden
        else:
            self.state_fn = self._build_state_function

    def forward(self, input_, mem=False):

        if hasattr(mem, "init_flag"):  # only triggered on first-pass
            mem = _SpikeTorchConv(mem, input_=input_)
        elif mem is False and hasattr(self.mem, "init_flag"):  # init_hidden case
            self.mem = _SpikeTorchConv(self.mem, input_=input_)

        if not self.init_hidden:
            self.reset = self.mem_reset(mem)
            mem = self.state_fn(input_, mem)

            if self.state_quant:
                mem = self.state_quant(mem)

            if self.inhibition:
                spk = self.fire_inhibition(mem.size(0), mem)
            else:
                spk = self.fire(mem)

            return spk, mem

        # intended for truncated-BPTT where instance variables are hidden states
        if self.init_hidden:
            self._lapicque_forward_cases(mem)
            self.reset = self.mem_reset(self.mem)
            self.mem = self.state_fn(input_)

            if self.state_quant:
                self.mem = self.state_quant(self.mem)

            if self.inhibition:
                self.spk = self.fire_inhibition(self.mem.size(0), self.mem)
            else:
                self.spk = self.fire(self.mem)

            if self.output:
                return self.spk, self.mem
            else:
                return self.spk

    def _base_state_function(self, input_, mem):
        base_fn = (
            input_ * self.R * (1 / (self.R * self.C)) * self.time_step
            + (1 - (self.time_step / (self.R * self.C))) * mem
        )
        return base_fn

    def _build_state_function(self, input_, mem):
        if self.reset_mechanism_val == 0:  # reset by subtraction
            state_fn = (
                self._base_state_function(input_, mem) - self.reset * self.threshold
            )
        elif self.reset_mechanism_val == 1:  # reset to zero
            state_fn = self._base_state_function(
                input_, mem
            ) - self.reset * self._base_state_function(input_, mem)
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            state_fn = self._base_state_function(input_, mem)
        return state_fn

    def _base_state_function_hidden(self, input_):
        base_fn = (
            input_ * self.R * (1 / (self.R * self.C)) * self.time_step
            + (1 - (self.time_step / (self.R * self.C))) * self.mem
        )
        return base_fn

    def _build_state_function_hidden(self, input_):
        if self.reset_mechanism_val == 0:  # reset by subtraction
            state_fn = (
                self._base_state_function_hidden(input_) - self.reset * self.threshold
            )
        elif self.reset_mechanism_val == 1:  # reset to zero
            state_fn = self._base_state_function_hidden(
                input_
            ) - self.reset * self._base_state_function_hidden(input_)
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            state_fn = self._base_state_function_hidden(input_)
        return state_fn

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

    def _lapicque_forward_cases(self, mem):
        if mem is not False:
            raise TypeError(
                "When `init_hidden=True`, Lapicque expects 1 input argument."
            )

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

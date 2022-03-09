import torch
import torch.nn as nn
from .neurons import *


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



    :param alpha: synaptic current decay rate. Clipped between 0 and 1 during the forward-pass. May be a single-valued tensor (i.e., equal decay rate for all neurons in a layer), or multi-valued (one weight per neuron).
    :type alpha: float or torch.tensor

    :param beta: membrane potential decay rate. Clipped between 0 and 1 during the forward-pass. May be a single-valued tensor (i.e., equal decay rate for all neurons in a layer), or multi-valued (one weight per neuron).
    :type beta: float or torch.tensor

    :param threshold: Threshold for :math:`mem` to reach in order to generate a spike `S=1`. Defaults to 1
    :type threshold: float, optional

    :param spike_grad: Surrogate gradient for the term dS/dU. Defaults to None (corresponds to Heaviside surrogate gradient. See `snntorch.surrogate` for more options)
    :type spike_grad: surrogate gradient function from snntorch.surrogate, optional

    :param init_hidden: Instantiates state variables as instance variables. Defaults to False
    :type init_hidden: bool, optional

    :param inhibition: If `True`, suppresses all spiking other than the neuron with the highest state. Defaults to False
    :type inhibition: bool, optional

    :param learn_alpha: Option to enable learnable alpha. Defaults to False
    :type learn_alpha: bool, optional

    :param learn_beta: Option to enable learnable beta. Defaults to False
    :type learn_beta: bool, optional

    :param learn_threshold: Option to enable learnable threshold. Defaults to False
    :type learn_threshold: bool, optional

    :param reset_mechanism: Defines the reset mechanism applied to :math:`mem` each time the threshold is met. Reset-by-subtraction: "subtract", reset-to-zero: "zero, none: "none". Defaults to "subtract"
    :type reset_mechanism: str, optional

    :param state_quant: If specified, hidden states :math:`mem` and :math:`syn` are quantized to a valid state for the forward pass. Defaults to False
    :type state_quant: quantization function from snntorch.quant, optional

    :param output: If `True` as well as `init_hidden=True`, states are returned when neuron is called. Defaults to False
    :type output: bool, optional


    Inputs: \\input_, syn_0, mem_0
        - **input_** of shape `(batch, input_size)`: tensor containing input features
        - **syn_0** of shape `(batch, input_size)`: tensor containing input features
        - **mem_0** of shape `(batch, input_size)`: tensor containing the initial membrane potential for each element in the batch.

    Outputs: spk, syn_1, mem_1
        - **spk** of shape `(batch, input_size)`: tensor containing the output spikes.
        - **syn_1** of shape `(batch, input_size)`: tensor containing the next synaptic current for each element in the batch
        - **mem_1** of shape `(batch, input_size)`: tensor containing the next membrane potential for each element in the batch

    Learnable Parameters:
        - **Synaptic.alpha** (torch.Tensor) - optional learnable weights must be manually passed in, of shape `1` or (input_size).
        - **Synaptic.beta** (torch.Tensor) - optional learnable weights must be manually passed in, of shape `1` or (input_size).
        - **Synaptic.threshold** (torch.Tensor) - optional learnable thresholds must be manually passed in, of shape `1` or`` (input_size).

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
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
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
            state_quant,
            output,
        )

        self._alpha_register_buffer(alpha, learn_alpha)

        if self.init_hidden:
            self.syn, self.mem = self.init_synaptic()
            self.state_fn = self._build_state_function_hidden
        else:
            self.state_fn = self._build_state_function

    def forward(self, input_, syn=False, mem=False):

        if hasattr(syn, "init_flag") or hasattr(
            mem, "init_flag"
        ):  # only triggered on first-pass
            syn, mem = _SpikeTorchConv(syn, mem, input_=input_)
        elif mem is False and hasattr(self.mem, "init_flag"):  # init_hidden case
            self.syn, self.mem = _SpikeTorchConv(self.syn, self.mem, input_=input_)

        if not self.init_hidden:
            self.reset = self.mem_reset(mem)
            syn, mem = self.state_fn(input_, syn, mem)

            if self.state_quant:
                syn = self.state_quant(syn)
                mem = self.state_quant(mem)

            if self.inhibition:
                spk = self.fire_inhibition(mem.size(0), mem)
            else:
                spk = self.fire(mem)

            return spk, syn, mem

        # intended for truncated-BPTT where instance variables are hidden states
        if self.init_hidden:
            self._synaptic_forward_cases(mem, syn)
            self.reset = self.mem_reset(self.mem)
            self.syn, self.mem = self.state_fn(input_)

            if self.state_quant:
                self.syn = self.state_quant(self.syn)
                self.mem = self.state_quant(self.mem)

            if self.inhibition:
                self.spk = self.fire_inhibition(self.mem.size(0), self.mem)
            else:
                self.spk = self.fire(self.mem)

            if self.output:
                return self.spk, self.syn, self.mem
            else:
                return self.spk

    def _base_state_function(self, input_, syn, mem):
        base_fn_syn = self.alpha.clamp(0, 1) * syn + input_
        base_fn_mem = self.beta.clamp(0, 1) * mem + base_fn_syn
        return base_fn_syn, base_fn_mem

    def _base_state_reset_zero(self, input_, syn, mem):
        base_fn_syn = self.alpha.clamp(0, 1) * syn + input_
        base_fn_mem = self.beta.clamp(0, 1) * mem + base_fn_syn
        return 0, base_fn_mem

    def _build_state_function(self, input_, syn, mem):
        if self.reset_mechanism_val == 0:  # reset by subtraction
            state_fn = tuple(
                map(
                    lambda x, y: x - y,
                    self._base_state_function(input_, syn, mem),
                    (0, self.reset * self.threshold),
                )
            )
        elif self.reset_mechanism_val == 1:  # reset to zero
            state_fn = tuple(
                map(
                    lambda x, y: x - self.reset * y,
                    self._base_state_function(input_, syn, mem),
                    self._base_state_reset_zero(input_, syn, mem),
                )
            )
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            state_fn = self._base_state_function(input_, syn, mem)
        return state_fn

    def _base_state_function_hidden(self, input_):
        base_fn_syn = self.alpha.clamp(0, 1) * self.syn + input_
        base_fn_mem = self.beta.clamp(0, 1) * self.mem + input_
        return base_fn_syn, base_fn_mem

    def _base_state_reset_zero_hidden(self, input_):
        base_fn_syn = self.alpha.clamp(0, 1) * self.syn + input_
        base_fn_mem = self.beta.clamp(0, 1) * self.mem + base_fn_syn
        return 0, base_fn_mem

    def _build_state_function_hidden(self, input_):
        if self.reset_mechanism_val == 0:  # reset by subtraction
            state_fn = tuple(
                map(
                    lambda x, y: x - y,
                    self._base_state_function_hidden(input_),
                    (0, self.reset * self.threshold),
                )
            )
        elif self.reset_mechanism_val == 1:  # reset to zero
            state_fn = tuple(
                map(
                    lambda x, y: x - self.reset * y,
                    self._base_state_function_hidden(input_),
                    self._base_state_reset_zero_hidden(input_),
                )
            )
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            state_fn = self._base_state_function_hidden(input_)
        return state_fn

    def _alpha_register_buffer(self, alpha, learn_alpha):
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.as_tensor(alpha)
        if learn_alpha:
            self.alpha = nn.Parameter(alpha)
        else:
            self.register_buffer("alpha", alpha)

    def _synaptic_forward_cases(self, mem, syn):
        if mem is not False or syn is not False:
            raise TypeError(
                "When `init_hidden=True`, Synaptic expects 1 input argument."
            )

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

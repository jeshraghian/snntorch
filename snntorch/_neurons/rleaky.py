import torch
import torch.nn as nn

# from torch import functional as F
from .neurons import *


class RLeaky(LIF):
    """
    First-order recurrent leaky integrate-and-fire neuron model.
    Input is assumed to be a current injection appended to the voltage spike output.
    Membrane potential decays exponentially with rate beta.
    For :math:`U[T] > U_{\\rm thr} ⇒ S[T+1] = 1`.

    If `reset_mechanism = "subtract"`, then :math:`U[t+1]` will have `threshold` subtracted from it whenever the neuron emits a spike:

    .. math::

            U[t+1] = βU[t] + I_{\\rm in}[t+1] + V(S_{\\rm out}[t]) - RU_{\\rm thr}

    Where :math:`V(\\cdot)` acts either as a linear layer, a convolutional operator, or elementwise product on :math:`S_{\\rm out}`.

    * If `all_to_all = "True"` and `linear_features` is specified, then :math:`V(\\cdot)` acts as a recurrent linear layer of the same size as :math:`S_{\\rm out}`.
    * If `all_to_all = "True"` and `conv2d_channels` and `kernel_size` are specified, then :math:`V(\\cdot)` acts as a recurrent convlutional layer with padding to ensure the output matches the size of the input.
    * If `all_to_all = "False"`, then :math:`V(\\cdot)` acts as an elementwise multiplier with :math:`V`.

    If `reset_mechanism = "zero"`, then :math:`U[t+1]` will be set to `0` whenever the neuron emits a spike:

    .. math::

            U[t+1] = βU[t] + I_{\\rm syn}[t+1] +  V(S_{\\rm out}[t] - R(βU[t] + I_{\\rm in}[t+1] +  V(S_{\\rm out}[t])

    * :math:`I_{\\rm in}` - Input current
    * :math:`U` - Membrane potential
    * :math:`U_{\\rm thr}` - Membrane threshold
    * :math:`S_{\\rm out}` - Output spike
    * :math:`R` - Reset mechanism: if active, :math:`R = 1`, otherwise :math:`R = 0`
    * :math:`β` - Membrane potential decay rate
    * :math:`V` - Explicit recurrent weight

    Example::

        import torch
        import torch.nn as nn
        import snntorch as snn

        beta = 0.5

        # shared recurrent connection for a given layer
        V1 = 0.5

        # independent connection p/neuron
        V2 = torch.rand(num_outputs)

        # Define Network
        class Net(nn.Module):
            def __init__(self):
                super().__init__()

                # initialize layers
                self.fc1 = nn.Linear(num_inputs, num_hidden)
                self.lif1 = snn.RLeaky(beta=beta, linear_features=num_hidden)
                self.fc2 = nn.Linear(num_hidden, num_outputs)
                self.lif2 = snn.RLeaky(beta=beta, all_to_all=False, V=V1)

            def forward(self, x, mem1, spk1, mem2):
                cur1 = self.fc1(x)
                spk1, mem1 = self.lif1(cur1, spk1, mem1)
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, spk2, mem2)
                return mem1, spk1, mem2, spk2

    :param beta: membrane potential decay rate. Clipped between 0 and 1 during the forward-pass. May be a single-valued tensor (i.e., equal decay rate for all neurons in a layer), or multi-valued (one weight per neuron).
    :type beta: float or torch.tensor

    :param V: Recurrent weights to scale output spikes, only used when `all_to_all=False`. Defaults to 1.
    :type V: float or torch.tensor

    :param all_to_all: Enables output spikes to be connected in dense or convolutional recurrent structures instead of 1-to-1 connections. Defaults to True.
    :type all_to_all: bool, optional

    :param linear_features: Size of each output sample. Must be specified if `all_to_all=True` and the input data is 1D. Defaults to None
    :type linear_features: int, optional

    :param conv2d_channels: Number of channels in each output sample. Must be specified if `all_to_all=True` and the input data is 3D. Defaults to None
    :type conv2d_channels: int, optional

    :param kernel_size:  Size of the convolving kernel. Must be specified if `all_to_all=True` and the input data is 3D. Defaults to None
    :type kernel_size: int or tuple

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

    :param learn_recurrent: Option to enable learnable recurrent weights. Defaults to True
    :type learn_recurrent: bool, optional

    :param learn_threshold: Option to enable learnable threshold. Defaults to False
    :type learn_threshold: bool, optional

    :param reset_mechanism: Defines the reset mechanism applied to :math:`mem` each time the threshold is met. Reset-by-subtraction: "subtract", reset-to-zero: "zero, none: "none". Defaults to "subtract"
    :type reset_mechanism: str, optional

    :param state_quant: If specified, hidden state :math:`mem` is quantized to a valid state for the forward pass. Defaults to False
    :type state_quant: quantization function from snntorch.quant, optional

    :param output: If `True` as well as `init_hidden=True`, states are returned when neuron is called. Defaults to False
    :type output: bool, optional




    Inputs: \\input_, spk_0, mem_0
        - **input_** of shape `(batch, input_size)`: tensor containing input features
        - **spk_0** of shape `(batch, input_size)`: tensor containing output spike features
        - **mem_0** of shape `(batch, input_size)`: tensor containing the initial membrane potential for each element in the batch.

    Outputs: spk_1, mem_1
        - **spk_1** of shape `(batch, input_size)`: tensor containing the output spikes.
        - **mem_1** of shape `(batch, input_size)`: tensor containing the next membrane potential for each element in the batch

    Learnable Parameters:
        - **RLeaky.beta** (torch.Tensor) - optional learnable weights must be manually passed in, of shape `1` or (input_size).
        - **RLeaky.recurrent.weight** (torch.Tensor) - optional learnable weights are automatically generated if `all_to_all=True`. `RLeaky.recurrent` stores a `nn.Linear` or `nn.Conv2d` layer depending on input arguments provided.
        - **RLeaky.V** (torch.Tensor) - optional learnable weights must be manually passed in, of shape `1` or (input_size). It is only used where `all_to_all=False` for 1-to-1 recurrent connections.
        - **RLeaky.threshold** (torch.Tensor) - optional learnable thresholds must be manually passed in, of shape `1` or`` (input_size).

    """

    def __init__(
        self,
        beta,
        V=1.0,
        all_to_all=True,
        linear_features=None,
        conv2d_channels=None,
        kernel_size=None,
        threshold=1.0,
        spike_grad=None,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        learn_recurrent=True,  # changed learn_V
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
    ):
        super(RLeaky, self).__init__(
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

        self.all_to_all = all_to_all
        self.learn_recurrent = learn_recurrent

        # linear params
        self.linear_features = linear_features

        # Conv2d params
        self.kernel_size = kernel_size
        self.conv2d_channels = conv2d_channels

        # catch cases
        self._rleaky_init_cases()

        # initialize recurrent connections
        if self.all_to_all:  # init all-all connections
            self._init_recurrent_net()
        else:  # initialize 1-1 connections
            self._V_register_buffer(V, learn_recurrent)
            self._init_recurrent_one_to_one()

        if not learn_recurrent:
            self._disable_recurrent_grad()

        if self.init_hidden:
            self.spk, self.mem = self.init_rleaky()
            self.state_fn = self._build_state_function_hidden
        else:
            self.state_fn = self._build_state_function

    def forward(self, input_, spk=False, mem=False):
        if hasattr(spk, "init_flag") or hasattr(
            mem, "init_flag"
        ):  # only triggered on first-pass
            spk, mem = _SpikeTorchConv(spk, mem, input_=input_)
        elif mem is False and hasattr(self.mem, "init_flag"):  # init_hidden case
            self.spk, self.mem = _SpikeTorchConv(self.spk, self.mem, input_=input_)

        # TO-DO: alternatively, we could do torch.exp(-1 / self.beta.clamp_min(0)),
        # giving actual time constants instead of values in [0, 1] as initial beta
        # beta = self.beta.clamp(0, 1)

        if not self.init_hidden:
            self.reset = self.mem_reset(mem)
            mem = self.state_fn(input_, spk, mem)

            if self.state_quant:
                mem = self.state_quant(mem)

            if self.inhibition:
                spk = self.fire_inhibition(mem.size(0), mem)  # batch_size
            else:
                spk = self.fire(mem)

            return spk, mem

        # intended for truncated-BPTT where instance variables are hidden states
        if self.init_hidden:
            self._rleaky_forward_cases(spk, mem)
            self.reset = self.mem_reset(self.mem)
            self.mem = self.state_fn(input_)

            if self.state_quant:
                self.mem = self.state_quant(self.mem)

            if self.inhibition:
                self.spk = self.fire_inhibition(self.mem.size(0), self.mem)
            else:
                self.spk = self.fire(self.mem)

            if self.output:  # read-out layer returns output+states
                return self.spk, self.mem
            else:  # hidden layer e.g., in nn.Sequential, only returns output
                return self.spk

    def _init_recurrent_net(self):
        if self.all_to_all:
            if self.linear_features:
                self._init_recurrent_linear()
            elif self.kernel_size is not None:
                self._init_recurrent_conv2d()
        else:
            self._init_recurrent_one_to_one()

    def _init_recurrent_linear(self):
        self.recurrent = nn.Linear(self.linear_features, self.linear_features)

    def _init_recurrent_conv2d(self):
        self._init_padding()
        self.recurrent = nn.Conv2d(
            in_channels=self.conv2d_channels,
            out_channels=self.conv2d_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )

    def _init_padding(self):
        if type(self.kernel_size) is int:
            self.padding = self.kernel_size // 2, self.kernel_size // 2
        else:
            self.padding = self.kernel_size[0] // 2, self.kernel_size[1] // 2

    def _init_recurrent_one_to_one(self):
        self.recurrent = RecurrentOneToOne(self.V)

    def _disable_recurrent_grad(self):
        for param in self.recurrent.parameters():
            param.requires_grad = False

    def _base_state_function(self, input_, spk, mem):
        base_fn = self.beta.clamp(0, 1) * mem + input_ + self.recurrent(spk)
        return base_fn

    def _build_state_function(self, input_, spk, mem):
        if self.reset_mechanism_val == 0:  # reset by subtraction
            state_fn = self._base_state_function(
                input_, spk, mem - self.reset * self.threshold
            )
        elif self.reset_mechanism_val == 1:  # reset to zero
            state_fn = self._base_state_function(
                input_, mem
            ) - self.reset * self._base_state_function(input_, spk, mem)
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            state_fn = self._base_state_function(input_, spk, mem)
        return state_fn

    def _base_state_function_hidden(self, input_):
        base_fn = self.beta.clamp(0, 1) * self.mem + input_ + self.recurrent(self.spk)
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

    def _rleaky_forward_cases(self, spk, mem):
        if mem is not False or spk is not False:
            raise TypeError("When `init_hidden=True`, RLeaky expects 1 input argument.")

    def _rleaky_init_cases(self):
        all_to_all_bool = bool(self.all_to_all)
        linear_features_bool = self.linear_features
        conv2d_channels_bool = bool(self.conv2d_channels)
        kernel_size_bool = bool(self.kernel_size)

        if all_to_all_bool:
            if not (linear_features_bool):
                if not (conv2d_channels_bool or kernel_size_bool):
                    raise TypeError(
                        "When `all_to_all=True`, RLeaky requires either `linear_features` or (`conv2d_channels` and `kernel_size`) to be specified. The shape should match the shape of the output spike of the layer."
                    )
                elif conv2d_channels_bool ^ kernel_size_bool:
                    raise TypeError(
                        "`conv2d_channels` and `kernel_size` must both be specified. The shape of `conv2d_channels` should match the shape of the output spikes."
                    )
            elif (linear_features_bool and kernel_size_bool) or (
                linear_features_bool and conv2d_channels_bool
            ):
                raise TypeError(
                    "`linear_features` cannot be specified at the same time as `conv2d_channels` or `kernel_size`. A linear layer and conv2d layer cannot both be specified at the same time."
                )
        else:
            if linear_features_bool or conv2d_channels_bool or kernel_size_bool:
                raise TypeError(
                    "When `all_to_all`=False, none of `linear_features`, `conv2d_channels`, or `kernel_size` should be specified. The weight `V` is used instead."
                )

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], RLeaky):
                cls.instances[layer].mem.detach_()
                cls.instances[layer].spk.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables.
        Assumes hidden states have a batch dimension already."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], RLeaky):
                cls.instances[layer].spk, cls.instances[layer].mem = cls.instances[
                    layer
                ].init_rleaky()


class RecurrentOneToOne(nn.Module):
    def __init__(self, V):
        super(RecurrentOneToOne, self).__init__()
        self.V = V

    def forward(self, x):
        return x * self.V  # element-wise or global multiplication

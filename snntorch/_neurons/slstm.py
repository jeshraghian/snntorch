import torch
from torch._C import Value
import torch.nn as nn
import torch.nn.functional as F
from .neurons import *


class SLSTM(SpikingNeuron):

    """
    A spiking long short-term memory cell. Hidden states are :math:`h, c`.

    The input is expected to be of size :math:`(N, X)` where :math:`N` is the batch size.

    Unlike the LSTM module in PyTorch, only one time step is simulated each time the cell is called.

    .. math::
            \\begin{array}{ll} \\\\
            i_t = \\sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\\\
            f_t = \\sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\\\
            g_t = \\tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\\\
            o_t = \\sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\\\
            c_t = f_t ∗  c_{t-1} + i_t ∗  g_t \\\\
            h_t = o_t ∗  \\tanh(c_t) \\\\
        \\end{array}

    where :math:`\\sigma` is the sigmoid function and ∗ is the Hadamard product.
    The output state :math:`h_{t+1}` is thresholded to determine whether an output spike is generated.
    To conform to standard LSTM state behavior, the default reset mechanism is set to `reset="none"`, i.e., no reset is applied. If this is changed, the reset is only applied to :math:`h_t`.

    Example::

        import torch
        import torch.nn as nn
        import snntorch as snn

        beta = 0.5

        # Define Network
        class Net(nn.Module):
            def __init__(self):
                super().__init__()

                num_inputs = 784
                num_hidden1 = 1000
                num_hidden2 = 10

                spike_grad_lstm = surrogate.straight_through_estimator()
                spike_grad_fc = surrogate.fast_sigmoid(slope=5)

                # initialize layers
                self.slstm1 = snn.SLSTM(num_inputs, num_hidden1, spike_grad=spike_grad_lstm)
                self.slstm2 = snn.SLSTM(num_hidden1, num_hidden2, spike_grad=spike_grad_lstm)

            def forward(self, x):
                # Initialize hidden states and outputs at t=0
                c1, h1 = self.lif1.init_slstm()
                c2, h2 = self.lif2.init_slstm()

                # Record the final layer
                spk2_rec = []
                mem2_rec = []

                for step in range(num_steps):
                    spk1, c1, h1 = self.lif1(x.flatten(1), c1, h1)
                    spk2, c2, h2 = self.lif2(spk1, c2, h2)

                    spk2_rec.append(spk2)
                    mem2_rec.append(mem2)

                return torch.stack(spk2_rec), torch.stack(mem2_rec)



    :param input_size: number of expected features in the input :math:`x`
    :type input_size: int

    :param hidden_size: the number of features in the hidden state :math:`h`
    :type hidden_size: int

    :param bias: If `True`, adds a learnable bias to the output. Defaults to `True`
    :type bias: bool, optional

    :param threshold: Threshold for :math:`h` to reach in order to generate a spike `S=1`. Defaults to 1
    :type threshold: float, optional

    :param spike_grad: Surrogate gradient for the term dS/dh. Defaults to a straight-through-estimator
    :type spike_grad: surrogate gradient function from snntorch.surrogate, optional

    :param learn_threshold: Option to enable learnable threshold. Defaults to False
    :type learn_threshold: bool, optional

    :param init_hidden: Instantiates state variables as instance variables. Defaults to False
    :type init_hidden: bool, optional

    :param inhibition: If `True`, suppresses all spiking other than the neuron with the highest state. Defaults to False
    :type inhibition: bool, optional

    :param reset_mechanism: Defines the reset mechanism applied to :math:`h` each time the threshold is met. Reset-by-subtraction: "subtract", reset-to-zero: "zero, none: "none". Defaults to "none"
    :type reset_mechanism: str, optional

    :param output: If `True` as well as `init_hidden=True`, states are returned when neuron is called. Defaults to False
    :type output: bool, optional
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        threshold=1.0,
        spike_grad=None,
        learn_threshold=False,
        init_hidden=False,
        inhibition=False,
        reset_mechanism="none",
        output=False,
    ):

        super().__init__(
            threshold,
            spike_grad,
            init_hidden,
            inhibition,
            learn_threshold,
            reset_mechanism,
            output,
        )

        if self.init_hidden:
            self.c, self.h = self.init_slstm()
            self.state_fn = self._build_state_function_hidden
        else:
            self.state_fn = self._build_state_function

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.lstm_cell = nn.LSTMCell(self.input_size, self.hidden_size, bias=self.bias)

    def forward(self, input_, c=False, h=False):
        if hasattr(h, "init_flag") or hasattr(
            c, "init_flag"
        ):  # only triggered on first-pass

            c, h = _SpikeTorchConv(c, h, input_=self._reshape_input(input_))
        elif h is False and hasattr(self.h, "init_flag"):  # init_hidden case
            self.c, self.h = _SpikeTorchConv(
                self.c, self.h, input_=self._reshape_input(input_)
            )

        if not self.init_hidden:
            self.reset = self.mem_reset(h)
            c, h = self.state_fn(input_, c, h)
            spk = self.fire(h)
            return spk, c, h

        if self.init_hidden:
            # self._slstm_forward_cases(h, c)
            self.reset = self.mem_reset(self.h)
            self.c, self.h = self.state_fn(input_)
            self.spk = self.fire(self.h)

            if self.output:
                return self.spk, self.c, self.h
            else:
                return self.spk

    def _base_state_function(self, input_, c, h):
        base_fn_h, base_fn_c = self.lstm_cell(input_, (h, c))
        return base_fn_c, base_fn_h

    def _base_state_reset_zero(self, input_, c, h):
        base_fn_h, _ = self.lstm_cell(input_, (h, c))
        return 0, base_fn_h

    def _build_state_function(self, input_, c, h):
        if self.reset_mechanism_val == 0:  # reset by subtraction
            state_fn = tuple(
                map(
                    lambda x, y: x - y,
                    self._base_state_function(input_, c, h),
                    (0, self.reset * self.threshold),
                )
            )
        elif self.reset_mechanism_val == 1:  # reset to zero
            state_fn = tuple(
                map(
                    lambda x, y: x - self.reset * y,
                    self._base_state_function(input_, c, h),
                    self._base_state_reset_zero(input_, c, h),
                )
            )
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            state_fn = self._base_state_function(input_, c, h)
        return state_fn

    def _base_state_function_hidden(self, input_):
        base_fn_h, base_fn_c = self.lstm_cell(input_, (self.h, self.c))
        return base_fn_c, base_fn_h

    def _base_state_reset_zero_hidden(self, input_):
        base_fn_h, _ = self.lstm_cell(input_, (self.h, self.c))
        return 0, base_fn_h

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

    def _reshape_input(self, input_):
        if input_.is_cuda:
            device = "cuda"
        else:
            device = "cpu"
        b, _ = input_.size()
        return torch.zeros(b, self.hidden_size).to(device)

    @staticmethod
    def init_slstm():
        """
        Used to initialize h and c as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert the hidden states to the same as the input.
        """
        h = _SpikeTensor(init_flag=False)
        c = _SpikeTensor(init_flag=False)

        return h, c

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], SLSTM):
                cls.instances[layer].c.detach_()
                cls.instances[layer].h.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], SLSTM):
                cls.instances[layer].c = _SpikeTensor(init_flag=False)
                cls.instances[layer].h = _SpikeTensor(init_flag=False)

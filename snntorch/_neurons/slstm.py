import torch
from torch._C import Value
import torch.nn as nn
import torch.nn.functional as F
from .neurons import *


class SLSTM(SpikingNeuron):

    """
    A spiking long short-term memory cell.
    Hidden states are membrane potential and synaptic current :math:`mem, syn`, which correspond to the hidden and cell states :math:`h, c` in the original LSTM formulation.

    The input is expected to be of size :math:`(N, X)` where :math:`N` is the batch size.

    Unlike the LSTM module in PyTorch, only one time step is simulated each time the cell is called.

    .. math::
            \\begin{array}{ll} \\\\
            i_t = \\sigma(W_{ii} x_t + b_{ii} + W_{hi} mem_{t-1} + b_{hi}) \\\\
            f_t = \\sigma(W_{if} x_t + b_{if} + W_{hf} mem_{t-1} + b_{hf}) \\\\
            g_t = \\tanh(W_{ig} x_t + b_{ig} + W_{hg} mem_{t-1} + b_{hg}) \\\\
            o_t = \\sigma(W_{io} x_t + b_{io} + W_{ho} mem_{t-1} + b_{ho}) \\\\
            syn_t = f_t ∗  syn_{t-1} + i_t ∗  g_t \\\\
            mem_t = o_t ∗  \\tanh(syn_t) \\\\
        \\end{array}

    where :math:`\\sigma` is the sigmoid function and ∗ is the Hadamard product.
    The output state :math:`mem_{t+1}` is thresholded to determine whether an output spike is generated.
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

                # initialize layers
                self.slstm1 = snn.SLSTM(num_inputs, num_hidden1, spike_grad=spike_grad_lstm)
                self.slstm2 = snn.SLSTM(num_hidden1, num_hidden2, spike_grad=spike_grad_lstm)

            def forward(self, x):
                # Initialize hidden states and outputs at t=0
                syn1, mem1 = self.slstm1.init_slstm()
                syn2, mem2 = self.slstm2.init_slstm()

                # Record the final layer
                spk2_rec = []
                mem2_rec = []

                for step in range(num_steps):
                    spk1, syn1, mem1 = self.slstm1(x.flatten(1), syn1, mem1)
                    spk2, syn2, mem2 = self.slstm2(spk1, syn2, mem2)

                    spk2_rec.append(spk2)
                    mem2_rec.append(mem2)

                return torch.stack(spk2_rec), torch.stack(mem2_rec)

    :param input_size: number of expected features in the input :math:`x`
    :type input_size: int

    :param hidden_size: the number of features in the hidden state :math:`mem`
    :type hidden_size: int

    :param bias: If `True`, adds a learnable bias to the output. Defaults to `True`
    :type bias: bool, optional

    :param threshold: Threshold for :math:`h` to reach in order to generate a spike `S=1`. Defaults to 1
    :type threshold: float, optional

    :param spike_grad: Surrogate gradient for the term dS/dU. Defaults to a straight-through-estimator
    :type spike_grad: surrogate gradient function from snntorch.surrogate, optional

    :param learn_threshold: Option to enable learnable threshold. Defaults to False
    :type learn_threshold: bool, optional

    :param init_hidden: Instantiates state variables as instance variables. Defaults to False
    :type init_hidden: bool, optional

    :param inhibition: If `True`, suppresses all spiking other than the neuron with the highest state. Defaults to False
    :type inhibition: bool, optional

    :param reset_mechanism: Defines the reset mechanism applied to :math:`mem` each time the threshold is met. Reset-by-subtraction: "subtract", reset-to-zero: "zero, none: "none". Defaults to "none"
    :type reset_mechanism: str, optional

    :param state_quant: If specified, hidden states :math:`mem` and :math:`syn` are quantized to a valid state for the forward pass. Defaults to False
    :type state_quant: quantization function from snntorch.quant, optional

    :param output: If `True` as well as `init_hidden=True`, states are returned when neuron is called. Defaults to False
    :type output: bool, optional


    Inputs: \\input_, syn_0, mem_0
        - **input_** of shape `(batch, input_size)`: tensor containing input features
        - **syn_0** of shape `(batch, hidden_size)`: tensor containing the initial synaptic current (or cell state) for each element in the batch.
        - **mem_0** of shape `(batch, hidden_size)`: tensor containing the initial membrane potential (or hidden state) for each element in the batch.

    Outputs: spk, syn_1, mem_1
        - **spk** of shape `(batch, hidden_size)`: tensor containing the output spike
        - **syn_1** of shape `(batch, hidden_size)`: tensor containing the next synaptic current (or cell state) for each element in the batch
        - **mem_1** of shape `(batch, hidden_size)`: tensor containing the next membrane potential (or hidden state) for each element in the batch

    Learnable Parameters:
        - **SLSTM.lstm_cell.weight_ih** (torch.Tensor) - the learnable input-hidden weights, of shape (4*hidden_size, input_size)
        - **SLSTM.lstm_cell.weight_ih** (torch.Tensor) – the learnable hidden-hidden weights, of shape (4*hidden_size, hidden_size)
        - **SLSTM.lstm_cell.bias_ih** – the learnable input-hidden bias, of shape (4*hidden_size)
        - **SLSTM.lstm_cell.bias_hh** – the learnable hidden-hidden bias, of shape (4*hidden_size)

    """

    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        threshold=1.0,
        spike_grad=None,
        init_hidden=False,
        inhibition=False,
        learn_threshold=False,
        reset_mechanism="none",
        state_quant=False,
        output=False,
    ):

        super().__init__(
            threshold,
            spike_grad,
            init_hidden,
            inhibition,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
        )

        if self.init_hidden:
            self.syn, self.mem = self.init_slstm()
            self.state_fn = self._build_state_function_hidden
        else:
            self.state_fn = self._build_state_function

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.lstm_cell = nn.LSTMCell(self.input_size, self.hidden_size, bias=self.bias)

    def forward(self, input_, syn=False, mem=False):
        if hasattr(mem, "init_flag") or hasattr(
            syn, "init_flag"
        ):  # only triggered on first-pass

            syn, mem = _SpikeTorchConv(syn, mem, input_=self._reshape_input(input_))
        elif mem is False and hasattr(self.mem, "init_flag"):  # init_hidden case
            self.syn, self.mem = _SpikeTorchConv(
                self.syn, self.mem, input_=self._reshape_input(input_)
            )

        if not self.init_hidden:
            self.reset = self.mem_reset(mem)
            syn, mem = self.state_fn(input_, syn, mem)

            if self.state_quant:
                syn = self.state_quant(syn)
                mem = self.state_quant(mem)

            spk = self.fire(mem)
            return spk, syn, mem

        if self.init_hidden:
            # self._slstm_forward_cases(mem, syn)
            self.reset = self.mem_reset(self.mem)
            self.syn, self.mem = self.state_fn(input_)

            if self.state_quant:
                self.syn = self.state_quant(self.syn)
                self.mem = self.state_quant(self.mem)

            self.spk = self.fire(self.mem)

            if self.output:
                return self.spk, self.syn, self.mem
            else:
                return self.spk

    def _base_state_function(self, input_, syn, mem):
        base_fn_mem, base_fn_syn = self.lstm_cell(input_, (mem, syn))
        return base_fn_syn, base_fn_mem

    def _base_state_reset_zero(self, input_, syn, mem):
        base_fn_mem, _ = self.lstm_cell(input_, (mem, syn))
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
        base_fn_mem, base_fn_syn = self.lstm_cell(input_, (self.mem, self.syn))
        return base_fn_syn, base_fn_mem

    def _base_state_reset_zero_hidden(self, input_):
        base_fn_mem, _ = self.lstm_cell(input_, (self.mem, self.syn))
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
        Used to initialize mem and syn as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert the hidden states to the same as the input.
        """
        mem = _SpikeTensor(init_flag=False)
        syn = _SpikeTensor(init_flag=False)

        return mem, syn

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], SLSTM):
                cls.instances[layer].syn.detach_()
                cls.instances[layer].mem.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], SLSTM):
                cls.instances[layer].syn = _SpikeTensor(init_flag=False)
                cls.instances[layer].mem = _SpikeTensor(init_flag=False)

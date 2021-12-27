import torch
from torch._C import Value
import torch.nn as nn
import torch.nn.functional as F
from .neurons import *


class SConvLSTM(SpikingNeuron):

    """
    A spiking 2d convolutional long short-term memory cell. Hidden states are :math:`h, c`.

    The input is expected to be of size :math:`(N, C_{in}, H_{in}, W_{in})` where :math:`N` is the batch size.

    Unlike the LSTM module in PyTorch, only one time step is simulated each time the cell is called.

    .. math::
            \\begin{array}{ll} \\\\
            i_t = \\sigma(W_{ii} ⋆ x_t + b_{ii} + W_{hi} ⋆ h_{t-1} + b_{hi}) \\\\
            f_t = \\sigma(W_{if} ⋆ x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\\\
            g_t = \\tanh(W_{ig} ⋆ x_t + b_{ig} + W_{hg} ⋆ h_{t-1} + b_{hg}) \\\\
            o_t = \\sigma(W_{io} ⋆ x_t + b_{io} + W_{ho} ⋆ h_{t-1} + b_{ho}) \\\\
            c_t = f_t ∗  c_{t-1} + i_t ∗  g_t \\\\
            h_t = o_t ∗  \\tanh(c_t) \\\\
        \\end{array}

    where ⋆ is the 2D cross-correlation operator and ∗ is the Hadamard product.
    The output state :math:`h_{t+1}` is thresholded to determine whether an output spike is generated.
    To conform to standard LSTM state behavior, the default reset mechanism is set to `reset="none"`, i.e., no reset is applied. If this is changed, the reset is only applied to :math:`h_t`.

    Options to apply max-pooling or average-pooling to the state :math:`h_t` are also enabled. Note that it is preferable to apply pooling to the state rather than the spike, as it does not make sense to apply pooling to activations of 1's and 0's which may lead to random tie-breaking.

    Example::

        import torch
        import torch.nn as nn
        import snntorch as snn

        beta = 0.5

        # Define Network
        class Net(nn.Module):
            def __init__(self):
                super().__init__()

                in_channels = 1
                hidden_channels = 8
                out_channels = 16
                kernel_size = 3
                max_pool = 2
                avg_pool = 2
                flattened_input = 49 * 16
                num_outputs = 10
                beta = 0.5

                spike_grad_lstm = surrogate.straight_through_estimator()
                spike_grad_fc = surrogate.fast_sigmoid(slope=5)

                # initialize layers
                self.sclstm1 = snn.SConvLSTM(in_channels, hidden_channels, kernel_size, max_pool=max_pool, spike_grad=spike_grad_lstm)
                self.sclstm2 = snn.SConvLSTM(hidden_channels, out_channels, kernel_size, avg_pool=avg_pool, spike_grad=spike_grad_lstm)
                self.fc2 = nn.Linear(flattened_input, num_outputs)
                self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad_fc)

            def forward(self, x, mem1, spk1, mem2):
                # Initialize hidden states and outputs at t=0
                c1, h1 = self.lif1.init_sconvlstm()
                c2, h2 = self.lif2.init_sconvlstm()
                mem3 = self.lif3.init_leaky()

                # Record the final layer
                spk3_rec = []
                mem3_rec = []


                for step in range(num_steps):
                    spk1, c1, h1 = self.lif1(x, c1, h1)
                    spk2, c2, h2 = self.lif2(spk1, c2, h2)
                    cur = self.fc1(spk2.flatten(1))
                    spk3, mem3 = self.lif3(cur, mem3)

                    spk3_rec.append(spk3)
                    mem3_rec.append(mem3)

                return torch.stack(spk3_rec), torch.stack(mem3_rec)



    :param input_dim: number of input channels
    :type input_dim: int

    :param kernel_size: Size of the convolving kernel
    :type kernel_size: int, tuple, or list

    :param bias: If `True`, adds a learnable bias to the output. Defaults to `True`
    :type bias: bool, optional

    :param max_pool: Applies max-pooling to the hidden state :math:`h` prior to thresholding if specified. Defaults to 0
    :type max_pool: int, tuple, or list, optional

    :param avg_pool: Applies average-pooling to the hidden state :math:`h` prior to thresholding if specified. Defaults to 0
    :type avg_pool: int, tuple, or list, optional

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
        input_dim,
        hidden_dim,
        kernel_size,
        bias=True,
        max_pool=0,
        avg_pool=0,
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
            self.c, self.h = self.init_sconvlstm()
            self.state_fn = self._build_state_function_hidden
        else:
            self.state_fn = self._build_state_function

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.max_pool = max_pool
        self.avg_pool = avg_pool
        self.bias = bias
        self._sconvlstm_cases()

        # padding is essential to keep same shape for next step
        if type(self.kernel_size) is int:
            self.padding = kernel_size // 2, kernel_size // 2
        else:
            self.padding = kernel_size[0] // 2, kernel_size[1] // 2

        # Note, this applies the same Conv to all 4 gates
        # Regular LSTMs have different dense layers applied to all 4 gates
        # Consider: a separate nn.Conv2d instance p/gate?
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

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
            if self.max_pool:
                spk = self.fire(F.max_pool2d(h, self.max_pool))
            elif self.avg_pool:
                spk = self.fire(F.avg_pool2d(h, self.max_pool))
            else:
                spk = self.fire(h)
            return spk, c, h

        if self.init_hidden:
            # self._sconvlstm_forward_cases(h, c)
            self.reset = self.mem_reset(self.h)
            self.c, self.h = self.state_fn(input_)

            if self.max_pool:
                self.spk = self.fire(F.max_pool2d(self.h, self.max_pool))
            elif self.avg_pool:
                self.spk = self.fire(F.avg_pool2d(self.h, self.avg_pool))
            else:
                self.spk = self.fire(self.h)

            if self.output:
                return self.spk, self.c, self.h
            else:
                return self.spk

    def _base_state_function(self, input_, c, h):

        combined = torch.cat(
            [input_, h], dim=1
        )  # concatenate along channel axis (BxCxHxW)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        base_fn_c = f * c + i * g
        base_fn_h = o * torch.tanh(base_fn_c)

        return base_fn_c, base_fn_h

    def _base_state_reset_zero(self, input_, c, h):
        combined = torch.cat([input_, h], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        base_fn_c = f * c + i * g
        base_fn_h = o * torch.tanh(base_fn_c)

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
        combined = torch.cat([input_, self.h], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        base_fn_c = f * self.c + i * g
        base_fn_h = o * torch.tanh(base_fn_c)

        return base_fn_c, base_fn_h

    def _base_state_reset_zero_hidden(self, input_):
        combined = torch.cat([input_, self.h], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        base_fn_c = f * self.c + i * g
        base_fn_h = o * torch.tanh(base_fn_c)

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

    @staticmethod
    def init_sconvlstm():
        """
        Used to initialize h and c as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert the hidden states to the same as the input.
        """
        h = _SpikeTensor(init_flag=False)
        c = _SpikeTensor(init_flag=False)

        return h, c

    def _reshape_input(self, input_):
        if input_.is_cuda:
            device = "cuda"
        else:
            device = "cpu"
        b, _, h, w = input_.size()
        return torch.zeros(b, self.hidden_dim, h, w).to(device)

    def _sconvlstm_cases(self):
        if self.max_pool and self.avg_pool:
            raise ValueError(
                "Only one of either `max_pool` or `avg_pool` may be specified, not both."
            )

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], SConvLSTM):
                cls.instances[layer].c.detach_()
                cls.instances[layer].h.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], SConvLSTM):
                cls.instances[layer].c = _SpikeTensor(init_flag=False)
                cls.instances[layer].h = _SpikeTensor(init_flag=False)


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers,
        batch_first=False,
        bias=True,
        return_all_layers=False,
    ):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (
                isinstance(kernel_size, list)
                and all([isinstance(elem, tuple) for elem in kernel_size])
            )
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    # @staticmethod
    # def _extend_for_multilayer(param, num_layers):
    #     if not isinstance(param, list):
    #         param = [param] * num_layers
    #     return param

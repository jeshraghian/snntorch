from .neurons import _SpikeTensor, _SpikeTorchConv, LIF
import torch
import torch.nn as nn

class LeakyParallel(nn.Module):
    """
    A parallel implementation of the Leaky neuron with an input linear layer.
    All time steps are passed to the input at once. 

    First-order leaky integrate-and-fire neuron model.
    Input is assumed to be a current injection.
    Membrane potential decays exponentially with rate beta.
    For :math:`U[T] > U_{\\rm thr} ⇒ S[T+1] = 1`.

    .. math::

            U[t+1] = βU[t] + I_{\\rm in}[t+1]


    * :math:`I_{\\rm in}` - Input current
    * :math:`U` - Membrane potential
    * :math:`U_{\\rm thr}` - Membrane threshold
    * :math:`β` - Membrane potential decay rate

    Example::

        import torch
        import torch.nn as nn
        import snntorch as snn

        beta = 0.5

        # Define Network
        class Net(nn.Module):
            def __init__(self):
                super().__init__()

                # initialize layers
                self.lif1 = snn.ParallelLeaky(input_size=784, hidden_size=128)
                self.lif2 = snn.ParallelLeaky(input_size=128, hidden_size=10, beta=beta)

            def forward(self, x):
                spk1 = self.lif1(x)
                spk2 = self.lif2(spk1)
                return spk2     

        
    :param input_size: The number of expected features in the input `x`
    :type input_size: int

    :param hidden_size: The number of features in the hidden state `h`
    :type hidden_size: int

    :param beta: membrane potential decay rate. Clipped between 0 and 1
        during the forward-pass. May be a single-valued tensor (i.e., equal
        decay rate for all neurons in a layer), or multi-valued (one weight per
        neuron). If left unspecified, then the decay rates will be randomly initialized based on PyTorch's initialization for RNN. Defaults to None
    :type beta: float or torch.tensor, optional

    :param bias: If `False`, then the layer does not use bias weights `b_ih` and `b_hh`. Defaults to True
    :type bias: Bool, optional

    :param threshold: Threshold for :math:`mem` to reach in order to
        generate a spike `S=1`. Defaults to 1
    :type threshold: float, optional

    :param dropout: If non-zero, introduces a Dropout layer on the RNN output with dropout probability equal to dropout. Defaults to 0
    :type dropout: float, optional

    :param spike_grad: Surrogate gradient for the term dS/dU. Defaults to
        None (corresponds to ATan surrogate gradient. See
        `snntorch.surrogate` for more options)
    :type spike_grad: surrogate gradient function from snntorch.surrogate,
        optional

    :param surrogate_disable: Disables surrogate gradients regardless of
        `spike_grad` argument. Useful for ONNX compatibility. Defaults
        to False
    :type surrogate_disable: bool, Optional

    :param init_hidden: Instantiates state variables as instance variables.
        Defaults to False
    :type init_hidden: bool, optional

    :param inhibition: If `True`, suppresses all spiking other than the
        neuron with the highest state. Defaults to False
    :type inhibition: bool, optional

    :param learn_beta: Option to enable learnable beta. Defaults to False
    :type learn_beta: bool, optional

    :param learn_threshold: Option to enable learnable threshold. Defaults
        to False
    :type learn_threshold: bool, optional



    Inputs: \\input_
        - **input_** of shape of  shape `(L, H_{in})` for unbatched input, 
            or `(L, N, H_{in})` containing the features of the input sequence. 

    Outputs: spk
        - **spk** of shape `(L, batch, input_size)`: tensor containing the
            output spikes.
        
    where:

    `L = sequence length`
    
    `N = batch size`

    `H_{in} = input_size`

    `H_{out} = hidden_size`

    Learnable Parameters:
        - **rnn.weight_ih_l** (torch.Tensor) - the learnable input-hidden weights of shape (hidden_size, input_size)
        - **rnn.weight_hh_l** (torch.Tensor) - the learnable hidden-hidden weights of the k-th layer which are sampled from `beta` of shape (hidden_size, hidden_size)
        - **bias_ih_l** - the learnable input-hidden bias of the k-th layer, of shape (hidden_size)
        - **bias_hh_l** - the learnable hidden-hidden bias of the k-th layer, of shape (hidden_size)
        - **threshold** (torch.Tensor) - optional learnable thresholds
            must be manually passed in, of shape `1` or`` (input_size).
        - **graded_spikes_factor** (torch.Tensor) - optional learnable graded spike factor

    """

    def __init__(
        self,
        input_size,
        hidden_size,
        beta=None,
        bias=True,
        threshold=1.0,
        dropout=0.0,
        spike_grad=None,
        surrogate_disable=False,
        learn_beta=False,
        learn_threshold=False,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
        device=None,
        dtype=None,
    ):
        super(LeakyParallel, self).__init__()
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='relu', 
                          bias=bias, batch_first=False, dropout=dropout, device=device, dtype=dtype)
        
        self._beta_buffer(beta, learn_beta)
        if self.beta is not None:
            self.beta = self.beta.clamp(0, 1) 

        if spike_grad is None:
            self.spike_grad = self.ATan.apply
        else:
            self.spike_grad = spike_grad

        self._threshold_buffer(threshold, learn_threshold)
        self._graded_spikes_buffer(
            graded_spikes_factor, learn_graded_spikes_factor
        )

        self.surrogate_disable = surrogate_disable
        if self.surrogate_disable:
            self.spike_grad = self._surrogate_bypass

        with torch.no_grad():
            if self.beta is not None:
                # Set all weights to the scalar value of self.beta
                if isinstance(self.beta, float) or isinstance(self.beta, int):
                    self.rnn.weight_hh_l0.fill_(self.beta)
                elif isinstance(self.beta, torch.Tensor) or isinstance(self.beta, torch.FloatTensor):
                    if len(self.beta) == 1:
                        self.rnn.weight_hh_l0.fill_(self.beta[0])
                elif len(self.beta) == hidden_size:
                    # Replace each value with the corresponding value in self.beta
                    for i in range(hidden_size):
                        self.rnn.weight_hh_l0.data[i].fill_(self.beta[i])
                else:
                    raise ValueError("Beta must be either a single value or of length 'hidden_size'.")
        
        if not learn_beta:
            # Make the weights non-learnable
            self.rnn.weight_hh_l0.requires_grad_(False)


    def forward(self, input_):
        mem = self.rnn(input_)
        # mem[0] contains relu'd outputs, mem[1] contains final hidden state
        mem_shift = mem[0] - self.threshold
        spk = self.spike_grad(mem_shift)
        spk = spk * self.graded_spikes_factor
        return spk
    
    @staticmethod
    def _surrogate_bypass(input_):
        return (input_ > 0).float()

    @staticmethod
    class ATan(torch.autograd.Function):
        """
        Surrogate gradient of the Heaviside step function.

        **Forward pass:** Heaviside step function shifted.

            .. math::

                S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
                0 & \\text{if U < U$_{\\rm thr}$}
                \\end{cases}

        **Backward pass:** Gradient of shifted arc-tan function.

            .. math::

                    S&≈\\frac{1}{π}\\text{arctan}(πU \\frac{α}{2}) \\\\
                    \\frac{∂S}{∂U}&=\\frac{1}{π}\
                    \\frac{1}{(1+(πU\\frac{α}{2})^2)}


        :math:`alpha` defaults to 2, and can be modified by calling
        ``surrogate.atan(alpha=2)``.

        Adapted from:

        *W. Fang, Z. Yu, Y. Chen, T. Masquelier, T. Huang, Y. Tian (2021)
        Incorporating Learnable Membrane Time Constants to Enhance Learning
        of Spiking Neural Networks. Proc. IEEE/CVF Int. Conf. Computer
        Vision (ICCV), pp. 2661-2671.*"""

        @staticmethod
        def forward(ctx, input_, alpha=2.0):
            ctx.save_for_backward(input_)
            ctx.alpha = alpha
            out = (input_ > 0).float()
            return out

        @staticmethod
        def backward(ctx, grad_output):
            (input_,) = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad = (
                ctx.alpha
                / 2
                / (1 + (torch.pi / 2 * ctx.alpha * input_).pow_(2))
                * grad_input
            )
            return grad, None
        
    
    def _beta_buffer(self, beta, learn_beta):
        if not isinstance(beta, torch.Tensor):
            if beta is not None:
                beta = torch.as_tensor([beta])  # TODO: or .tensor() if no copy
            self.register_buffer("beta", beta)

    def _graded_spikes_buffer(
    self, graded_spikes_factor, learn_graded_spikes_factor
    ):
        if not isinstance(graded_spikes_factor, torch.Tensor):
            graded_spikes_factor = torch.as_tensor(graded_spikes_factor)
        if learn_graded_spikes_factor:
            self.graded_spikes_factor = nn.Parameter(graded_spikes_factor)
        else:
            self.register_buffer("graded_spikes_factor", graded_spikes_factor)

    def _threshold_buffer(self, threshold, learn_threshold):
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.as_tensor(threshold)
        if learn_threshold:
            self.threshold = nn.Parameter(threshold)
        else:
            self.register_buffer("threshold", threshold)
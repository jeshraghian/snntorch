import torch
from torch import nn
from torch.nn import functional as F
from profilehooks import profile

# from .neurons import LIF
from .stateleaky import StateLeaky

class Conv2dLeaky(StateLeaky):
    """
    TODO: write some docstring similar to SNN.Leaky

     Jason wrote:
-      beta = (1 - delta_t / tau), can probably set delta_t to "1"
-      if tau > delta_t, then beta: (0, 1)
    """

    def __init__(
        self,
        beta,
        in_channels, 
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        device=None, 
        dtype=None,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        learn_beta=False,
        learn_threshold=False,
        state_quant=False, 
        output=True,
        graded_spikes_factor=1.0, 
        learn_graded_spikes_factor=False,
    ):
        super().__init__(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            surrogate_disable=surrogate_disable,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            state_quant=state_quant,
            output=output,
            graded_spikes_factor=graded_spikes_factor,
            learn_graded_spikes_factor=learn_graded_spikes_factor,
        )

        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, padding_mode=padding_mode,
                                device=device, dtype=dtype, bias=bias)

    @property
    def beta(self): 
        return (self.tau-1) / self.tau

    # @profile(skip=True, stdout=True, filename='baseline.prof')
    def forward(self, input_):

        # not checked logic of function below
        input_ = self.conv2d(input_.reshape(input_.size(0) * input_.size(1), *input_.shape[2:]))  # TODO: input_ must be transformed T x B x C --> (T*B) x C
        self.mem = self._base_state_function(input_)

        if self.state_quant:
            self.mem = self.state_quant(self.mem)

        if self.output:
            self.spk = self.fire(self.mem) * self.graded_spikes_factor
            return self.spk, self.mem

        else:
            return self.mem


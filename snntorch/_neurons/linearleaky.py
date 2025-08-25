import torch
from torch import nn
from torch.nn import functional as F
from profilehooks import profile

# from .neurons import LIF
from .stateleaky import StateLeaky


class LinearLeaky(StateLeaky):
    """
        TODO: write some docstring similar to SNN.Leaky

         Jason wrote:
    -      beta = (1 - delta_t / tau), can probably set delta_t to "1"
    -      if tau > delta_t, then beta: (0, 1)
    """

    def __init__(
        self,
        beta,
        in_features,
        out_features,
        bias=True,
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
            channels=out_features,
        )

        self.linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            device=device,
            dtype=dtype,
            bias=bias,
        )

    @property
    def beta(self):
        return (self.tau - 1) / self.tau

    # @profile(skip=True, stdout=True, filename='baseline.prof')
    def forward(self, input_):
        num_steps, batch, channels = input_.shape

        input_ = self.linear(input_.reshape(-1, self.linear.in_features))

        input_ = input_.reshape(num_steps, batch, self.linear.out_features)
        self.mem = self._base_state_function(input_)

        if self.state_quant:
            self.mem = self.state_quant(self.memfoll)

        if self.output:
            self.spk = self.fire(self.mem) * self.graded_spikes_factor
            return self.spk, self.mem

        else:
            return self.mem


# TODO: throw exceptions if calling subclass methods we don't want to use
# fire_inhibition
# mem_reset, init, detach, zeros, reset_mem, init_leaky
# detach_hidden, reset_hidden

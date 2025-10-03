import torch
from torch import nn
from torch.nn import functional as F
from profilehooks import profile
from .neurons import LIF

import torch
from torch.autograd import Function
from torch.nn import functional as F


def causal_conv1d(input_tensor, kernel_tensor):
    batch_size, in_channels, num_steps = input_tensor.shape
    # kernel_tensor: (channels, 1, kernel_size)
    out_channels, _, kernel_size = kernel_tensor.shape

    # for causal convolution, output at time t only depends on inputs up to t
    # therefore, we pad only on the left side
    padding = kernel_size - 1
    padded_input = F.pad(input_tensor, (padding, 0))

    # kernel is flipped to turn cross-correlation performed by F.conv1d into convolution
    flipped_kernel = torch.flip(kernel_tensor, dims=[-1])

    # perform convolution with the padded input (output length = num_steps length)
    causal_conv_result = F.conv1d(
        padded_input, flipped_kernel, groups=in_channels
    )

    return causal_conv_result


class StateLeaky(LIF):
    """StateLeaky neuron model."""

    def __init__(
        self,
        beta,
        channels,
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

        self._tau_buffer(self.beta, learn_beta, channels)

    def _tau_buffer(self, beta, learn_beta, channels):
        if not isinstance(beta, torch.Tensor):
            beta = torch.as_tensor(beta)

        if (
            beta.shape != (channels,)
            and beta.shape != ()
            and beta.shape != (1,)
        ):
            raise ValueError(
                f"Beta shape {beta.shape} must be either ({channels},) or (1,)"
            )

        tau = 1 / (1 - beta + 1e-12)
        if learn_beta:
            self.tau = nn.Parameter(tau)
        else:
            self.register_buffer("tau", tau)

    def _base_state_function(self, input_):
        num_steps, batch, channels = input_.shape
        input_ = input_.permute(1, 2, 0)
        assert input_.shape == (batch, channels, num_steps)
        device = input_.device

        # time axis shape (1, 1, num_steps)
        time_steps = torch.arange(num_steps, device=device).view(
            1, 1, num_steps
        )
        assert time_steps.shape == (1, 1, num_steps)

        # single channel case
        if self.tau.shape == () or self.tau.shape == (1,):
            # tau is scalar, broadcast across channels
            tau = self.tau.to(device)
            decay_filter = torch.exp(-time_steps / tau).expand(
                channels, 1, num_steps
            )
        else:
            # tau is (channels,), reshape to (channels, 1, 1) so it broadcasts correctly
            tau = self.tau.to(device).view(channels, 1, 1)
            assert tau.shape == (channels, 1, 1)
            decay_filter = torch.exp(
                -time_steps / tau
            )  # directly (channels, 1, num_steps)

        assert decay_filter.shape == (channels, 1, num_steps)
        assert input_.shape == (batch, channels, num_steps)

        # depthwise convolution: each channel gets its own decay filter
        conv_result = causal_conv1d(input_, decay_filter)
        assert conv_result.shape == (batch, channels, num_steps)

        return conv_result.permute(2, 0, 1)  # (num_steps, batch, channels)

    @property
    def beta(self):
        return (self.tau - 1) / self.tau

    # @profile(skip=False, stdout=False, filename="baseline.prof")
    def forward(self, input_):
        mem = self._base_state_function(input_)

        if self.state_quant:
            mem = self.state_quant(mem)

        if self.output:
            self.spk = self.fire(mem) * self.graded_spikes_factor
            return self.spk, mem

        else:
            return mem


# TODO: throw exceptions if calling subclass methods we don't want to use
# fire_inhibition
# mem_reset, init, detach, zeros, reset_mem, init_leaky
# detach_hidden, reset_hidden


if __name__ == "__main__":
    device = "cuda"
    leaky_linear = StateLeaky(beta=0.9).to(device)
    timesteps = 5
    batch = 1
    channels = 1
    print("timesteps: ", timesteps)
    print("batch: ", batch)
    print("channels: ", channels)
    print()
    input_ = (
        torch.arange(1, timesteps * batch * channels + 1)
        .float()
        .view(timesteps, batch, channels)
        .to(device)
    )
    print("--------input tensor-----------")
    print(input_)
    print()
    out = leaky_linear.forward(input_)
    print("--------output-----------")
    print(out)

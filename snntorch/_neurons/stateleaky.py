import torch
from torch import nn
from torch.nn import functional as F
from profilehooks import profile

from .neurons import LIF
# from .linearleaky import LinearLeaky


class StateLeaky(LIF):
    """
    TODO: write some docstring similar to SNN.Leaky

     Jason wrote:
-      beta = (1 - delta_t / tau), can probably set delta_t to "1"
-      if tau > delta_t, then beta: (0, 1)
    """

    def __init__(
        self,
        beta,
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

        self._tau_buffer(self.beta, learn_beta)

    @property
    def beta(self): 
        return (self.tau-1) / self.tau

    # @profile(skip=True, stdout=True, filename='baseline.prof')
    def forward(self, input_):
        print(input_.shape)
        self.mem = self._base_state_function(input_)

        if self.state_quant:
            self.mem = self.state_quant(self.mem)

        if self.output:
            self.spk = self.fire(self.mem) * self.graded_spikes_factor
            return self.spk, self.mem

        else:
            return self.mem

    def _base_state_function(self, input_):
        # init time steps arr
        num_steps, batch, channels = input_.shape
        time_steps = torch.arange(0, num_steps, device=input_.device)
        assert time_steps.shape == (num_steps,)
        time_steps = time_steps.unsqueeze(1).expand(num_steps, channels)

        # init decay filter
        print(time_steps.shape)
        print(self.tau.shape)
        decay_filter = torch.exp(-time_steps / self.tau).to(input_.device)
        print("------------decay filter------------")
        print(decay_filter)
        print()
        assert decay_filter.shape == (num_steps, channels)

        # prepare for convolution
        input_ = input_.permute(1, 2, 0)
        assert input_.shape == (batch, channels, num_steps)
        decay_filter = decay_filter.permute(1, 0).unsqueeze(1)
        assert decay_filter.shape == (channels, 1, num_steps)

        conv_result = self.full_mode_conv1d_truncated(input_, decay_filter)
        assert conv_result.shape == (batch, channels, num_steps)

        return conv_result.permute(2, 0, 1)  # return membrane potential trace
    
    def _tau_buffer(self, beta, learn_beta):
        if not isinstance(beta, torch.Tensor):
            beta = torch.as_tensor(beta)
        
        tau = 1 / (1 - beta + 1e-12)

        if learn_beta:
            self.tau = nn.Parameter(tau)
        else:
            self.register_buffer("tau", tau)
    
    def full_mode_conv1d_truncated(self, input_tensor, kernel_tensor):
        # input_tensor: (batch, channels, num_steps)
        # kernel_tensor: (channels, 1, kernel_size)
        kernel_tensor = torch.flip(kernel_tensor, dims=[-1])

        # get dimensions
        batch_size, in_channels, num_steps = input_tensor.shape
        out_channels, _, kernel_size = kernel_tensor.shape

        # pad the input tensor on both sides
        padding = kernel_size - 1
        padded_input = F.pad(input_tensor, (padding, padding))

        # print(padded_input.shape)
        # print(input_tensor.shape)
        # print("------input / kernel-------------")
        # print(input_tensor)
        # print(kernel_tensor)

        # perform convolution with the padded input
        conv_result = F.conv1d(padded_input, kernel_tensor, groups=in_channels)

        # truncate the result to match the original input length
        truncated_result = conv_result[..., 0:num_steps]

        return truncated_result

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
    input_ = torch.arange(1, timesteps * batch * channels + 1).float().view(timesteps, batch, channels).to(device)
    print("--------input tensor-----------")
    print(input_)
    print()
    out = leaky_linear.forward(input_)
    print("--------output-----------")
    print(out)

from warnings import warn

import torch
from torch import nn
from torch.nn import functional as F
import torch
from torch.autograd import Function
from torch.nn import functional as F

from .neurons import LIF


def causal_conv1d(input_tensor, kernel_tensor):
    _batch_size, in_channels, _num_steps = input_tensor.shape
    # kernel_tensor: (channels, 1, kernel_size)
    _out_channels, _, kernel_size = kernel_tensor.shape

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
    r"""
    First-order state-only leaky neuron model that uses a causal exponential
    decay kernel to generate the membrane potential over time via depthwise
    causal convolution. Unlike :class:`Leaky`, no stepwise recurrent reset is
    applied inside the state update; spikes (if requested) are emitted by
    thresholding the resulting state.

    The effective per-channel decay filter is

    .. math::

        h[t] = e^{-t/\tau}, \quad t = 0, 1, \ldots

    Optionally, the decay kernel can be truncated to a finite memory window
    via ``kernel_truncation_steps``. When set to an integer ``K > 0``, only the
    most recent ``K`` taps contribute to each output time step (older
    contributions are dropped). Output sequence length is unchanged.

    Example::

        import torch
        from snntorch._neurons.stateleaky import StateLeaky

        T, B, C = 16, 2, 4
        x = torch.randn(T, B, C)
        lif = StateLeaky(beta=0.9, channels=C, output=True, kernel_truncation_steps=8)
        spk, mem = lif(x)

    :param beta: membrane potential decay rate. May be a single-valued tensor
        (shared across channels) or multi-valued of shape ``(channels,)``.
        Internally represented via :math:`\tau = 1/(1-\beta)`.
    :type beta: float or torch.tensor

    :param channels: Number of channels processed depthwise. Must match the
        input's channel dimension.
    :type channels: int

    :param threshold: Threshold used to generate spikes when ``output=True``.
        Defaults to 1.0
    :type threshold: float, optional

    :param spike_grad: Surrogate gradient for the term dS/dU when spikes are
        produced. Defaults to None (corresponds to ATan surrogate gradient. See
        ``snntorch.surrogate`` for more options)
    :type spike_grad: surrogate gradient function from snntorch.surrogate,
        optional

    :param surrogate_disable: Disables surrogate gradients regardless of
        ``spike_grad`` argument. Useful for ONNX compatibility. Defaults to
        False
    :type surrogate_disable: bool, Optional

    :param learn_beta: Option to enable learnable beta (via ``tau``).
        Defaults to False
    :type learn_beta: bool, optional

    :param learn_threshold: Option to enable learnable threshold. Defaults to
        False
    :type learn_threshold: bool, optional

    :param state_quant: If specified, hidden state :math:`mem` is quantized to
        a valid state for the forward pass. Defaults to False
    :type state_quant: quantization function from snntorch.quant, optional

    :param output: If ``True``, returns states (and spikes) when the neuron is
        called. If ``False``, returns membrane only. Defaults to True
    :type output: bool, optional

    :param graded_spikes_factor: Output spikes are scaled by this value, if
        specified. Defaults to 1.0
    :type graded_spikes_factor: float or torch.tensor

    :param learn_graded_spikes_factor: Option to enable learnable graded
        spikes. Defaults to False
    :type learn_graded_spikes_factor: bool, optional

    :param kernel_truncation_steps: If set to integer ``K > 0``, keeps the ``K``
        most recent taps of the exponential kernel per output time step, and
        discards older contributions. If ``None``, uses the full time window.
        Defaults to None
    :type kernel_truncation_steps: int or None, optional

    Inputs: \input_
        - **input_** of shape ``(T, B, C)``: time-major input tensor

    Outputs: spk, mem
        - If ``output=True``:
            - **spk** of shape ``(T, B, C)``: output spikes
            - **mem** of shape ``(T, B, C)``: membrane potential
        - If ``output=False``:
            - **mem** of shape ``(T, B, C)``: membrane potential

    Learnable Parameters:
        - **StateLeaky.beta** (via ``tau``) - optional learnable per-channel
          parameter when ``learn_beta=True``
        - **StateLeaky.threshold** - optional learnable threshold when
          ``learn_threshold=True``
        - **StateLeaky.graded_spikes_factor** - optional learnable scaling when
          ``learn_graded_spikes_factor=True``
    """

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
        kernel_truncation_steps=None,
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

        # warn on non-applicable-but-harmless settings when spikes are disabled
        if not output:
            if (
                spike_grad is not None
                or surrogate_disable
                or learn_threshold
                or learn_graded_spikes_factor
                or (
                    isinstance(graded_spikes_factor, torch.Tensor)
                    and not torch.all(graded_spikes_factor == 1.0)
                )
                or (
                    not isinstance(graded_spikes_factor, torch.Tensor)
                    and graded_spikes_factor != 1.0
                )
            ):
                warn(
                    "StateLeaky: spike-related settings are unused when output=False (no spikes emitted).",
                    UserWarning,
                )

        self.kernel_truncation_steps = kernel_truncation_steps

        self._tau_buffer(self.beta, learn_beta, channels)

    def fire_inhibition(self, batch_size, mem):
        raise NotImplementedError(
            "StateLeaky does not support inhibition; use standard Leaky for inhibition paths."
        )

    def mem_reset(self, mem):
        raise NotImplementedError(
            "StateLeaky does not maintain stepwise resets; mem_reset is not applicable."
        )

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

        # determine kernel size (may be truncated)
        if self.kernel_truncation_steps is None:
            kernel_size = num_steps
        else:
            kernel_size = min(self.kernel_truncation_steps, num_steps)

        # time axis shape (1, 1, kernel_size)
        time_steps = torch.arange(kernel_size, device=device).view(
            1, 1, kernel_size
        )
        assert time_steps.shape == (1, 1, kernel_size)

        # single channel case
        if self.tau.shape == () or self.tau.shape == (1,):
            # tau is scalar, broadcast across channels
            tau = self.tau.to(device)
            decay_filter = torch.exp(-time_steps / tau).expand(
                channels, 1, kernel_size
            )
        else:
            # tau is (channels,), reshape to (channels, 1, 1) so it broadcasts correctly
            tau = self.tau.to(device).view(channels, 1, 1)
            assert tau.shape == (channels, 1, 1)
            decay_filter = torch.exp(
                -time_steps / tau
            )  # directly (channels, 1, kernel_size)

        assert decay_filter.shape == (channels, 1, kernel_size)
        assert input_.shape == (batch, channels, num_steps)

        # depthwise convolution: each channel gets its own decay filter
        conv_result = causal_conv1d(input_, decay_filter)
        assert conv_result.shape == (batch, channels, num_steps)

        return conv_result.permute(2, 0, 1)  # (num_steps, batch, channels)

    @property
    def beta(self):
        return (self.tau - 1) / self.tau

    def forward(self, input_):
        mem = self._base_state_function(input_)

        if self.state_quant:
            mem = self.state_quant(mem)

        if self.output:
            self.spk = self.fire(mem) * self.graded_spikes_factor
            return self.spk, mem

        else:
            return mem

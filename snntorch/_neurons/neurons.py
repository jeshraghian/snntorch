from warnings import warn
import torch
import torch.nn as nn

import math


__all__ = [
    "SpikingNeuron",
    "LIF",
    "_SpikeTensor",
    "_SpikeTorchConv",
]

dtype = torch.float


class SpikingNeuron(nn.Module):
    """Parent class for spiking neuron models."""

    instances = []
    """Each :mod:`snntorch.SpikingNeuron` neuron
    (e.g., :mod:`snntorch.Synaptic`) will populate the
    :mod:`snntorch.SpikingNeuron.instances` list with a new entry.
    The list is used to initialize and clear neuron states when the
    argument `init_hidden=True`."""

    reset_dict = {
        "subtract": 0,
        "zero": 1,
        "none": 2,
    }

    def __init__(
        self,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
    ):
        super(SpikingNeuron, self).__init__()

        SpikingNeuron.instances.append(self)
        self.init_hidden = init_hidden
        self.inhibition = inhibition
        self.output = output
        self.surrogate_disable = surrogate_disable

        self._snn_cases(reset_mechanism, inhibition)
        self._snn_register_buffer(
            threshold=threshold,
            learn_threshold=learn_threshold,
            reset_mechanism=reset_mechanism,
            graded_spikes_factor=graded_spikes_factor,
            learn_graded_spikes_factor=learn_graded_spikes_factor,
        )
        self._reset_mechanism = reset_mechanism

        if spike_grad is None:
            self.spike_grad = self.ATan.apply
        else:
            self.spike_grad = spike_grad

        if self.surrogate_disable:
            self.spike_grad = self._surrogate_bypass

        self.state_quant = state_quant

    def fire(self, mem):
        """Generates spike if mem > threshold.
        Returns spk."""

        if self.state_quant:
            mem = self.state_quant(mem)

        mem_shift = mem - self.threshold
        spk = self.spike_grad(mem_shift)

        spk = spk * self.graded_spikes_factor

        return spk

    def fire_inhibition(self, batch_size, mem):
        """Generates spike if mem > threshold, only for the largest membrane.
        All others neurons will be inhibited for that time step.
        Returns spk."""
        mem_shift = mem - self.threshold
        index = torch.argmax(mem_shift, dim=1)
        spk_tmp = self.spike_grad(mem_shift)

        mask_spk1 = torch.zeros_like(spk_tmp)
        mask_spk1[torch.arange(batch_size), index] = 1
        spk = spk_tmp * mask_spk1
        # reset = spk.clone().detach()

        return spk

    def mem_reset(self, mem):
        """Generates detached reset signal if mem > threshold.
        Returns reset."""
        mem_shift = mem - self.threshold
        reset = self.spike_grad(mem_shift).clone().detach()

        return reset

    def _snn_cases(self, reset_mechanism, inhibition):
        self._reset_cases(reset_mechanism)

        if inhibition:
            warn(
                "Inhibition is an unstable feature that has only been tested "
                "for dense (fully-connected) layers. Use with caution!",
                UserWarning,
            )

    def _reset_cases(self, reset_mechanism):
        if (
            reset_mechanism != "subtract"
            and reset_mechanism != "zero"
            and reset_mechanism != "none"
        ):
            raise ValueError(
                "reset_mechanism must be set to either 'subtract', "
                "'zero', or 'none'."
            )

    def _snn_register_buffer(
        self,
        threshold,
        learn_threshold,
        reset_mechanism,
        graded_spikes_factor,
        learn_graded_spikes_factor,
    ):
        """Set variables as learnable parameters else register them in the
        buffer."""

        self._threshold_buffer(threshold, learn_threshold)
        self._graded_spikes_buffer(
            graded_spikes_factor, learn_graded_spikes_factor
        )

        # reset buffer
        try:
            # if reset_mechanism_val is loaded from .pt, override
            # reset_mechanism
            if torch.is_tensor(self.reset_mechanism_val):
                self.reset_mechanism = list(SpikingNeuron.reset_dict)[
                    self.reset_mechanism_val
                ]
        except AttributeError:
            # reset_mechanism_val has not yet been created, create it
            self._reset_mechanism_buffer(reset_mechanism)

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

    def _reset_mechanism_buffer(self, reset_mechanism):
        """Assign mapping to each reset mechanism state.
        Must be of type tensor to store in register buffer. See reset_dict
        for mapping."""
        reset_mechanism_val = torch.as_tensor(
            SpikingNeuron.reset_dict[reset_mechanism]
        )
        self.register_buffer("reset_mechanism_val", reset_mechanism_val)

    def _V_register_buffer(self, V, learn_V):
        if not isinstance(V, torch.Tensor):
            V = torch.as_tensor(V)
        if learn_V:
            self.V = nn.Parameter(V)
        else:
            self.register_buffer("V", V)

    @property
    def reset_mechanism(self):
        """If reset_mechanism is modified, reset_mechanism_val is triggered
        to update.
        0: subtract, 1: zero, 2: none."""
        return self._reset_mechanism

    @reset_mechanism.setter
    def reset_mechanism(self, new_reset_mechanism):
        self._reset_cases(new_reset_mechanism)
        self.reset_mechanism_val = torch.as_tensor(
            SpikingNeuron.reset_dict[new_reset_mechanism]
        )
        self._reset_mechanism = new_reset_mechanism

    @classmethod
    def init(cls):
        """Removes all items from :mod:`snntorch.SpikingNeuron.instances`
        when called."""
        cls.instances = []

    @staticmethod
    def detach(*args):
        """Used to detach input arguments from the current graph.
        Intended for use in truncated backpropagation through time where
        hidden state variables are global variables."""
        for state in args:
            state.detach_()

    @staticmethod
    def zeros(*args):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are global variables."""
        for state in args:
            state = torch.zeros_like(state)

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

    # def atan(alpha=2.0):
    #     """ArcTan surrogate gradient enclosed with a parameterized slope."""
    #     alpha = alpha

    #     def inner(x):
    #         return ATan.apply(x, alpha)

    #     return inner


class LIF(SpikingNeuron):
    """Parent class for leaky integrate and fire neuron models."""

    def __init__(
        self,
        beta,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
    ):
        super().__init__(
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
            graded_spikes_factor,
            learn_graded_spikes_factor,
        )

        self._lif_register_buffer(
            beta,
            learn_beta,
        )
        self._reset_mechanism = reset_mechanism

        if spike_grad is None:
            self.spike_grad = self.ATan.apply
        else:
            self.spike_grad = spike_grad

        if self.surrogate_disable:
            self.spike_grad = self._surrogate_bypass

    def _lif_register_buffer(
        self,
        beta,
        learn_beta,
    ):
        """Set variables as learnable parameters else register them in the
        buffer."""
        self._beta_buffer(beta, learn_beta)

    def _beta_buffer(self, beta, learn_beta):
        if not isinstance(beta, torch.Tensor):
            beta = torch.as_tensor(beta)  # TODO: or .tensor() if no copy
        if learn_beta:
            self.beta = nn.Parameter(beta)
        else:
            self.register_buffer("beta", beta)

    def _V_register_buffer(self, V, learn_V):
        if V is not None:
            if not isinstance(V, torch.Tensor):
                V = torch.as_tensor(V)
        if learn_V:
            self.V = nn.Parameter(V)
        else:
            self.register_buffer("V", V)

    @staticmethod
    def init_leaky():
        """
        Used to initialize mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert
        the hidden states to the same as the input.
        """
        mem = _SpikeTensor(init_flag=False)

        return mem

    @staticmethod
    def init_rleaky():
        """
        Used to initialize spk and mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert
        the hidden states to the same as the input.
        """
        spk = _SpikeTensor(init_flag=False)
        mem = _SpikeTensor(init_flag=False)

        return spk, mem

    @staticmethod
    def init_synaptic():
        """Used to initialize syn and mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert
        the hidden states to the same as the input.
        """

        syn = _SpikeTensor(init_flag=False)
        mem = _SpikeTensor(init_flag=False)

        return syn, mem

    @staticmethod
    def init_rsynaptic():
        """
        Used to initialize spk, syn and mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert
        the hidden states to the same as the input.
        """
        spk = _SpikeTensor(init_flag=False)
        syn = _SpikeTensor(init_flag=False)
        mem = _SpikeTensor(init_flag=False)

        return spk, syn, mem

    @staticmethod
    def init_lapicque():
        """
        Used to initialize mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert
        the hidden states to the same as the input.
        """

        return LIF.init_leaky()

    @staticmethod
    def init_alpha():
        """Used to initialize syn_exc, syn_inh and mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert
        the hidden states to the same as the input.
        """
        syn_exc = _SpikeTensor(init_flag=False)
        syn_inh = _SpikeTensor(init_flag=False)
        mem = _SpikeTensor(init_flag=False)

        return syn_exc, syn_inh, mem
    

class NoisyLIF(SpikingNeuron):
    """Parent class for noisy leaky integrate and fire neuron models."""

    def __init__(
        self,
        beta,
        threshold=1.0,
        noise_type='gaussian',
        noise_scale=0.3,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
    ):
        super().__init__(
            threshold,
            None,
            False,
            init_hidden,
            inhibition,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
            graded_spikes_factor,
            learn_graded_spikes_factor,
        )

        self._lif_register_buffer(
            beta,
            learn_beta,
        )
        self._reset_mechanism = reset_mechanism
        self._noise_scale = noise_scale

        if noise_type == 'gaussian':
            self.spike_grad = self.Gaussian.apply
        elif noise_type == 'logistic':
            self.spike_grad = self.Logistic.apply
        elif noise_type == 'triangular':
            self.spike_grad = self.Triangular.apply
        elif noise_type == 'uniform':
            self.spike_grad = self.Uniform.apply
        elif noise_type == 'atan':
            pass
        else:
            raise ValueError("Invalid noise type. Valid options: gaussian, logistic, triangular, \
                             uniform, atan")

        if self.surrogate_disable:
            self.spike_grad = self._surrogate_bypass

    def _lif_register_buffer(
        self,
        beta,
        learn_beta,
    ):
        """Set variables as learnable parameters else register them in the
        buffer."""
        self._beta_buffer(beta, learn_beta)

    def _beta_buffer(self, beta, learn_beta):
        if not isinstance(beta, torch.Tensor):
            beta = torch.as_tensor(beta)  # TODO: or .tensor() if no copy
        if learn_beta:
            self.beta = nn.Parameter(beta)
        else:
            self.register_buffer("beta", beta)

    def _V_register_buffer(self, V, learn_V):
        if V is not None:
            if not isinstance(V, torch.Tensor):
                V = torch.as_tensor(V)
        if learn_V:
            self.V = nn.Parameter(V)
        else:
            self.register_buffer("V", V)

    @staticmethod
    def init_leaky():
        """
        Used to initialize mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert
        the hidden states to the same as the input.
        """
        mem = _SpikeTensor(init_flag=False)

        return mem

    @staticmethod
    def init_rleaky():
        """
        Used to initialize spk and mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert
        the hidden states to the same as the input.
        """
        spk = _SpikeTensor(init_flag=False)
        mem = _SpikeTensor(init_flag=False)

        return spk, mem

    @staticmethod
    def init_synaptic():
        """Used to initialize syn and mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert
        the hidden states to the same as the input.
        """

        syn = _SpikeTensor(init_flag=False)
        mem = _SpikeTensor(init_flag=False)

        return syn, mem

    @staticmethod
    def init_rsynaptic():
        """
        Used to initialize spk, syn and mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert
        the hidden states to the same as the input.
        """
        spk = _SpikeTensor(init_flag=False)
        syn = _SpikeTensor(init_flag=False)
        mem = _SpikeTensor(init_flag=False)

        return spk, syn, mem

    @staticmethod
    def init_lapicque():
        """
        Used to initialize mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert
        the hidden states to the same as the input.
        """

        return LIF.init_leaky()

    @staticmethod
    def init_alpha():
        """Used to initialize syn_exc, syn_inh and mem as an empty SpikeTensor.
        ``init_flag`` is used as an attribute in the forward pass to convert
        the hidden states to the same as the input.
        """
        syn_exc = _SpikeTensor(init_flag=False)
        syn_inh = _SpikeTensor(init_flag=False)
        mem = _SpikeTensor(init_flag=False)

        return syn_exc, syn_inh, mem

    def mem_reset(self, mem):
        """Generates detached reset signal if mem > threshold.
        Returns reset."""
        mem_shift = mem - self.threshold
        reset = self.spike_grad(mem_shift, 0, self._noise_scale).clone().detach()

        return reset
    
    @staticmethod
    class Gaussian(torch.autograd.Function):
        r"""
        Gaussian noise. This is the original and default type because the iterative form is derived
          from an Ito SDE. Let us denote the cumulative distribution function of the noise by CDF, 
          its probability density function as PDF. 

        **Forward pass:** Probabilistic firing .

            .. math::

                S &~ \\text{Bernoulli}(P(\\text{spiking})) \\\\
                P(\\text{firing}) = CDF$_{\\rm noise}$

        **Backward pass:** Noise-driven learning corresponds to the specified membrane noise.

            .. math::
                    \\frac{∂S}{∂U}&= PDF$_{\\rm noise}$ (U-\\text{threshold})

        Refer to:

        Ma et al. Exploiting Noise as a Resource for Computation and Learning in Spiking Neural 
        Networks. Patterns. Cell Press. 2023. 
        """

        @staticmethod
        def forward(ctx, input_, mu, sigma):
            ctx.save_for_backward(input_)
            ctx.mu = mu
            ctx.sigma = sigma
            p_spike = 1/2 * (
                1 + torch.erf((input_ - mu) / (sigma * math.sqrt(2)))
            )
            return torch.bernoulli(p_spike)

        @staticmethod
        def backward(ctx, grad_output):
            (input_,) = ctx.saved_tensors
            grad_input = grad_output.clone()

            temp = (
                1 / (math.sqrt(2*math.pi) * ctx.sigma)
            ) * torch.exp(
                -0.5 * ((input_ - ctx.mu) / ctx.sigma).pow_(2)
            )
            return grad_input*temp, None, None
    
    @staticmethod
    class Logistic(torch.autograd.Function):
        r"""
        Logistic neuronal noise. The resulting noise-driven learning covers the sigmoidal surrogate
        gradients in training conventional deterministic spiking models. 

        Refer to: 

        Ma et al. Exploiting Noise as a Resource for Computation and Learning in Spiking Neural 
        Networks. Patterns. Cell Press. 2023. 
        """
        @staticmethod
        def forward(ctx, input_, mu, scale):
            ctx.save_for_backward(input_)
            ctx.mu = mu
            ctx.scale = scale
            """ p_spike = 1 / (
                1 + torch.exp(-(input_ - ctx.mu) / ctx.scale)
            ).clamp(0, 1) """
            p_spike = torch.special.expit((input_ - ctx.mu + 1e-8) / (ctx.scale + 1e-8)).nan_to_num_()
            return torch.bernoulli(p_spike)

        @staticmethod
        def backward(ctx, grad_output):
            (input_,) = ctx.saved_tensors
            grad_input = grad_output.clone()

            temp = torch.exp(
                -(input_ - ctx.mu) / ctx.scale
            ) / ctx.scale / (1 + torch.exp(-(input_ - ctx.mu) / ctx.scale)).pow_(2)
            return grad_input*temp, None, None
        
    @staticmethod
    class Triangular(torch.autograd.Function):
        r"""
        Triangular neuronal noise. The resulting noise-driven learning covers the triangular 
        surrogate gradients in training conventional deterministic spiking models. 
        """
        @staticmethod
        def forward(ctx, input_, mu, a):
            ctx.save_for_backward(input_)
            ctx.mu = mu
            ctx.a = a
            mask1 = (input_ < -a).int()
            mask2 = (input_ >= a).int()
            mask3 = ((input_ >= 0) & (input_ < a)).int()
            p_spike = mask2 + \
                (1-mask1)*(1-mask2)*(1-mask3) * (input_ + a)**2 / 2 / a**2 + \
                mask3 * (1 - (input_ - a)**2 / 2 / a**2)
            return torch.bernoulli(p_spike)

        @staticmethod
        def backward(ctx, grad_output):
            (input_,) = ctx.saved_tensors
            grad_input = grad_output.clone()

            mask1 = (input_ < -ctx.a).int()
            mask2 = (input_ >= ctx.a).int()
            temp = (1-mask1)*(1-mask2) * (ctx.a - input_.abs()) / ctx.a**2
            return grad_input*temp, None, None
        
    @staticmethod
    class Uniform(torch.autograd.Function):
        r"""
        Uniform neuronal noise. The resulting noise-driven learning covers the Gate (rectangular) 
        surrogate gradients. 
        """
        @staticmethod
        def forward(ctx, input_, mu, a):
            ctx.save_for_backward(input_)
            ctx.mu = mu
            ctx.a = a

            p_spike = ((input_ - -ctx.a) / (a - -ctx.a)).clamp(0, 1)
            return torch.bernoulli(p_spike)

        @staticmethod
        def backward(ctx, grad_output):
            (input_,) = ctx.saved_tensors
            grad_input = grad_output.clone()

            temp = ((input_ >= -ctx.a).int() & (input_ <= ctx.a).int()) * (
                1 / (ctx.a - -ctx.a)
            )
            return grad_input*temp, None, None


class _SpikeTensor(torch.Tensor):
    """Inherits from torch.Tensor with additional attributes.
    ``init_flag`` is set at the time of initialization.
    When called in the forward function of any neuron, they are parsed and
    replaced with a torch.Tensor variable.
    """

    @staticmethod
    def __new__(cls, *args, init_flag=False, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(
        self,
        *args,
        init_flag=True,
    ):
        # super().__init__() # optional
        self.init_flag = init_flag


def _SpikeTorchConv(*args, input_):
    """Convert SpikeTensor to torch.Tensor of the same size as ``input_``."""

    states = []
    # if len(input_.size()) == 0:
    #     _batch_size = 1  # assume batch_size=1 if 1D input
    # else:
    #     _batch_size = input_.size(0)
    if (
        len(args) == 1 and type(args) is not tuple
    ):  # if only one hidden state, make it iterable
        args = (args,)
    for arg in args:
        arg = arg.to("cpu")
        arg = torch.Tensor(arg)  # wash away the SpikeTensor class
        arg = torch.zeros_like(input_, requires_grad=True)
        states.append(arg)
    if len(states) == 1:  # otherwise, list isn't unpacked
        return states[0]

    return states

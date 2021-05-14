import torch

# Spike-gradient functions

# slope = 25
# """``snntorch.surrogate.slope`` parameterizes the transition rate of the surrogate gradients."""


class FastSigmoid(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of fast sigmoid function.

        .. math::

                S&≈\\frac{U}{1 + k|U|} \\\\
                \\frac{∂S}{∂U}&=\\frac{1}{(1+k|U|)^2}

    :math:`k` defaults to 25, and can be modified by calling ``surrogate.fast_sigmoid(slope=25)``.

    Adapted from:

    *F. Zenke, S. Ganguli (2018) SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks. Neural Computation, pp. 1514-1541.*"""

    @staticmethod
    def forward(ctx, input_, slope=25):
        ctx.save_for_backward(input_)
        ctx.slope = slope
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (ctx.slope * torch.abs(input_) + 1.0) ** 2
        return grad, None


def fast_sigmoid(slope=25):
    """FastSigmoid surrogate gradient enclosed with a parameterized slope."""
    slope = slope

    def inner(x):
        return FastSigmoid.apply(x, slope)

    return inner


class Sigmoid(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of sigmoid function.

        .. math::

                S&≈\\frac{1}{1 + {\\rm exp}(-kU)} \\\\
                \\frac{∂S}{∂U}&=\\frac{k {\\rm exp}(-kU)}{[{\\rm exp}(-kU)+1]^2}

    :math:`k` defaults to 25, and can be modified by calling ``surrogate.sigmoid(slope=25)``.


    Adapted from:

    *F. Zenke, S. Ganguli (2018) SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks. Neural Computation, pp. 1514-1541.*"""

    @staticmethod
    def forward(ctx, input_, slope=25):
        ctx.save_for_backward(input_)
        ctx.slope = slope
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            grad_input
            * ctx.slope
            * torch.exp(-ctx.slope * input_)
            / ((torch.exp(-ctx.slope * input_) + 1) ** 2)
        )
        return grad, None


def sigmoid(slope=25):
    """Sigmoid surrogate gradient enclosed with a parameterized slope."""
    slope = slope

    def inner(x):
        return Sigmoid.apply(x, slope)

    return inner


class SpikeRateEscape(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of Boltzmann Distribution.

        .. math::

                \\frac{∂S}{∂U}=k{\\rm exp}(-β|U-1|)

    :math:`β` defaults to 1, and can be modified by calling ``surrogate.spike_rate_escape(beta=1)``.
    :math:`k` defaults to 25, and can be modified by calling ``surrogate.spike_rate_escape(slope=25)``.


    Adapted from:

    * Wulfram Gerstner and Werner M. Kistler, Spiking neuron models: Single neurons, populations, plasticity. Cambridge University Press, 2002.*"""

    @staticmethod
    def forward(ctx, input_, beta=1, slope=25):
        ctx.save_for_backward(input_)
        ctx.beta = beta
        ctx.slope = slope
        out = (input_ > 0).float()
        return out

    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * ctx.slope * torch.exp(-ctx.beta * torch.abs(input_ - 1))
        return grad, None, None


def spike_rate_escape(beta=1, slope=25):
    """SpikeRateEscape surrogate gradient enclosed with a parameterized slope."""
    beta = beta
    slope = slope

    def inner(x):
        return SpikeRateEscape.apply(x, slope)

    return inner


class SpikeOperator(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Spike operator function.

        .. math::

            S=\\begin{cases} \\frac{U(t)}{U} & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of spike operator.

        .. math::

                \\frac{∂S}{∂U}&=\\begin{cases} \\frac{1}{U} & \\text{if U ≥ U$_{\\rm thr}$} \\\\
                0 & \\text{if U < U$_{\\rm thr}$}
                \\end{cases}

    :math:`U_{\\rm thr}` defaults to 1, and can be modified by calling ``surrogate.spike_operator(threshold=1)``.
    .. warning:: ``threshold`` should match the threshold of the neuron, which defaults to 1 as well.

                """

    @staticmethod
    def forward(ctx, input_, threshold=1):
        out = (input_ > 0).float()
        ctx.save_for_backward(input_, out)
        ctx.threshold = threshold
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_, out) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (grad_input * out) / (input_ + ctx.threshold)
        return grad, None


def spike_operator(threshold=1):
    """Spike operator gradient enclosed with a parameterized threshold."""
    threshold = threshold

    def inner(x):
        return SpikeOperator.apply(x, threshold)

    return inner


class StochasticSpikeOperator(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Spike operator function.

        .. math::

            S=\\begin{cases} \\frac{U(t)}{U} & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of spike operator, where the subthreshold gradient is sampled from uniformly distributed noise on the interval :math:`(𝒰\\sim[-0.5, 0.5)+μ) σ^2`, where :math:`μ` is the mean and :math:`σ^2` is the variance.

        .. math::

                S&≈\\begin{cases} \\frac{U(t)}{U}  & \\text{if U ≥ U$_{\\rm thr}$} \\\\
                (𝒰\\sim[-0.5, 0.5) + μ) σ^2 & \\text{if U < U$_{\\rm thr}$}\\end{cases} \\\\
                \\frac{∂S}{∂U}&=\\begin{cases} \\frac{1}{U}  & \\text{if U ≥ U$_{\\rm thr}$} \\\\
                (𝒰\\sim[-0.5, 0.5) + μ) σ^2 & \\text{if U < U$_{\\rm thr}$}
                \\end{cases}

    :math:`U_{\\rm thr}` defaults to 1, and can be modified by calling ``surrogate.SSO(threshold=1)``.

    :math:`μ` defaults to 0, and can be modified by calling ``surrogate.SSO(mean=0)``.

    :math:`σ^2` defaults to 0.2, and can be modified by calling ``surrogate.SSO(variance=0.5)``.

    The above defaults set the gradient to the following expression:

    .. math::

                \\frac{∂S}{∂U}&=\\begin{cases} \\frac{1}{U}  & \\text{if U ≥ U$_{\\rm thr}$} \\\\
                (𝒰\\sim[-0.1, 0.1) & \\text{if U < U$_{\\rm thr}$}
                \\end{cases}

    .. warning:: ``threshold`` should match the threshold of the neuron, which defaults to 1 as well.

    """

    @staticmethod
    def forward(ctx, input_, threshold=1, mean=0, variance=0.2):
        out = (input_ > 0).float()
        ctx.save_for_backward(input_, out)
        ctx.threshold = threshold
        ctx.mean = mean
        ctx.variance = variance
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_, out) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (grad_input * out) / (input_ + ctx.threshold) + (
            grad_input * (~out.bool()).float()
        ) * ((torch.rand_like(input_) - 0.5 + ctx.mean) * ctx.variance)

        return grad, None, None, None


def SSO(threshold=1, mean=0, variance=0.2):
    """Stochastic spike operator gradient enclosed with a parameterized threshold, mean and variance."""
    threshold = threshold
    mean = mean
    variance = variance

    def inner(x):
        return StochasticSpikeOperator.apply(x, threshold, mean, variance)

    return inner


class LocalStochasticSpikeOperator(torch.autograd.Function):

    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Spike operator function.

        .. math::

            S=\\begin{cases} \\frac{U(t)}{U} & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Local gradient of spike operator, where the subthreshold gradient is sampled from uniformly distributed noise on the interval :math:`(𝒰\\sim[-0.5, 0.5)+μ) σ^2`, where :math:`μ` is the mean and :math:`σ^2` is the variance.

        .. math::

                S&≈\\begin{cases} \\frac{U(t)}{U}\\Big{|}_{U(t)→U_{\\rm thr}}  & \\text{if U ≥ U$_{\\rm thr}$} \\\\
                (𝒰\\sim[-0.5, 0.5) + μ) σ^2 & \\text{if U < U$_{\\rm thr}$}\\end{cases} \\\\
                \\frac{∂S}{∂U}&=\\begin{cases} 1  & \\text{if U ≥ U$_{\\rm thr}$} \\\\
                (𝒰\\sim[-0.5, 0.5) + μ) σ^2 & \\text{if U < U$_{\\rm thr}$}
                \\end{cases}

    :math:`μ` defaults to 0, and can be modified by calling ``surrogate.LSSO(mean=0)``.

    :math:`σ^2` defaults to 0.2, and can be modified by calling ``surrogate.LSSO(variance=0.5)``.

    The above defaults set the gradient to the following expression:

    .. math::

                \\frac{∂S}{∂U}&=\\begin{cases} 1  & \\text{if U ≥ U$_{\\rm thr}$} \\\\
                (𝒰\\sim[-0.1, 0.1) & \\text{if U < U$_{\\rm thr}$}
                \\end{cases}

    """

    @staticmethod
    def forward(ctx, input_, mean=0, variance=0.2):
        out = (input_ > 0).float()
        ctx.save_for_backward(input_, out)
        ctx.mean = mean
        ctx.variance = variance
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_, out) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * out + (grad_input * (~out.bool()).float()) * (
            (torch.rand_like(input_) - 0.5 + ctx.mean) * ctx.variance
        )
        return grad, None, None


def LSSO(mean=0, variance=0.2):
    """Local stochastic spike operator gradient enclosed with a parameterized mean and variance."""
    mean = mean
    variance = variance

    def inner(x):
        return LocalStochasticSpikeOperator.apply(x, mean, variance)

    return inner


class LeakyLocalSpikeOperator(torch.autograd.Function):

    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Spike operator function.

        .. math::

            S=\\begin{cases} \\frac{U(t)}{U} & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Leaky local gradient of spike operator, where the subthreshold gradient is treated as a small constant slope.

        .. math::

                S&≈\\begin{cases} \\frac{U(t)}{U}\\Big{|}_{U(t)→U_{\\rm thr}}  & \\text{if U ≥ U$_{\\rm thr}$} \\\\
                kU & \\text{if U < U$_{\\rm thr}$}\\end{cases} \\\\
                \\frac{∂S}{∂U}&=\\begin{cases} 1  & \\text{if U ≥ U$_{\\rm thr}$} \\\\
                k & \\text{if U < U$_{\\rm thr}$}
                \\end{cases}

    :math:`k` defaults to 0.1, and can be modified by calling ``surrogate.LLSO(slope=0.1)``.

    The gradient is identical to that of a threshold-shifted Leaky ReLU."""

    @staticmethod
    def forward(ctx, input_, slope=0.1):
        out = (input_ > 0).float()
        ctx.save_for_backward(out)
        ctx.slope = slope
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * out + (~out.bool()).float() * ctx.slope * grad_input
        return grad


def LLSO(slope=0.1):
    """Leaky local spike operator gradient enclosed with a parameterized slope."""
    slope = slope

    def inner(x):
        return LocalStochasticSpikeOperator.apply(x, slope)

    return inner


# piecewise linear func
# tanh surrogate func

import torch

# Spike-gradient functions

slope = 25
"""``snntorch.surrogate.slope`` parameterizes the transition rate of the surrogate gradients."""


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

    :math:`k` can be modified by altering ``snntorch.surrogate.slope``.

    Adapted from:

    *F. Zenke, S. Ganguli (2018) SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks. Neural Computation, pp. 1514-1541.*"""

    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (slope * torch.abs(input_) + 1.0) ** 2
        return grad


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

    :math:`k` can be modified by altering ``snntorch.surrogate.slope``.

    Adapted from:

    *F. Zenke, S. Ganguli (2018) SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks. Neural Computation, pp. 1514-1541.*"""

    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            grad_input
            * slope
            * torch.exp(-slope * input_)
            / ((torch.exp(-slope * input_) + 1) ** 2)
        )
        return grad


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

    :math:`β` is parameterized when defining ``snntorch.surrogate.SpikeRateEscape``.
    :math:`k` can be modified by altering ``snntorch.surrogate.slope``.

    Adapted from:

    * Wulfram Gerstner and Werner M. Kistler, Spiking neuron models: Single neurons, populations, plasticity. Cambridge University Press, 2002.*"""

    @staticmethod
    def forward(ctx, input_, beta=1):
        ctx.save_for_backward(input_)
        ctx.beta = beta
        out = (input_ > 0).float()
        return out

    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        beta = ctx.beta
        grad_input = grad_output.clone()
        grad = grad_input * slope * torch.exp(-beta * torch.abs(input_ - 1))
        return grad


class SpikeOperator(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.
    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of spike operator.

        .. math::

                S&≈\\frac{U(t)}{U} \\\\

                \\frac{∂S}{∂U}=\\begin{cases} \\frac{1}{U} \\text{if U ≥ U$_{\\rm thr}$} \\\\
                0 & \\text{if U < U$_{\\rm thr}$}
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
        threshold = ctx.threshold
        grad_input = grad_output.clone()
        grad = (grad_input * out) / (input_ + threshold)
        return grad


class StochasticSpikeOperator(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.
    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of spike operator with uniformly distributed noise on the interval :math:`\\mathcal{U}`[-0.25, 0.25) × var  for subthreshold membrane potentials.

        .. math::
it s
                S&≈\\frac{U(t)}{U} \\\\

                \\frac{∂S}{∂U}=\\begin{cases} \\frac{1}{U} \\text{if U ≥ U$_{\\rm thr}$} \\\\
                U[-0.25, 0.25) & \\text{if U < U$_{\\rm thr}$}
                \\end{cases}
                """

    @staticmethod
    def forward(ctx, input_, threshold=1, variance=0.25):
        out = (input_ > 0).float()
        ctx.save_for_backward(input_, out)
        ctx.threshold = threshold
        ctx.variance = variance
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_, out) = ctx.saved_tensors
        threshold = ctx.threshold
        variance = ctx.variance
        grad_input = grad_output.clone()
        grad = (grad_input * out) / (input_ + threshold) + (
            grad_input * (~out.bool()).float()
        ) * ((torch.rand_like(input_) - 0.5) * variance)
        # grad += ((torch.rand(input_.size()) - 0.5) * variance) * (~grad.bool()).float()

        return grad


class LocalStochasticSpikeOperator(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.
    **Forward pass:** Heaviside step function shifted.

        .. math::

                S&≈\\frac{U(t)}{U} \\\\

    **Backward pass:** Local gradient of spike operator with uniformly distributed noise on the interval :math:`[-0.25, 0.25) × var`  for subthreshold membrane potentials.

        .. math::

                S&≈\\frac{U(t)}{U} \\\\

                \\frac{∂S}{∂U}=\\begin{cases} 1 \\text{if U ≥ U$_{\\rm thr}$} \\\\
                U(-0.25, 0.25) & \\text{if U < U$_{\\rm thr}$}
                \\end{cases}
                """

    @staticmethod
    def forward(ctx, input_, variance=0.1):
        out = (input_ > 0).float()
        ctx.save_for_backward(input_, out)
        ctx.variance = variance
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_, out) = ctx.saved_tensors
        variance = ctx.variance
        grad_input = grad_output.clone()
        grad = grad_input * out + (grad_input * (~out.bool()).float()) * (
            (torch.rand_like(input_) - 0.5) * variance
        )
        return grad


class LeakyLocalSpikeOperator(torch.autograd.Function):
    """Default spiking function for neuron.

        **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            leaky & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

        **Backward pass:** Heaviside step function shifted with leakage term.

        .. math::

            \\frac{∂S}{∂U}=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            var & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}


        Same rationale as the Heaviside gradient, but with the Leaky ReLU gradient."""

    @staticmethod
    def forward(ctx, input_, leaky=0.1):
        out = (input_ > 0).float()
        ctx.save_for_backward(out)
        ctx.leaky = leaky
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        leaky = ctx.leaky
        grad_input = grad_output.clone()
        grad = grad_input * out + (~out.bool()).float() * leaky * grad_input
        return grad


# piecewise linear func
# tanh surrogate func

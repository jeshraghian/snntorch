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

    def __init__(self, beta=1):
        self.beta = beta

    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        out = (input_ > 0).float()
        return out

    def backward(self, ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * slope * torch.exp(-self.beta * torch.abs(input_ - 1))
        return grad


# piecewise linear func
# tanh surrogate func

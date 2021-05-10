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
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        input_[input_ > 0] = grad_input / input_
        input_[input_ <= 0] = 0
        grad = input_.clone()
        return grad


class StochasticSpikeOperator(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.
    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of spike operator with uniformly distributed noise on the interval :math:`[-0.25, 0.25) × var`  for subthreshold membrane potentials.

        .. math::

                S&≈\\frac{U(t)}{U} \\\\

                \\frac{∂S}{∂U}=\\begin{cases} \\frac{1}{U} \\text{if U ≥ U$_{\\rm thr}$} \\\\
                U[-0.25, 0.25) & \\text{if U < U$_{\\rm thr}$}
                \\end{cases}
                """

    def __init__(self, var=0.25):
        super(StochasticSpikeOperator, self).__init__(var)

        self.var = var

    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(self, ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        input_[input_ > 0] = grad_input / input_
        input_[input_ <= 0] = (torch.rand(input_.size()) - 0.5) * self.var
        grad = input_.clone()
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

    def __init__(self, var=0.25):
        super(StochasticSpikeOperator, self).__init__(var)

        self.var = var

    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(self, ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        input_[input_ > 0] = grad_input
        input_[input_ <= 0] = (torch.rand(input_.size()) - 0.5) * self.var
        grad = input_.clone()
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

    def __init__(self, leaky=0.25):

        super(LeakyLocalSpikeOperator, self).__init__(leaky)

        self.leaky = leaky

    @staticmethod
    def forward(ctx, input_):
        out = (input_ > 0).float()
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(self, ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        input_[input_ > 0] = grad_input
        input_[input_ <= 0] = grad_input * self.leaky
        grad = input_.clone()
        return grad


# piecewise linear func
# tanh surrogate func

import torch
import math

# Spike-gradient functions

# slope = 25
# """``snntorch.surrogate.slope``
# parameterizes the transition rate of the surrogate gradients."""


class StraightThroughEstimator(torch.autograd.Function):
    """
    Straight Through Estimator.

    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of fast sigmoid function.

        .. math::

                \\frac{∂S}{∂U}=1


    """

    @staticmethod
    def forward(ctx, input_):
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


def straight_through_estimator():
    """Straight Through Estimator surrogate gradient enclosed
    with a parameterized slope."""

    def inner(x):
        return StraightThroughEstimator.apply(x)

    return inner


class Triangular(torch.autograd.Function):
    """
    Triangular Surrogate Gradient.

    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of the triangular function.

        .. math::

                \\frac{∂S}{∂U}=\\begin{cases} U_{\\rm thr} &
                \\text{if U < U$_{\\rm thr}$} \\\\
                -U_{\\rm thr}  & \\text{if U ≥ U$_{\\rm thr}$}
                \\end{cases}


    """

    @staticmethod
    def forward(ctx, input_, threshold):
        ctx.save_for_backward(input_)
        ctx.threshold = threshold
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * ctx.threshold
        grad[input_ >= 0] = -grad[input_ >= 0]
        return grad, None


def triangular(threshold=1):
    """Triangular surrogate gradient enclosed with
    a parameterized threshold."""
    threshold = threshold

    def inner(x):
        return Triangular.apply(x, threshold)

    return inner


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

    :math:`k` defaults to 25, and can be modified by calling \
        ``surrogate.fast_sigmoid(slope=25)``.

    Adapted from:

    *F. Zenke, S. Ganguli (2018) SuperSpike: Supervised Learning in
    Multilayer Spiking Neural Networks. Neural Computation, pp. 1514-1541.*"""

    @staticmethod
    def forward(ctx, input_, slope):
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
                \\frac{∂S}{∂U}&=\\frac{1}{π}\\frac{1}{(1+(πU\\frac{α}{2})^2)}


    α defaults to 2, and can be modified by calling \
        ``surrogate.atan(alpha=2)``.

    Adapted from:

    *W. Fang, Z. Yu, Y. Chen, T. Masquelier, T. Huang,
    Y. Tian (2021) Incorporating Learnable Membrane Time Constants
    to Enhance Learning of Spiking Neural Networks. Proc. IEEE/CVF
    Int. Conf. Computer Vision (ICCV), pp. 2661-2671.*"""

    @staticmethod
    def forward(ctx, input_, alpha):
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


def atan(alpha=2.0):
    """ArcTan surrogate gradient enclosed with a parameterized slope."""
    alpha = alpha

    def inner(x):
        return ATan.apply(x, alpha)

    return inner


# @staticmethod
class Heaviside(torch.autograd.Function):
    """Default spiking function for neuron.

    **Forward pass:** Heaviside step function shifted.

    .. math::

        S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
        0 & \\text{if U < U$_{\\rm thr}$}
        \\end{cases}

    **Backward pass:** Heaviside step function shifted.

    .. math::

        \\frac{∂S}{∂U}=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
        0 & \\text{if U < U$_{\\rm thr}$}
        \\end{cases}

    Although the backward pass is clearly not the analytical
    solution of the forward pass, this assumption holds true
    on the basis that a reset necessarily occurs after a spike
    is generated when :math:`U ≥ U_{\\rm thr}`."""

    @staticmethod
    def forward(ctx, input_):
        out = (input_ > 0).float()
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        grad = grad_output * out
        return grad


def heaviside():
    """Heaviside surrogate gradient wrapper."""

    def inner(x):
        return Heaviside.apply(x)

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
                \\frac{∂S}{∂U}&=\\frac{k
                {\\rm exp}(-kU)}{[{\\rm exp}(-kU)+1]^2}

    :math:`k` defaults to 25, and can be modified by calling \
        ``surrogate.sigmoid(slope=25)``.


    Adapted from:

    *F. Zenke, S. Ganguli (2018) SuperSpike: Supervised Learning
    in Multilayer Spiking
    Neural Networks. Neural Computation, pp. 1514-1541.*"""

    @staticmethod
    def forward(ctx, input_, slope):
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

    :math:`β` defaults to 1, and can be modified by calling \
        ``surrogate.spike_rate_escape(beta=1)``.
    :math:`k` defaults to 25, and can be modified by calling \
        ``surrogate.spike_rate_escape(slope=25)``.


    Adapted from:

    * Wulfram Gerstner and Werner M. Kistler,
    Spiking neuron models: Single neurons, populations, plasticity.
    Cambridge University Press, 2002.*"""

    @staticmethod
    def forward(ctx, input_, beta, slope):
        ctx.save_for_backward(input_)
        ctx.beta = beta
        ctx.slope = slope
        out = (input_ > 0).float()
        return out

    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            grad_input
            * ctx.slope
            * torch.exp(-ctx.beta * torch.abs(input_ - 1))
        )
        return grad, None, None


def spike_rate_escape(beta=1, slope=25):
    """SpikeRateEscape surrogate gradient
    enclosed with a parameterized slope."""
    beta = beta
    slope = slope

    def inner(x):
        return SpikeRateEscape.apply(x, beta, slope)

    return inner


class StochasticSpikeOperator(torch.autograd.Function):

    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Spike operator function.

        .. math::

            S=\\begin{cases} \\frac{U(t)}{U}
            & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of spike operator,
    where the subthreshold gradient is sampled from uniformly
    distributed noise on the interval :math:`(𝒰\\sim[-0.5, 0.5)+μ) σ^2`,
    where :math:`μ` is the mean and :math:`σ^2` is the variance.

        .. math::

            S&≈\\begin{cases} \\frac{U(t)}{U}\\Big{|}_{U(t)→U_{\\rm thr}}
            & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            (𝒰\\sim[-0.5, 0.5) + μ) σ^2 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases} \\\\
            \\frac{∂S}{∂U}&=\\begin{cases} 1  & \\text{if U ≥ U$_{\\rm thr}$}
            \\\\
            (𝒰\\sim[-0.5, 0.5) + μ) σ^2 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    :math:`μ` defaults to 0, and can be modified by calling \
        ``surrogate.SSO(mean=0)``.

    :math:`σ^2` defaults to 0.2, and can be modified by calling \
        ``surrogate.SSO(variance=0.5)``.

    The above defaults set the gradient to the following expression:

    .. math::

                \\frac{∂S}{∂U}&=\\begin{cases} 1
                & \\text{if U ≥ U$_{\\rm thr}$} \\\\
                (𝒰\\sim[-0.1, 0.1) & \\text{if U < U$_{\\rm thr}$}
                \\end{cases}

    """

    @staticmethod
    def forward(ctx, input_, mean, variance):
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


def SSO(mean=0, variance=0.2):
    """Stochastic spike operator gradient enclosed with a parameterized mean
    and variance."""
    mean = mean
    variance = variance

    def inner(x):
        return StochasticSpikeOperator.apply(x, mean, variance)

    return inner


class LeakySpikeOperator(torch.autograd.Function):

    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Spike operator function.

        .. math::

            S=\\begin{cases} \\frac{U(t)}{U} & \\text{if U ≥ U$_{\\rm thr}$}
            \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Leaky gradient of spike operator, where
    the subthreshold gradient is treated as a small constant slope.

        .. math::

                S&≈\\begin{cases} \\frac{U(t)}{U}\\Big{|}_{U(t)→U_{\\rm thr}}
                & \\text{if U ≥ U$_{\\rm thr}$} \\\\
                kU & \\text{if U < U$_{\\rm thr}$}\\end{cases} \\\\
                \\frac{∂S}{∂U}&=\\begin{cases} 1
                & \\text{if U ≥ U$_{\\rm thr}$} \\\\
                k & \\text{if U < U$_{\\rm thr}$}
                \\end{cases}

    :math:`k` defaults to 0.1, and can be modified by calling \
        ``surrogate.LSO(slope=0.1)``.

    The gradient is identical to that of a threshold-shifted Leaky ReLU."""

    @staticmethod
    def forward(ctx, input_, slope):
        out = (input_ > 0).float()
        ctx.save_for_backward(out)
        ctx.slope = slope
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            grad_input * out + (~out.bool()).float() * ctx.slope * grad_input
        )
        return grad


def LSO(slope=0.1):
    """Leaky spike operator gradient enclosed with a parameterized slope."""
    slope = slope

    def inner(x):
        return StochasticSpikeOperator.apply(x, slope)

    return inner


class SparseFastSigmoid(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of fast sigmoid function clipped below B.

        .. math::

                S&≈\\frac{U}{1 + k|U|}H(U-B) \\\\
                \\frac{∂S}{∂U}&=\\begin{cases} \\frac{1}{(1+k|U|)^2}
                & \\text{\\rm if U > B}
                0 & \\text{\\rm otherwise}
                \\end{cases}

    :math:`k` defaults to 25, and can be modified by calling \
        ``surrogate.SFS(slope=25)``.
    :math:`B` defaults to 1, and can be modified by calling \
        ``surrogate.SFS(B=1)``.

    Adapted from:

    *N. Perez-Nieves and D.F.M. Goodman (2021) Sparse Spiking
    Gradient Descent. https://arxiv.org/pdf/2105.08810.pdf.*"""

    @staticmethod
    def forward(ctx, input_, slope, B):
        ctx.save_for_backward(input_)
        ctx.slope = slope
        ctx.B = B
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            grad_input
            / (ctx.slope * torch.abs(input_) + 1.0) ** 2
            * (input_ > ctx.B).float()
        )
        return grad, None, None


def SFS(slope=25, B=1):
    """SparseFastSigmoid surrogate gradient enclosed with a
    parameterized slope and sparsity threshold."""
    slope = slope
    B = B

    def inner(x):
        return SparseFastSigmoid.apply(x, slope, B)

    return inner


class CustomSurrogate(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Spike operator function.

        .. math::

            S=\\begin{cases} \\frac{U(t)}{U} & \\text{if U ≥ U$_{\\rm thr}$}
            \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** User-defined custom surrogate gradient function.

    The user defines the custom surrogate gradient in a separate function.
    It is passed in the forward static method and used in the backward
    static method.

    The arguments of the custom surrogate gradient function are always
    the input of the forward pass (input_), the gradient of the input 
    (grad_input) and the output of the forward pass (out).
    
    ** Important Note: The hyperparameters of the custom surrogate gradient
    function have to be defined inside of the function itself. **

    Example::

        import torch
        import torch.nn as nn
        import snntorch as snn
        from snntorch import surrogate

        def custom_fast_sigmoid(input_, grad_input, spikes):
            ## The hyperparameter slope is defined inside the function.
            slope = 25
            grad = grad_input / (slope * torch.abs(input_) + 1.0) ** 2
            return grad

        spike_grad = surrogate.custom_surrogate(custom_fast_sigmoid)

        net_seq = nn.Sequential(nn.Conv2d(1, 12, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta,
                            spike_grad=spike_grad,
                            init_hidden=True),
                    nn.Conv2d(12, 64, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta,
                            spike_grad=spike_grad,
                            init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(64*4*4, 10),
                    snn.Leaky(beta=beta,
                            spike_grad=spike_grad,
                            init_hidden=True,
                            output=True)
                    ).to(device)

    """

    @staticmethod
    def forward(ctx, input_, custom_surrogate_function):
        out = (input_ > 0).float()
        ctx.save_for_backward(input_, out)
        ctx.custom_surrogate_function = custom_surrogate_function
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_, out = ctx.saved_tensors
        custom_surrogate_function = ctx.custom_surrogate_function

        grad_input = grad_output.clone()
        grad = custom_surrogate_function(input_, grad_input, out)
        return grad, None


def custom_surrogate(custom_surrogate_function):
    """Custom surrogate gradient enclosed within a wrapper."""
    func = custom_surrogate_function

    def inner(data):
        return CustomSurrogate.apply(data, func)

    return inner


# class InverseSpikeOperator(torch.autograd.Function):
#     """
#     Surrogate gradient of the Heaviside step function.

#     **Forward pass:** Spike operator function.

#         .. math::

#             S=\\begin{cases} \\frac{U(t)}{U} & \\text{if U ≥
#             U$_{\\rm thr}$} \\\\
#             0 & \\text{if U < U$_{\\rm thr}$}
#             \\end{cases}

#     **Backward pass:** Gradient of spike operator.

#         .. math::

#                 \\frac{∂S}{∂U}&=\\begin{cases} \\frac{1}{U}
#                 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
#                 0 & \\text{if U < U$_{\\rm thr}$}
#                 \\end{cases}

#     :math:`U_{\\rm thr}` defaults to 1, and can be modified by calling
#     ``surrogate.spike_operator(threshold=1)``.
#     .. warning:: ``threshold`` should match the threshold of the neuron,
#     which defaults to 1 as well.

#                 """

#     @staticmethod
#     def forward(ctx, input_, threshold=1):
#         out = (input_ > 0).float()
#         ctx.save_for_backward(input_, out)
#         ctx.threshold = threshold
#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         (input_, out) = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         grad = (grad_input * out) / (input_ + ctx.threshold)
#         return grad, None


# def inverse_spike_operator(threshold=1):
#     """Spike operator gradient enclosed with a parameterized threshold."""
#     threshold = threshold

#     def inner(x):
#         return InverseSpikeOperator.apply(x, threshold)

#     return inner


# class InverseStochasticSpikeOperator(torch.autograd.Function):
#     """
#     Surrogate gradient of the Heaviside step function.

#     **Forward pass:** Spike operator function.

#         .. math::

#             S=\\begin{cases} \\frac{U(t)}{U}
#             & \\text{if U ≥ U$_{\\rm thr}$} \\\\
#             0 & \\text{if U < U$_{\\rm thr}$}
#             \\end{cases}

#     **Backward pass:** Gradient of spike operator,
#     where the subthreshold gradient is sampled from
#     uniformly distributed noise on the interval
#     :math:`(𝒰\\sim[-0.5, 0.5)+μ) σ^2`,
#     where :math:`μ` is the mean and :math:`σ^2` is the variance.

#         .. math::

#                 S&≈\\begin{cases} \\frac{U(t)}{U}
#                 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
#                 (𝒰\\sim[-0.5, 0.5) + μ) σ^2
#                 & \\text{if U < U$_{\\rm thr}$}\\end{cases} \\\\
#                 \\frac{∂S}{∂U}&=\\begin{cases} \\frac{1}{U}
#                 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
#                 (𝒰\\sim[-0.5, 0.5) + μ) σ^2
#                 & \\text{if U < U$_{\\rm thr}$}
#                 \\end{cases}

#     :math:`U_{\\rm thr}` defaults to 1, and can be modified by calling
#     ``surrogate.SSO(threshold=1)``.

#     :math:`μ` defaults to 0, and can be modified by calling
#     ``surrogate.SSO(mean=0)``.

#     :math:`σ^2` defaults to 0.2, and can be modified by calling
#     ``surrogate.SSO(variance=0.5)``.

#     The above defaults set the gradient to the following expression:

#     .. math::

#                 \\frac{∂S}{∂U}&=\\begin{cases} \\frac{1}{U}
#                 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
#                 (𝒰\\sim[-0.1, 0.1) & \\text{if U < U$_{\\rm thr}$}
#                 \\end{cases}

#     .. warning:: ``threshold`` should match the threshold of the neuron,
#     which defaults to 1 as well.

#     """

#     @staticmethod
#     def forward(ctx, input_, threshold=1, mean=0, variance=0.2):
#         out = (input_ > 0).float()
#         ctx.save_for_backward(input_, out)
#         ctx.threshold = threshold
#         ctx.mean = mean
#         ctx.variance = variance
#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         (input_, out) = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         grad = (grad_input * out) / (input_ + ctx.threshold) + (
#             grad_input * (~out.bool()).float()
#         ) * ((torch.rand_like(input_) - 0.5 + ctx.mean) * ctx.variance)

#         return grad, None, None, None


# def ISSO(threshold=1, mean=0, variance=0.2):
#     """Stochastic spike operator gradient enclosed with a parameterized
#     threshold, mean and variance."""
#     threshold = threshold
#     mean = mean
#     variance = variance

#     def inner(x):
#         return InverseStochasticSpikeOperator.
#         apply(x, threshold, mean, variance)

#     return inner


# piecewise linear func
# tanh surrogate func

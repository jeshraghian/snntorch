import torch.nn as nn


class _BatchNormTTList(nn.ModuleList):
    """ModuleList holding one batch-norm layer per time step.

    Apply the layer for time step ``step`` with ``bntt(x, step)`` or,
    equivalently, ``bntt[step](x)``. Indexing and state_dict keys are those
    of a plain ``torch.nn.ModuleList``.
    """

    def forward(self, x, step=None):
        if step is None:
            raise TypeError(
                "BatchNormTT holds one batch-norm layer per time step; "
                "call it as bntt(x, step) or bntt[step](x)"
            )
        return self[step](x)


def BatchNormTT1d(
    input_features, time_steps, eps=1e-5, momentum=0.1, affine=True
):
    """
    Generate a torch.nn.ModuleList of 1D Batch Normalization Layer with
    length time_steps.
    Input to this layer is the same as the  vanilla torch.nn.BatchNorm1d
    layer.

    Batch Normalisation Through Time (BNTT) as presented in:
    'Revisiting Batch Normalization for Training Low-Latency Deep Spiking
    Neural Networks From Scratch'
    By Youngeun Kim & Priyadarshini Panda
    arXiv preprint arXiv:2010.01729

    Original GitHub repo:
    https://github.com/Intelligent-Computing-Lab-Yale/
    BNTT-Batch-Normalization-Through-Time

    Using LIF neuron as the neuron of choice for the math shown below.

    Typically, for a single post-synaptic neuron i, we can represent its
    membrane potential :math:`U_{i}^{t}` at time-step t as:

    .. math::

            U_{i}^{t} = λ u_{i}^{t-1} + \\sum_j w_{ij}S_{j}^{t}

    where:

    * λ - a leak factor which is less than one
    * j - the index of the pre-synaptic neuron
    * :math:`S_{j}` - the binary spike activation
    * :math:`w_{ij}` - the weight of the connection between the pre & \
    post neurons.

    With Batch Normalization Throught Time, the membrane potential can be
    modeled as:

    .. math::

            U_{i}^{t} = λu_{i}^{t-1} + BNTT_{γ^{t}}

                      = λu_{i}^{t-1} + γ _{i}^{t} (\\frac{\\sum_j
                      w_{ij}S_{j}^{t} -
                      µ_{i}^{t}}{\\sqrt{(σ _{i}^{t})^{2} + ε}})

    :param input_features: number of features of the input
    :type input_features: int

    :param time_steps: number of time-steps of the SNN
    :type time_steps: int

    :param eps: a value added to the denominator for numerical stability
    :type eps: float

    :param momentum: the value used for the running_mean and running_var \
    computation
    :type momentum: float

    :param affine: a boolean value that when set to True, the Batch Norm \
    layer will have learnable affine parameters
    :type affine: bool

    Example::

        bntt = snn.BatchNormTT1d(64, time_steps=25)
        for step in range(25):
            cur = bntt(fc(x[step]), step)  # same as bntt[step](fc(x[step]))

    Inputs: input_features, time_steps
        - **input_features**: same number of features as the input
        - **time_steps**: the number of time-steps to unroll in the SNN

    Outputs: bntt
        -  **bntt** of shape `(time_steps)`: torch.nn.ModuleList of \
        BatchNorm1d layers for the specified number of time-steps

    """
    bntt = _BatchNormTTList(
        [
            nn.BatchNorm1d(
                input_features, eps=eps, momentum=momentum, affine=affine
            )
            for _ in range(time_steps)
        ]
    )

    # Disable bias/beta of Batch Norm
    for bn in bntt:
        bn.bias = None

    return bntt


def BatchNormTT2d(
    input_features, time_steps, eps=1e-5, momentum=0.1, affine=True
):
    """
    Generate a torch.nn.ModuleList of 2D Batch Normalization Layer with
    length time_steps.
    Input to this layer is the same as the  vanilla torch.nn.BatchNorm2d layer.

    Batch Normalisation Through Time (BNTT) as presented in:
    'Revisiting Batch Normalization for Training Low-Latency Deep Spiking
    Neural Networks From Scratch'
    By Youngeun Kim & Priyadarshini Panda
    arXiv preprint arXiv:2010.01729

    Using LIF neuron as the neuron of choice for the math shown below.

    Typically, for a single post-synaptic neuron i, we can represent its
    membrane potential :math:`U_{i}^{t}` at time-step t as:

    .. math::

            U_{i}^{t} = λ u_{i}^{t-1} + \\sum_j w_{ij}S_{j}^{t}

    where:

    * λ - a leak factor which is less than one
    * j - the index of the pre-synaptic neuron
    * :math:`S_{j}` - the binary spike activation
    * :math:`w_{ij}` - the weight of the connection between the pre & post \
    neurons.

    With Batch Normalization Throught Time, the membrane potential can be \
    modeled as:

    .. math::

            U_{i}^{t} = λ u_{i}^{t-1} + BNTT_{γ^{t}}

                      = λ u_{i}^{t-1}
                      + γ_{i}^{t} (\\frac{\\sum_j
                      w_{ij}S_{j}^{t}
                      - µ_{i}^{t}}{\\sqrt{(σ _{i}^{t})^{2} + ε}})

    :param input_features: number of channels of the input
    :type input_features: int

    :param time_steps: number of time-steps of the SNN
    :type time_steps: int

    :param eps: a value added to the denominator for numerical stability
    :type eps: float

    :param momentum: the value used for the running_mean and running_var \
        computation
    :type momentum: float

    :param affine: a boolean value that when set to True, the Batch Norm \
        layer will have learnable affine parameters
    :type affine: bool

    Example::

        bntt = snn.BatchNormTT2d(64, time_steps=25)
        for step in range(25):
            cur = bntt(conv(x[step]), step)  # same as bntt[step](conv(x[step]))

    Inputs: input_features, time_steps
        - **input_features**: same number of channels as the input
        - **time_steps**: the number of time-steps to unroll in the SNN

    Outputs: bntt
        -  **bntt** of shape `(time_steps)`: torch.nn.ModuleList of \
        BatchNorm2d layers for the specified number of time-steps

    """
    bntt = _BatchNormTTList(
        [
            nn.BatchNorm2d(
                input_features, eps=eps, momentum=momentum, affine=affine
            )
            for _ in range(time_steps)
        ]
    )

    # Disable bias/beta of Batch Norm
    for bn in bntt:
        bn.bias = None

    return bntt

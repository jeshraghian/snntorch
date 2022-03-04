import torch


class StateQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, levels):

        device = "cpu"
        if input_.is_cuda:
            device = "cuda"

        levels = levels.to(device)

        size = input_.size()
        input_ = input_.flatten()

        # Broadcast mem along new direction same # of times as num_levels
        repeat_dims = torch.ones(len(input_.size())).tolist()
        repeat_dims.append(len(levels))
        repeat_dims = [int(item) for item in repeat_dims]
        repeat_dims = tuple(repeat_dims)
        input_ = input_.unsqueeze(-1).repeat(repeat_dims)

        # find closest valid quant state
        idx_match = torch.min(torch.abs(levels - input_), dim=-1)[1]
        quant_tensor = levels[idx_match]

        return quant_tensor.reshape(size)

    # STE
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


def state_quant(
    num_bits=8,
    uniform=True,
    thr_centered=True,
    threshold=1,
    lower_limit=0,
    upper_limit=0.2,
    multiplier=None,
):
    """Quantize State Variables.

    Example::

        import torch
        import snntorch as snn
        from snntorch import quant

        beta = 0.5

        quant_mem = quant.state_quant(num_bits=4, uniform=True)

        # mem will now be quantized. By default, uniform quantization is applied between -(thr + thr*lower_limit) and (thr + thr*upper_limit)
        # This can be changed to non-uniform by setting `uniform=False`
        lif = snn.Leaky(beta=beta, state_quant=quant_mem)

        rand_input = torch.rand(1)
        mem = lif.init_leaky()

        # forward-pass for one step
        spk, mem = lif(rand_input, mem)

    """
    num_levels = 2 << num_bits - 1

    # linear / uniform quantization - ignores thr_centered
    if uniform:
        levels = torch.linspace(
            -threshold - threshold * lower_limit,
            threshold + threshold * upper_limit,
            num_levels,
        )

    # exponential / non-uniform quantization
    else:
        if multiplier is None:
            if num_bits == 1:
                multiplier = 0.05
            if num_bits == 2:
                multiplier = 0.1
            elif num_bits == 3:
                multiplier = 0.3
            elif num_bits == 4:
                multiplier = 0.5
            elif num_bits == 5:
                multiplier = 0.7
            elif num_bits == 6:
                multiplier = 0.9
            elif num_bits == 7:
                multiplier = 0.925
            elif num_bits > 7:
                multiplier = 0.95

        # asymmetric: shifted to threshold
        if thr_centered:
            levels = torch.tensor(
                [multiplier ** j for j in reversed(range(num_levels))]
            )  # .to(device)
            levels = (-levels - min(-levels)) * (
                threshold * upper_limit + threshold * lower_limit
            ) - (threshold - threshold * lower_limit)

        # centered about zero
        else:
            levels = sum(
                [
                    [-(multiplier ** j) for j in range(num_levels >> 1)],
                    [multiplier ** j for j in reversed(range(num_levels >> 1))],
                ],
                [],
            )
            levels = (levels - min(levels)) * (
                threshold * upper_limit + threshold * lower_limit
            ) - (threshold - threshold * lower_limit)

    def inner(x):
        return StateQuant.apply(x, levels)

    return inner

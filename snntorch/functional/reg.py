import torch


class l1_rate_sparsity:
    """L1 regularization using total spike count as the penalty term.
    Lambda is a scalar factor for regularization."""

    def __init__(self, Lambda=1e-5):
        self.Lambda = Lambda
        self.__name__ = "l1_rate_sparsity"

    def __call__(self, spk_out):
        return self.Lambda * torch.sum(spk_out)


class kl_rate_sparsity:
    """Kullback-Leibler (KL) divergence regularization targeting a specific spike firing rate.

    Calculates KL divergence penalty between actual average spike rate per neuron and target_rate:

        .. math::

            KL(p || q) = p \\log \\frac{p}{q + \\epsilon} + (1 - p) \\log \\frac{1 - p}{1 - q + \\epsilon}

    :param target_rate: Target firing rate ratio (e.g. 0.05 for 5% average spiking), defaults to 0.05
    :type target_rate: float, optional
    :param Lambda: Regularization strength multiplier, defaults to 1e-4
    :type Lambda: float, optional
    """

    def __init__(self, target_rate=0.05, Lambda=1e-4, eps=1e-8):
        self.target_rate = target_rate
        self.Lambda = Lambda
        self.eps = eps
        self.__name__ = "kl_rate_sparsity"

    def __call__(self, spk_out):
        if spk_out.dim() >= 2:
            mean_rate = torch.mean(spk_out.float(), dim=0)
        else:
            mean_rate = spk_out.float()

        p = torch.tensor(self.target_rate, dtype=mean_rate.dtype, device=mean_rate.device)
        p = torch.clamp(p, self.eps, 1.0 - self.eps)
        q = torch.clamp(mean_rate, self.eps, 1.0 - self.eps)

        kl = p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))
        return self.Lambda * torch.sum(kl)


class temporal_sparsity:
    """Temporal burst sparsity penalty to encourage smooth, non-bursting temporal firing.

    Penalizes squared differences of consecutive time-step spike outputs.

    :param Lambda: Regularization multiplier, defaults to 1e-5
    :type Lambda: float, optional
    """

    def __init__(self, Lambda=1e-5):
        self.Lambda = Lambda
        self.__name__ = "temporal_sparsity"

    def __call__(self, spk_out):
        if spk_out.dim() < 2 or spk_out.size(0) <= 1:
            return torch.tensor(0.0, device=spk_out.device, dtype=spk_out.dtype)
        diff = spk_out[1:] - spk_out[:-1]
        return self.Lambda * torch.sum(diff.pow(2))


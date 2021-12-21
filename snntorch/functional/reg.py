import torch


class l1_rate_sparsity:
    """L1 regularization using total spike count as the penalty term.
    Lambda is a scalar factor for regularization."""

    def __init__(self, Lambda=1e-5):
        self.Lambda = Lambda
        self.__name__ = "l1_rate_sparsity"

    def __call__(self, spk_out):
        return self.Lambda * torch.sum(spk_out)


# # def l2_sparsity(mem_out, Lambda=1e-6):
# #     """L2 regularization using accumulated membrane potential as the penalty term."""
# #     return Lambda * (torch.sum(mem_out)**2)

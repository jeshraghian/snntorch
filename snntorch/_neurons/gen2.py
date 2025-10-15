import torch
from torch import nn


class StateOuterProductCumsum(nn.Module):
    r"""
    Generation 2: Outer Product State with Decay (vectorized version)
    Parallel time update using cumulative sums, no explicit loop over T.
    """

    def __init__(self, truncation_steps=None):
        super().__init__()
        self.truncation_steps = truncation_steps

    def forward(self, v, k, alpha):
        """
        v: (T, B, d)
        k: (T, B, n)
        alpha: (T, B, n) ∈ [0,1]
        returns S: (T, B, d, n)
        """
        T, B, d = v.shape
        n = k.shape[-1]

        outer = torch.einsum("tbd,tbn->tbdn", v, k)

        # stable cumulative decay
        log_alpha = torch.log(alpha.clamp_min(1e-8))
        log_cumprod = torch.cumsum(log_alpha, dim=0)
        cumprod = torch.exp(log_cumprod)  # (T,B,n)

        decay = cumprod / (cumprod[-1:] + 1e-8)
        weighted = outer * decay.unsqueeze(2)
        S = torch.cumsum(weighted, dim=0)

        if self.truncation_steps is not None:
            S = S[-self.truncation_steps:]

        return S


class Gen2SingleInput(nn.Module):
    r"""
    Wrapper around StateOuterProductCumsum that accepts a *single* input x_t
    and internally projects it into key (k), value (v), and decay (alpha)
    signals using learnable linear layers.

    x_t -> Linear projections -> (v_t, k_t, α_t)
    -> StateOuterProductCumsum -> decayed associative memory S_t
    """

    def __init__(
        self,
        in_dim,
        d_value,
        d_key,
        truncation_steps=None,
        learnable_decay=True,
    ):
        super().__init__()
        self.d_value = d_value
        self.d_key = d_key

        # projections from input
        self.to_v = nn.Linear(in_dim, d_value)
        self.to_k = nn.Linear(in_dim, d_key)

        # optional learnable per-column decay
        if learnable_decay:
            self.to_alpha = nn.Linear(in_dim, d_key)
        else:
            self.register_buffer("to_alpha", None)

        self.state_update = StateOuterProductCumsum(truncation_steps)

    def forward(self, x):
        """
        x: (T, B, in_dim)
        returns: S (T, B, d_value, d_key)
        """
        v = self.to_v(x)
        k = self.to_k(x)

        if self.to_alpha is not None:
            # α ∈ (0,1)
            alpha = torch.sigmoid(self.to_alpha(x))
        else:
            # constant decay (like fixed β)
            alpha = torch.full_like(k, 0.9)

        S = self.state_update(v, k, alpha)
        return S

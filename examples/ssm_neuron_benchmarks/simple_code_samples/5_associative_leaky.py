import torch
import torch.nn as nn


class AssociativeLeaky(nn.Module):
    """
    Matrix-state spiking SSM (Gen-2):
        S_t = S_{t-1} * alpha_t^T + v_t k_t^T

    Implemented in closed form using prefix products (cumprod) and
    prefix sums (cumsum), then flattened and thresholded to spikes.
    """

    def __init__(self, in_dim, d_value, d_key, v_th=1.0):
        super().__init__()
        self.d_value = d_value  # rows of S_t
        self.d_key = d_key  # cols of S_t
        self.v_th = v_th
        self.eps = 1e-8

        # Projections from input x_t to (v_t, k_t, alpha_t)
        self.to_v = nn.Linear(in_dim, d_value)  # (T,B,in_dim) -> (T,B,d)
        self.to_k = nn.Linear(in_dim, d_key)  # (T,B,in_dim) -> (T,B,n)
        self.to_alpha = nn.Linear(in_dim, d_key)  # (T,B,in_dim) -> (T,B,n)

    def forward(self, x):
        """
        x: (T, B, in_dim)

        Returns:
            spk: (T, B, d_value * d_key)  binary spikes
            mem: (T, B, d_value * d_key)  flattened matrix state S_t
        """
        T, B, _ = x.shape
        d, n = self.d_value, self.d_key

        # Input projections
        v = self.to_v(x)  # (T,B,d)
        k = self.to_k(x)  # (T,B,n)
        alpha = torch.sigmoid(self.to_alpha(x))  # (T,B,n), in (0,1)

        # Rank-1 writes: B_t = v_t k_t^T
        writes = torch.einsum("tbd,tbn->tbdn", v, k)  # (T,B,d,n)

        # Prefix products P_t = prod_{u <= t} alpha_u  (time axis)
        P = torch.cumprod(alpha, dim=0)  # (T,B,n)

        # Scale each write by its inverse prefix: B_r / P_r
        invP = 1.0 / (P + self.eps)  # (T,B,n)
        scaled = writes * invP.unsqueeze(2)  # (T,B,d,n)

        # Prefix sum over scaled writes: sum_{r <= t} B_r / P_r
        S_scaled = torch.cumsum(scaled, dim=0)  # (T,B,d,n)

        # Reconstruct S_t = P_t * sum_{r <= t} B_r / P_r
        S = S_scaled * P.unsqueeze(2)  # (T,B,d,n)

        # Flatten matrix state to neurons and threshold
        mem = S.reshape(T, B, d * n)  # (T,B,d*n)
        spk = (mem > self.v_th).to(mem.dtype)
        return spk, mem

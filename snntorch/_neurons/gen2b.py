import torch
from torch import nn


class Gen2SingleInputReadout(nn.Module):
    def __init__(self, in_dim, d_value, d_key, time_chunk_size=None):
        super().__init__()
        self.to_v = nn.Linear(in_dim, d_value)  # (T,B,in)->(T,B,d)
        self.to_k = nn.Linear(in_dim, d_key)  # (T,B,in)->(T,B,n)
        self.to_q = nn.Linear(in_dim, d_key)  # (T,B,in)->(T,B,n)
        self.to_alpha = nn.Linear(in_dim, d_key)  # (T,B,in)->(T,B,n)
        self.time_chunk_size = time_chunk_size
        self.eps = 1e-8

    def forward(self, x):
        # x: (T, B, in_dim)
        T, B, _ = x.shape

        v = self.to_v(x)  # (T, B, d)
        k = self.to_k(x)  # (T, B, n)
        q = self.to_q(x)  # (T, B, n)
        alpha = torch.sigmoid(self.to_alpha(x))  # (T, B, n)

        C = self.time_chunk_size or T
        S_prev = None  # (B, d, n)
        d = v.shape[2]
        y = x.new_zeros((T, B, d))  # preallocate output

        for t0 in range(0, T, C):
            t1 = min(t0 + C, T)

            v_ch = v[t0:t1]  # (Tc, B, d)
            k_ch = k[t0:t1]  # (Tc, B, n)
            q_ch = q[t0:t1]  # (Tc, B, n)
            a_ch = alpha[t0:t1]  # (Tc, B, n)
            Tc = v_ch.shape[0]
            n = k_ch.shape[2]

            # Writes per step: v_t k_t^T -> (Tc,B,d,n)
            outer = torch.einsum("tbd,tbn->tbdn", v_ch, k_ch)

            # Prefix products p_t = ∏_{u<=t} a_u   (faster than log/exp)
            cumprod = torch.cumprod(
                a_ch.clamp_min(self.eps), dim=0
            )  # (Tc,B,n)

            # Divide-by-prefix, cumsum, then multiply-by-current-prefix
            inv_prefix = (1.0 / (cumprod + self.eps)).unsqueeze(
                2
            )  # (Tc,B,1,n)
            scaled = outer * inv_prefix  # (Tc,B,d,n) = B_r/p_r
            scaled = torch.cumsum(scaled, dim=0)  # Σ_{r<=t} B_r/p_r
            S_local = scaled * cumprod.unsqueeze(2)  # p_t * Σ_{r<=t} ...

            # Add carry from previous chunk: (∏_{u=t0..t} α_u) * S_prev
            if S_prev is not None:
                S_local = S_local + S_prev.unsqueeze(0) * cumprod.unsqueeze(2)

            # Readout with bmm instead of einsum
            Sb = S_local.reshape(Tc * B, d, n)  # (TcB,d,n)
            qb = q_ch.reshape(Tc * B, n, 1)  # (TcB,n,1)
            y_ch = torch.bmm(Sb, qb).reshape(Tc, B, d)  # (Tc,B,d)

            # Write into preallocated output
            y[t0:t1] = y_ch

            # Update carry for next chunk
            S_prev = S_local[-1]  # (B,d,n)

            # (Optional) free big temps ASAP
            del outer, scaled, S_local, Sb, qb, y_ch

        return y

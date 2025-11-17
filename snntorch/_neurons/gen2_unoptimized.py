import torch
from torch import nn
from torch.autograd import Function

# ===============================================================
# Custom autograd Function: fused readout + recomputation backward
# ===============================================================

# class StateOuterProductCumsumReadoutFn(Function):
#     @staticmethod
#     def forward(ctx, v, k, alpha, q, time_chunk_size=None):
#         pass
#
#     @staticmethod
#     def backward(ctx, grad_y):
#         pass
#
# """
# y = StateOuterProductCumsumReadoutFn.apply(
#     v, k, alpha, q, self.time_chunk_size
# )
# """


# ===============================================================
# High-level single-input wrapper (unfused; chunked forward)
# ===============================================================


class Gen2SingleInputReadout(nn.Module):
    def __init__(self, in_dim, d_value, d_key, time_chunk_size=None):
        super().__init__()
        self.to_v = nn.Linear(in_dim, d_value)  # (T, B, in_dim) -> (T, B, d)
        self.to_k = nn.Linear(in_dim, d_key)  # (T, B, in_dim) -> (T, B, n)
        self.to_q = nn.Linear(in_dim, d_key)  # (T, B, in_dim) -> (T, B, n)
        self.to_alpha = nn.Linear(in_dim, d_key)  # (T, B, in_dim) -> (T, B, n)
        self.time_chunk_size = time_chunk_size
        self.eps = 1e-8

    def forward(self, x):
        # x: (T, B, in_dim)
        T, B, _ = x.shape

        v = self.to_v(x)  # (T, B, d)
        k = self.to_k(x)  # (T, B, n)
        q = self.to_q(x)  # (T, B, n)
        alpha = torch.sigmoid(self.to_alpha(x))  # (T, B, n)

        chunk = self.time_chunk_size or T
        y_chunks = []
        S_prev = None  # carry: (B, d, n)

        for t0 in range(0, T, chunk):
            t1 = min(t0 + chunk, T)

            v_ch = v[t0:t1]  # (Tc, B, d)
            k_ch = k[t0:t1]  # (Tc, B, n)
            q_ch = q[t0:t1]  # (Tc, B, n)
            a_ch = alpha[t0:t1]  # (Tc, B, n)

            Tc = v_ch.shape[0]
            d = v_ch.shape[2]
            n = k_ch.shape[2]

            # Outer-product writes per step: v_t k_t^T -> (Tc, B, d, n)
            outer = torch.einsum("tbd,tbn->tbdn", v_ch, k_ch)  # (Tc, B, d, n)

            # Decay weights inside the chunk via cumprod in log-space
            log_a = (a_ch.clamp_min(self.eps)).log()  # (Tc, B, n)
            log_cumprod = torch.cumsum(log_a, dim=0)  # (Tc, B, n)
            cumprod = log_cumprod.exp()  # (Tc, B, n) = p_t = ∏_{u<=t} a_u

            assert cumprod.shape == (
                Tc,
                B,
                n,
            ), f"cumprod shape {tuple(cumprod.shape)} != expected (Tc, B, n)=({Tc}, {B}, {n})"

            # === Divide (per write r) by its own prefix p_r ===
            inv_prefix = 1.0 / (cumprod + 1e-8)  # (Tc, B, n)
            inv_prefix = inv_prefix.unsqueeze(2).expand(-1, -1, d, -1)
            scaled_writes = outer * inv_prefix  # (Tc, B, d, n)

            # State accumulation
            S_scaled = torch.cumsum(scaled_writes, dim=0)  # (Tc, B, d, n)
            S_local = S_scaled * cumprod.unsqueeze(2).expand(-1, -1, d, -1)

            assert S_local.shape == (
                Tc,
                B,
                d,
                n,
            ), f"S_local shape {tuple(S_local.shape)} != expected (Tc, B, d, n)=({Tc}, {B}, {d}, {n})"

            # --- add carry from previous chunk BEFORE readout ---
            if S_prev is not None:
                assert S_prev.shape == (B, d, n)
                S_prev_time = S_prev.unsqueeze(0).expand(Tc, -1, -1, -1)
                decay_time = cumprod.unsqueeze(2).expand(-1, -1, d, -1)
                carry = S_prev_time * decay_time
                S_local = S_local + carry

            # y_t = S_t q_t
            y_ch = torch.einsum("tbdn,tbn->tbd", S_local, q_ch)  # (Tc, B, d)
            y_chunks.append(y_ch)

            S_prev = S_local[-1]  # (B, d, n)

        y = torch.cat(y_chunks, dim=0)
        return y

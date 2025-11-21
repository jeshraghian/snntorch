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

        v = self.to_v(x)  # (T,B,d)
        k = self.to_k(x)  # (T,B,n)
        q = self.to_q(x)  # (T,B,n)
        alpha = torch.sigmoid(self.to_alpha(x))  # (T,B,n)

        C = self.time_chunk_size or T
        d = v.shape[2]
        n = k.shape[2]
        y = x.new_zeros((T, B, d))  # preallocate
        S_prev = None  # (B,d,n)

        # constant lower-triangular mask (on device and dtype)
        tril_cache = None

        for t0 in range(0, T, C):
            t1 = min(t0 + C, T)
            Tc = t1 - t0

            v_ch = v[t0:t1].contiguous()  # (Tc,B,d)
            k_ch = k[t0:t1].contiguous()  # (Tc,B,n)
            q_ch = q[t0:t1].contiguous()  # (Tc,B,n)
            a_ch = alpha[t0:t1].contiguous()  # (Tc,B,n)

            # prefix products p_t (per n), faster than log/exp for moderate Tc
            p = torch.cumprod(a_ch.clamp_min(self.eps), dim=0)  # (Tc,B,n)

            # Pre-scale for weights: q̃_t = q_t ⊙ p_t, k̃_r = k_r ⊘ p_r
            q_tilde = (q_ch * p).permute(1, 0, 2).contiguous()  # (B,Tc,n)
            k_tilde = (
                (k_ch / (p + self.eps)).permute(1, 0, 2).contiguous()
            )  # (B,Tc,n)

            # Weight matrix per batch: W = q̃ @ k̃^T  -> (B, Tc, Tc)
            W = torch.bmm(q_tilde, k_tilde.transpose(1, 2))

            # Causalize: keep r <= t
            if tril_cache is None or tril_cache.shape[-1] != Tc:
                tril_cache = torch.tril(
                    torch.ones((Tc, Tc), device=W.device, dtype=W.dtype)
                )
            W = W * tril_cache  # (B,Tc,Tc)

            # Readout: y_no_carry = W @ V, where V stacks v_r over time
            V = v_ch.permute(1, 0, 2).contiguous()  # (B,Tc,d)
            y_no_carry = torch.bmm(W, V)  # (B,Tc,d)

            # Carry contribution: y_carry_t = S_prev @ (q_t ⊙ p_t)^T
            if S_prev is not None:
                # q̃^T : (B,n,Tc); S_prev:(B,d,n) -> (B,d,Tc)
                y_carry = torch.bmm(
                    S_prev, q_tilde.transpose(1, 2)
                )  # (B,d,Tc)
                y_carry = y_carry.transpose(1, 2).contiguous()  # (B,Tc,d)
                y_ch = y_no_carry + y_carry
            else:
                y_ch = y_no_carry

            # write back to output
            y[t0:t1] = y_ch.permute(
                1, 0, 2
            ).contiguous()  # (Tc,B,d) -> (T,B,d)

            # ---- Update carry S_prev for next chunk, without (Tc,B,d,n) tensors ----
            # S_end = p_end * S_prev + sum_r (p_end/p_r) * (v_r k_r^T)
            p_end = p[-1]  # (B,n)
            # weights_r(n) = (p_end / p_r) * k_r(n)  -> (Tc,B,n)
            weights = (k_ch / (p + self.eps)) * p_end.unsqueeze(0)
            # sum over r: v_r (B,d) times scalar weights_r(n) -> (B,d,n)
            S_writes_end = torch.einsum("tbd,tbn->bdn", v_ch, weights)
            if S_prev is None:
                S_prev = S_writes_end
            else:
                S_prev = S_prev * p_end.unsqueeze(1) + S_writes_end  # (B,d,n)

            # free large temporaries early
            del q_tilde, k_tilde, W, V, y_no_carry, weights, S_writes_end

        return y

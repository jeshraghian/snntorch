import torch
from torch import nn


class Gen2SingleInputReadout(nn.Module):
    def __init__(
        self, in_dim, d_value, d_key, time_chunk_size=None, r_block=1024 * 10
    ):
        super().__init__()
        self.to_v = nn.Linear(in_dim, d_value)
        self.to_k = nn.Linear(in_dim, d_key)
        self.to_q = nn.Linear(in_dim, d_key)
        self.to_alpha = nn.Linear(in_dim, d_key)
        self.time_chunk_size = time_chunk_size
        self.r_block = r_block
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
        y = x.new_zeros((T, B, d))
        S_prev = None

        tril_cache = None  # optional cache if you later go back to T×T path

        for t0 in range(0, T, C):
            t1 = min(t0 + C, T)
            Tc = t1 - t0

            v_ch = v[t0:t1].contiguous()  # (Tc,B,d)
            k_ch = k[t0:t1].contiguous()  # (Tc,B,n)
            q_ch = q[t0:t1].contiguous()  # (Tc,B,n)
            a_ch = alpha[t0:t1].contiguous()  # (Tc,B,n)

            # Prefix products p_t in the chunk
            p = torch.cumprod(a_ch.clamp_min(self.eps), dim=0)  # (Tc,B,n)

            # q̃ and k̃ in (B,Tc,n) order for batched GEMMs
            q_tilde = (q_ch * p).permute(1, 0, 2).contiguous()  # (B,Tc,n)
            k_tilde_all = (
                (k_ch / (p + self.eps)).permute(1, 0, 2).contiguous()
            )  # (B,Tc,n)
            V_all = v_ch.permute(1, 0, 2).contiguous()  # (B,Tc,d)

            # Per-chunk output buffer in (B,Tc,d)
            y_ch = v_ch.new_zeros((B, Tc, d))

            # Block over r to avoid forming (B,Tc,Tc)
            R = self.r_block
            for r0 in range(0, Tc, R):
                r1 = min(r0 + R, Tc)
                Rlen = r1 - r0

                # (B,R,n), (B,R,d)
                k_blk = k_tilde_all[:, r0:r1, :]  # (B,R,n)
                v_blk = V_all[:, r0:r1, :]  # (B,R,d)

                # A = q̃ @ k̃^T  -> (B,Tc,R)
                A = torch.bmm(q_tilde, k_blk.transpose(1, 2))  # (B,Tc,R)

                # Causal mask for the current r-block: keep r<=t
                t_idx = torch.arange(Tc, device=A.device).unsqueeze(
                    1
                )  # (Tc,1)
                r_idx = torch.arange(r0, r1, device=A.device).unsqueeze(
                    0
                )  # (1,R)
                causal = (t_idx >= r_idx).to(A.dtype)  # (Tc,R)
                A = A * causal.unsqueeze(0)  # (B,Tc,R)

                # Accumulate contribution: (B,Tc,R) @ (B,R,d) -> (B,Tc,d)
                y_ch += torch.bmm(A, v_blk)

                del k_blk, v_blk, A, causal

            # Carry: y_carry_t = S_prev @ (q̃_t)^T, shape (B,Tc,d)
            if S_prev is not None:
                y_carry = torch.bmm(
                    S_prev, q_tilde.transpose(1, 2)
                )  # (B,d,Tc)
                y_ch += y_carry.transpose(1, 2).contiguous()  # (B,Tc,d)
                del y_carry

            # Write back to (T,B,d)
            y[t0:t1] = y_ch.permute(1, 0, 2).contiguous()  # (Tc,B,d)

            # Update carry for next chunk (no T×T / (Tc,B,d,n) tensors)
            p_end = p[-1]  # (B,n)
            weights = (k_ch / (p + self.eps)) * p_end.unsqueeze(0)  # (Tc,B,n)
            S_writes_end = torch.einsum(
                "tbd,tbn->bdn", v_ch, weights
            )  # (B,d,n)
            S_prev = (
                S_writes_end
                if S_prev is None
                else S_prev * p_end.unsqueeze(1) + S_writes_end
            )
            del weights, S_writes_end, q_tilde, k_tilde_all, V_all, y_ch, p

        return y

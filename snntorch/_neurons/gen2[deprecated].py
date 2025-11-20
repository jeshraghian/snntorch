import torch
from torch import nn
from torch.autograd import Function


# ===============================================================
# Custom autograd Function: fused readout + recomputation backward
# ===============================================================
class StateOuterProductCumsumReadoutFn(Function):
    @staticmethod
    def forward(ctx, v, k, alpha, q, time_chunk_size=None):
        """
        v: (T, B, d)
        k: (T, B, n)
        alpha: (T, B, n)
        q: (T, B, n)
        returns: y (T, B, d)
        """
        T, B, d = v.shape
        n = k.shape[-1]

        ctx.save_for_backward(v, k, alpha, q)
        ctx.time_chunk_size = time_chunk_size

        y_chunks = []
        chunk = time_chunk_size or T
        S_prev = None

        for t0 in range(0, T, chunk):
            t1 = min(t0 + chunk, T)
            v_chunk, k_chunk, a_chunk, q_chunk = (
                v[t0:t1],
                k[t0:t1],
                alpha[t0:t1],
                q[t0:t1],
            )

            # --- recompute local decay ---
            outer = torch.einsum("tbd,tbn->tbdn", v_chunk, k_chunk)
            log_a = torch.log(a_chunk.clamp_min(1e-8))
            log_cumprod = torch.cumsum(log_a, dim=0)
            cumprod = torch.exp(log_cumprod)
            decay = cumprod / (cumprod[-1:] + 1e-8)
            weighted = outer * decay.unsqueeze(2)
            S_local = torch.cumsum(weighted, dim=0)

            # --- carry from previous chunk ---
            if S_prev is not None:
                prefix_decay = torch.exp(log_cumprod[0])
                S_local += S_prev.unsqueeze(0) * prefix_decay.unsqueeze(1)

            # --- fused readout ---
            y_chunk = torch.einsum("tbdn,tbn->tbd", S_local, q_chunk)
            y_chunks.append(y_chunk)

            S_prev = S_local[-1]

        y = torch.cat(y_chunks, dim=0)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        v, k, alpha, q = ctx.saved_tensors
        time_chunk_size = ctx.time_chunk_size
        T, B, d = v.shape
        n = k.shape[-1]

        grad_v = torch.zeros_like(v)
        grad_k = torch.zeros_like(k)
        grad_alpha = torch.zeros_like(alpha)
        grad_q = torch.zeros_like(q)

        chunk = time_chunk_size or T
        S_prev = None

        for t0 in range(0, T, chunk):
            t1 = min(t0 + chunk, T)
            v_chunk, k_chunk, a_chunk, q_chunk = (
                v[t0:t1],
                k[t0:t1],
                alpha[t0:t1],
                q[t0:t1],
            )
            gy_chunk = grad_y[t0:t1]

            # Recompute forward intermediates for this chunk
            outer = torch.einsum("tbd,tbn->tbdn", v_chunk, k_chunk)
            log_a = torch.log(a_chunk.clamp_min(1e-8))
            log_cumprod = torch.cumsum(log_a, dim=0)
            cumprod = torch.exp(log_cumprod)
            decay = cumprod / (cumprod[-1:] + 1e-8)
            weighted = outer * decay.unsqueeze(2)
            S_local = torch.cumsum(weighted, dim=0)

            if S_prev is not None:
                prefix_decay = torch.exp(log_cumprod[0])
                S_local += S_prev.unsqueeze(0) * prefix_decay.unsqueeze(1)

            # === Backward for fused readout ===
            # y_t = einsum("tbdn,tbn->tbd")
            gS_local = torch.einsum("tbd,tbn->tbdn", gy_chunk, q_chunk)
            gq_chunk = torch.einsum("tbdn,tbd->tbn", S_local, gy_chunk)
            grad_q[t0:t1] += gq_chunk

            # === Propagate gradients through S_local ===
            g_weighted = torch.flip(
                torch.cumsum(torch.flip(gS_local, [0]), dim=0), [0]
            )
            g_outer = g_weighted * decay.unsqueeze(2)
            g_decay = (g_weighted * outer).sum(dim=2)

            grad_v[t0:t1] += torch.einsum("tbdn,tbn->tbd", g_outer, k_chunk)
            grad_k[t0:t1] += torch.einsum("tbdn,tbd->tbn", g_outer, v_chunk)

            g_log_a = g_decay * cumprod / (cumprod[-1:] + 1e-8)
            grad_alpha[t0:t1] += g_log_a * (1 / a_chunk.clamp_min(1e-8))

            S_prev = S_local[-1]

        return grad_v, grad_k, grad_alpha, grad_q, None


# ===============================================================
# nn.Module wrapper with fused readout
# ===============================================================
class StateOuterProductCumsumReadout(nn.Module):
    def __init__(self, truncation_steps=None, time_chunk_size=None):
        super().__init__()
        self.truncation_steps = truncation_steps
        self.time_chunk_size = time_chunk_size

    def forward(self, v, k, alpha, q):
        y = StateOuterProductCumsumReadoutFn.apply(
            v, k, alpha, q, self.time_chunk_size
        )
        if self.truncation_steps is not None:
            y = y[-self.truncation_steps :]
        return y


# ===============================================================
# High-level single-input wrapper (fused readout)
# ===============================================================
class Gen2SingleInputReadout(nn.Module):
    def __init__(
        self,
        in_dim,
        d_value,
        d_key,
        truncation_steps=None,
        learnable_decay=True,
        time_chunk_size=None,
    ):
        super().__init__()
        self.to_v = nn.Linear(in_dim, d_value)
        self.to_k = nn.Linear(in_dim, d_key)
        self.to_q = nn.Linear(in_dim, d_key)
        self.to_alpha = nn.Linear(in_dim, d_key) if learnable_decay else None

        self.state_update = StateOuterProductCumsumReadout(
            truncation_steps=truncation_steps,
            time_chunk_size=time_chunk_size,
        )

    def forward(self, x):
        v = self.to_v(x)
        k = self.to_k(x)
        q = self.to_q(x)
        alpha = (
            torch.sigmoid(self.to_alpha(x))
            if self.to_alpha is not None
            else torch.full_like(k, 0.9)
        )
        return self.state_update(v, k, alpha, q)

import torch
import torch.nn.functional as F


def iterative_rollout(alpha, u):
    # u: (T, D), alpha: scalar or (D,)
    T, D = u.shape
    x = torch.zeros(D, device=u.device, dtype=u.dtype)
    xs = []
    for t in range(T):
        x = alpha * x + u[t]
        xs.append(x)
    return torch.stack(xs, dim=0)


def conv_rollout(alpha, u):
    # u: (B, T, C), alpha: scalar or (C,)
    B, T, C = u.shape
    k = torch.arange(T, device=u.device)
    h = (alpha**k).view(1, 1, T)  # (1, 1, T)

    x = u.transpose(1, 2)  # (B, C, T)
    v = F.conv1d(x, h.expand(C, 1, T), padding=T - 1, groups=C)  # (B, C, 2T-1)
    v = v.transpose(1, 2)[:, :T, :]  # (B, T, C)
    return v


def prefix_scan(alpha, u):
    # u: (T, D), alpha: (T, D) or (T, 1)
    p = torch.cumprod(alpha, dim=0)  # (T, D)
    contrib = u / p  # (T, D)
    s = torch.cumsum(contrib, dim=0)  # (T, D)
    x = p * s
    return x

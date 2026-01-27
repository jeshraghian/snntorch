import torch
import torch.nn.functional as F


def iterative_rollout(beta, u):
    # u: (B, T, D), beta: (D,)
    B, T, D = u.shape
    x = torch.zeros(B, D, device=u.device, dtype=u.dtype)
    xs = []

    for t in range(T):
        x = x * beta + u[:, t]
        xs.append(x)

    return torch.stack(xs, dim=1)                    # (B, T, D)


def conv_rollout(beta, u):
    # u: (B, T, D), beta: (D,)
    B, T, D = u.shape

    k = torch.arange(start=T - 1, end=-1, step=-1, device=u.device)
    beta_exp = beta.unsqueeze(1)                     # (D, 1)
    k_exp = k.unsqueeze(0)                           # (1, T)
    h = (beta_exp ** k_exp).unsqueeze(1)             # (D, 1, T)

    x = u.transpose(1, 2)                            # (B, D, T)
    y = F.conv1d(x, h, padding=T - 1, groups=D)      # (B, D, 2T-1)

    return y[:, :, :T].transpose(1, 2)               # (B, T, D)


def prefix_scan(beta, u):
    # u: (B, T, D), beta: (D,)
    T = u.size(1)
    k = torch.arange(T, device=u.device)
    k_exp = k.unsqueeze(1)                           # (T, 1)
    p = beta ** k_exp                                # (T, D)

    return p * torch.cumsum(u / p, dim=1)

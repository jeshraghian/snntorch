import os
import sys
import importlib.util
import torch


# Ensure project root on sys.path when running directly
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_iterative_rollout_scalar_alpha_matches_manual():
    this_dir = os.path.dirname(__file__)
    mod_path = os.path.abspath(
        os.path.join(this_dir, "..", "0_linear_recurrence.py")
    )
    mod = _load_module_from_path("linear_recurrence_module", mod_path)

    T, D = 5, 3
    alpha = torch.tensor(0.5)
    u = torch.arange(T * D, dtype=torch.float32).view(T, D) / 10.0

    out = mod.iterative_rollout(alpha, u)

    # Manual recurrence: x[t] = alpha * x[t-1] + u[t], x[-1] = 0
    x = torch.zeros(D, dtype=u.dtype)
    expected = []
    for t in range(T):
        x = alpha * x + u[t]
        expected.append(x.clone())
    expected = torch.stack(expected, dim=0)

    assert torch.allclose(out, expected, atol=1e-6)


def test_conv_rollout_scalar_alpha_matches_iterative():
    this_dir = os.path.dirname(__file__)
    mod_path = os.path.abspath(
        os.path.join(this_dir, "..", "0_linear_recurrence.py")
    )
    mod = _load_module_from_path("linear_recurrence_module", mod_path)

    B, T, C = 2, 6, 4
    alpha = torch.tensor(0.7)
    u = torch.linspace(0.0, 1.0, steps=B * T * C, dtype=torch.float32).view(
        B, T, C
    )

    conv_out = mod.conv_rollout(alpha, u)

    # Match literal conv1d correlation with padding=T-1 followed by [:T] slicing
    # y[t] = sum_{m=0..T-1} h[m] * x[t + m - (T-1)], with x = u^T and h[m] = alpha**m
    x = u.transpose(1, 2)  # (B, C, T)
    expected = torch.zeros_like(conv_out)
    for b in range(B):
        for c in range(C):
            for t in range(T):
                s = 0.0
                for m in range(T):
                    idx = t + m - (T - 1)
                    if 0 <= idx < T:
                        s = s + alpha.pow(m) * x[b, c, idx]
                expected[b, t, c] = s

    assert torch.allclose(conv_out, expected, atol=1e-6)


def test_alpha_zero_identity_for_iterative():
    this_dir = os.path.dirname(__file__)
    mod_path = os.path.abspath(
        os.path.join(this_dir, "..", "0_linear_recurrence.py")
    )
    mod = _load_module_from_path("linear_recurrence_module", mod_path)

    # alpha = 0 => x[t] = u[t]
    T, D = 8, 5
    alpha0 = torch.tensor(0.0)
    u_td = torch.randn(T, D, dtype=torch.float32)
    out_iter = mod.iterative_rollout(alpha0, u_td)
    assert torch.allclose(out_iter, u_td, atol=0.0)


def test_alpha_one_cumsum_for_all():
    this_dir = os.path.dirname(__file__)
    mod_path = os.path.abspath(
        os.path.join(this_dir, "..", "0_linear_recurrence.py")
    )
    mod = _load_module_from_path("linear_recurrence_module", mod_path)

    # alpha = 1 => x[t] = sum_{k=0..t} u[k]
    T, D = 7, 3
    alpha1 = torch.tensor(1.0)
    u_td = torch.randn(T, D, dtype=torch.float32)
    out_iter = mod.iterative_rollout(alpha1, u_td)
    expected_td = torch.cumsum(u_td, dim=0)
    assert torch.allclose(out_iter, expected_td, atol=1e-6)

    B, C = 2, 4
    u_btc = torch.randn(B, T, C, dtype=torch.float32)
    out_conv = mod.conv_rollout(alpha1, u_btc)
    expected_btc = torch.cumsum(u_btc, dim=1)
    assert torch.allclose(out_conv, expected_btc, atol=1e-6)

    # prefix_scan supports time-varying alpha; using ones recovers cumsum
    alpha_td = torch.ones(T, D, dtype=torch.float32)
    out_prefix = mod.prefix_scan(alpha_td, u_td)
    assert torch.allclose(out_prefix, expected_td, atol=1e-6)


def test_prefix_scan_time_varying_alpha_matches_iterative():
    this_dir = os.path.dirname(__file__)
    mod_path = os.path.abspath(
        os.path.join(this_dir, "..", "0_linear_recurrence.py")
    )
    mod = _load_module_from_path("linear_recurrence_module", mod_path)

    T, D = 7, 2
    # Ensure nonzero alphas to avoid division issues; keep <1 for stability
    alpha = 0.2 + 0.6 * torch.rand(T, D, dtype=torch.float32)
    u = torch.randn(T, D, dtype=torch.float32) * 0.1

    out = mod.prefix_scan(alpha, u)

    # Expected via iterative recurrence with time-varying alpha
    x = torch.zeros(D, dtype=u.dtype)
    expected = []
    for t in range(T):
        x = alpha[t] * x + u[t]
        expected.append(x.clone())
    expected = torch.stack(expected, dim=0)

    assert torch.allclose(out, expected, atol=1e-5)


if __name__ == "__main__":
    test_iterative_rollout_scalar_alpha_matches_manual()
    print("test_iterative_rollout_scalar_alpha_matches_manual passed")
    test_conv_rollout_scalar_alpha_matches_iterative()
    print("test_conv_rollout_scalar_alpha_matches_iterative passed")
    test_prefix_scan_time_varying_alpha_matches_iterative()
    print("test_prefix_scan_time_varying_alpha_matches_iterative passed")

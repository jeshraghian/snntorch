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


def test_iterative_rollout_matches_manual():
    this_dir = os.path.dirname(__file__)
    mod_path = os.path.abspath(
        os.path.join(this_dir, "..", "0_linear_recurrence.py")
    )
    mod = _load_module_from_path("linear_recurrence_module", mod_path)

    B, T, D = 2, 5, 3
    beta = torch.linspace(0.2, 0.8, D)
    u = torch.arange(B * T * D, dtype=torch.float32).view(B, T, D) / 10.0

    out = mod.iterative_rollout(beta, u)

    # Manual recurrence: x[t] = beta * x[t-1] + u[t], x[-1] = 0
    x = torch.zeros(B, D, dtype=u.dtype)
    expected = []
    for t in range(T):
        x = x * beta + u[:, t]
        expected.append(x.clone())
    expected = torch.stack(expected, dim=1)

    assert torch.allclose(out, expected, atol=1e-6)


def test_conv_rollout_matches_iterative():
    this_dir = os.path.dirname(__file__)
    mod_path = os.path.abspath(
        os.path.join(this_dir, "..", "0_linear_recurrence.py")
    )
    mod = _load_module_from_path("linear_recurrence_module", mod_path)

    B, T, D = 2, 6, 4
    beta = torch.linspace(0.3, 0.9, D)
    u = torch.linspace(0.0, 1.0, steps=B * T * D, dtype=torch.float32).view(
        B, T, D
    )

    conv_out = mod.conv_rollout(beta, u)
    iter_out = mod.iterative_rollout(beta, u)

    assert torch.allclose(conv_out, iter_out, atol=1e-6)


def test_beta_zero_identity_for_iterative():
    this_dir = os.path.dirname(__file__)
    mod_path = os.path.abspath(
        os.path.join(this_dir, "..", "0_linear_recurrence.py")
    )
    mod = _load_module_from_path("linear_recurrence_module", mod_path)

    # beta = 0 => x[t] = u[t]
    B, T, D = 3, 8, 5
    beta0 = torch.zeros(D)
    u_btd = torch.randn(B, T, D, dtype=torch.float32)
    out_iter = mod.iterative_rollout(beta0, u_btd)
    assert torch.allclose(out_iter, u_btd, atol=0.0)


def test_beta_one_cumsum_for_all():
    this_dir = os.path.dirname(__file__)
    mod_path = os.path.abspath(
        os.path.join(this_dir, "..", "0_linear_recurrence.py")
    )
    mod = _load_module_from_path("linear_recurrence_module", mod_path)

    # beta = 1 => x[t] = sum_{k=0..t} u[k]
    B, T, D = 2, 7, 3
    beta1 = torch.ones(D)
    u_btd = torch.randn(B, T, D, dtype=torch.float32)
    out_iter = mod.iterative_rollout(beta1, u_btd)
    expected_btd = torch.cumsum(u_btd, dim=1)
    assert torch.allclose(out_iter, expected_btd, atol=1e-6)

    out_conv = mod.conv_rollout(beta1, u_btd)
    assert torch.allclose(out_conv, expected_btd, atol=1e-6)

    out_prefix = mod.prefix_scan(beta1, u_btd)
    assert torch.allclose(out_prefix, expected_btd, atol=1e-6)


def test_prefix_scan_matches_iterative():
    this_dir = os.path.dirname(__file__)
    mod_path = os.path.abspath(
        os.path.join(this_dir, "..", "0_linear_recurrence.py")
    )
    mod = _load_module_from_path("linear_recurrence_module", mod_path)

    B, T, D = 3, 7, 2
    # Ensure nonzero betas to avoid division issues; keep <1 for stability
    beta = 0.2 + 0.6 * torch.rand(D, dtype=torch.float32)
    u = torch.randn(B, T, D, dtype=torch.float32) * 0.1

    out = mod.prefix_scan(beta, u)

    # Expected via iterative recurrence with constant beta
    x = torch.zeros(B, D, dtype=u.dtype)
    expected = []
    for t in range(T):
        x = x * beta + u[:, t]
        expected.append(x.clone())
    expected = torch.stack(expected, dim=1)

    assert torch.allclose(out, expected, atol=1e-5)


if __name__ == "__main__":
    test_iterative_rollout_matches_manual()
    print("test_iterative_rollout_matches_manual passed")
    test_conv_rollout_matches_iterative()
    print("test_conv_rollout_matches_iterative passed")
    test_prefix_scan_matches_iterative()
    print("test_prefix_scan_matches_iterative passed")

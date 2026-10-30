import torch
from snntorch import surrogate


def _grad(spike_fn, x):
    # x is the surrogate input, i.e. U - U_thr
    x = x.clone().requires_grad_(True)
    spike_fn(x).sum().backward()
    return x.grad


def test_triangular_gradient_shape():
    x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    grad = _grad(surrogate.triangular(), x)
    expected = torch.tensor([0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0])
    assert torch.allclose(grad, expected)


def test_triangular_gradient_is_non_negative_and_peaks_at_threshold():
    x = torch.linspace(-3, 3, 601)
    grad = _grad(surrogate.triangular(threshold=2.0), x)
    assert (grad >= 0).all()
    assert torch.argmax(grad) == torch.argmin(torch.abs(x))
    # support is |U - U_thr| < U_thr
    assert (grad[torch.abs(x) >= 2.0] == 0).all()


def test_spike_rate_escape_peaks_at_threshold():
    x = torch.linspace(-3, 3, 601)
    grad = _grad(surrogate.spike_rate_escape(beta=1, slope=25), x)
    assert torch.argmax(grad) == torch.argmin(torch.abs(x))
    assert torch.isclose(grad.max(), torch.tensor(25.0))
    assert torch.allclose(grad, torch.flip(grad, dims=[0]))

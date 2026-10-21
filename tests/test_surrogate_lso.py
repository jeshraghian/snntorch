import torch
import snntorch as snn
from snntorch import surrogate


def test_lso_backward():
    slope = 0.1
    x = torch.tensor([-2.0, -0.5, 0.5, 2.0], requires_grad=True)
    surrogate.LSO(slope)(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([slope, slope, 1.0, 1.0]))


def test_lso_backward_through_leaky():
    net = snn.Leaky(beta=0.9, spike_grad=surrogate.LSO())
    x = torch.randn(4, 3, requires_grad=True)
    spk, _ = net(x, net.init_leaky())
    spk.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

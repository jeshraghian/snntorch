import pytest
import torch

from snntorch._neurons.associative import AssociativeLeaky


# -----------------------
# Fixtures
# -----------------------
@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def assoc_explicit_q(device):
    # d = n = 4 -> 16 neurons
    return AssociativeLeaky(
        in_dim=16,
        d_value=4,
        d_key=4,
        num_spiking_neurons=16,
        use_q_projection=True,
    ).to(device)


@pytest.fixture(scope="module")
def assoc_explicit_noq(device):
    return AssociativeLeaky(
        in_dim=16,
        d_value=4,
        d_key=4,
        num_spiking_neurons=16,
        use_q_projection=False,
    ).to(device)


@pytest.fixture(scope="module")
def assoc_fromN_q(device):
    return AssociativeLeaky.from_num_spiking_neurons(
        in_dim=16, num_spiking_neurons=16, use_q_projection=True
    ).to(device)


@pytest.fixture(scope="module")
def assoc_fromN_noq(device):
    return AssociativeLeaky.from_num_spiking_neurons(
        in_dim=16, num_spiking_neurons=16, use_q_projection=False
    ).to(device)


# -----------------------
# Constructor validation
# -----------------------
def test_d_value_d_key_positive():
    with pytest.raises(ValueError):
        # d_value must be a positive integer (0 is invalid)
        _ = AssociativeLeaky(
            in_dim=8, d_value=0, d_key=4, num_spiking_neurons=16
        )
    with pytest.raises(ValueError):
        # d_key must be a positive integer (0 is invalid)
        _ = AssociativeLeaky(
            in_dim=8, d_value=4, d_key=0, num_spiking_neurons=16
        )
    with pytest.raises(ValueError):
        # d_value must be positive (-1 is invalid)
        _ = AssociativeLeaky(
            in_dim=8, d_value=-1, d_key=4, num_spiking_neurons=16
        )
    with pytest.raises(ValueError):
        # d_key must be positive (-1 is invalid)
        _ = AssociativeLeaky(
            in_dim=8, d_value=4, d_key=-1, num_spiking_neurons=16
        )


def test_fromN_requires_perfect_square():
    # num_spiking_neurons must be a perfect square (18 is not)
    with pytest.raises(ValueError):
        _ = AssociativeLeaky.from_num_spiking_neurons(
            in_dim=8, num_spiking_neurons=18
        )


def test_num_spiking_neurons_consistency():
    # num_spiking_neurons must equal d_value * d_key (6 != 5)
    with pytest.raises(ValueError):
        _ = AssociativeLeaky(
            in_dim=8, d_value=2, d_key=3, num_spiking_neurons=5
        )


# -----------------------
# Forward shapes and basic sanity
# -----------------------
def _make_input(T=5, B=2, in_dim=16, device="cpu"):
    x = torch.arange(T * B * in_dim, dtype=torch.float32, device=device)
    x = (x / (T * B * in_dim)).view(T, B, in_dim)  # values in [0,1)
    return x


def test_forward_shape_explicit_q(device, assoc_explicit_q):
    # d = n = 4 -> output (T,B, d*d = 16)
    T, B, in_dim = 5, 2, 16
    x = _make_input(T, B, in_dim, device)
    y = assoc_explicit_q(x)
    assert y.shape == (T, B, 16)


def test_forward_shape_explicit_noq(device, assoc_explicit_noq):
    # d = 4, n = 4 -> output (T,B, d*n = 16)
    T, B, in_dim = 5, 2, 16
    x = _make_input(T, B, in_dim, device)
    y = assoc_explicit_noq(x)
    assert y.shape == (T, B, 16)


def test_forward_finite(device, assoc_explicit_q):
    T, B, in_dim = 5, 2, 16
    x = _make_input(T, B, in_dim, device)
    y = assoc_explicit_q(x)
    assert torch.isfinite(y).all().item()


# -----------------------
# Q‑projection semantics and output flag
# -----------------------
def test_q_projection_true_vs_false_shapes(device):
    # Same d,n but different projection flag
    T, B, in_dim = 5, 2, 16
    x = _make_input(T, B, in_dim, device)
    d, n = 4, 4
    m_q = AssociativeLeaky(
        in_dim=in_dim,
        d_value=d,
        d_key=n,
        num_spiking_neurons=d * n,
        use_q_projection=True,
    ).to(device)
    m_noq = AssociativeLeaky(
        in_dim=in_dim,
        d_value=d,
        d_key=n,
        num_spiking_neurons=d * n,
        use_q_projection=False,
    ).to(device)
    y_q = m_q(x)
    y_noq = m_noq(x)
    assert y_q.shape == (T, B, d * d)
    assert y_noq.shape == (T, B, d * n)


def test_output_flag_branch(device):
    # With q_projection=True, toggling output=False should preserve shape (T,B,d*d)
    T, B, in_dim = 5, 2, 16
    x = _make_input(T, B, in_dim, device)
    d = 4
    model = AssociativeLeaky(
        in_dim=in_dim,
        d_value=d,
        d_key=d,
        num_spiking_neurons=d * d,
        use_q_projection=True,
    ).to(device)
    model.output = False
    y = model(x)
    assert y.shape == (T, B, d * d)


# -----------------------
# Backward / gradients
# -----------------------
def test_backward_populates_internal_params(device):
    T, B, in_dim = 5, 2, 16
    x = _make_input(T, B, in_dim, device)
    model_q = AssociativeLeaky(
        in_dim=in_dim,
        d_value=4,
        d_key=4,
        num_spiking_neurons=16,
        use_q_projection=True,
    ).to(device)
    model_q.train()
    loss = model_q(x).sum()
    loss.backward()
    assert model_q.to_v.weight.grad is not None
    assert model_q.to_k.weight.grad is not None
    assert model_q.to_alpha.weight.grad is not None
    assert model_q.to_q is not None and model_q.to_q.weight.grad is not None


# -----------------------
# Batch chunking equivalence (forward + grad)
# -----------------------
def test_chunking_matches_full(device):
    torch.manual_seed(0)
    T, B, in_dim = 8, 6, 16
    chunk_size = 2
    x = _make_input(T, B, in_dim, device)
    # external pre-projection
    pre = torch.nn.Linear(in_dim, in_dim, bias=False).to(device)
    model = AssociativeLeaky(
        in_dim=in_dim,
        d_value=4,
        d_key=4,
        num_spiking_neurons=16,
        use_q_projection=True,
    ).to(device)
    model.train()

    # Full forward/backward
    pre.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    y_full = model(pre(x))
    loss_full = y_full.sum()
    loss_full.backward()
    grad_full = pre.weight.grad.detach().clone()

    # Chunked forward/backward across batch
    pre.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    ys = []
    for b_start in range(0, B, chunk_size):
        b_end = min(b_start + chunk_size, B)
        x_chunk = x[:, b_start:b_end, :]
        y_chunk = model(pre(x_chunk))
        ys.append(y_chunk)
        loss_chunk = y_chunk.sum()
        loss_chunk.backward()
    y_chunked = torch.cat(ys, dim=1)
    grad_chunked = pre.weight.grad.detach().clone()

    assert y_full.shape == y_chunked.shape
    assert torch.allclose(y_full, y_chunked, atol=1e-6)
    assert torch.allclose(grad_full, grad_chunked, atol=1e-6)


def test_fixture_smoke(
    assoc_explicit_q, assoc_explicit_noq, assoc_fromN_q, assoc_fromN_noq
):
    assert isinstance(assoc_explicit_q, AssociativeLeaky)
    assert isinstance(assoc_explicit_noq, AssociativeLeaky)
    assert isinstance(assoc_fromN_q, AssociativeLeaky)
    assert isinstance(assoc_fromN_noq, AssociativeLeaky)

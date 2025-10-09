import pytest
import torch
import torch.nn as nn

from snntorch._neurons.linearleaky import LinearLeaky
from snntorch._neurons.stateleaky import StateLeaky

"""
Tests for the LinearLeaky neuron class.

Test Structure:
--------------
1. Channel Configuration Tests:
    - Single batch, single channel (in_features == out_features)
    - Single batch, multiple channels (in_features == out_features)
    - Multiple batches, single channel (in_features == out_features)
    - Multiple batches, multiple channels (in_features == out_features)

2. Learning Parameter Tests:
    - Multi-beta learning (tests learn_beta=True)

3. Chunking Tests:
    - Tests chunking with gradients enabled, using the module’s internal linear

4. Equivalence Tests:
    - Equivalence vs separate nn.Linear followed by StateLeaky

5. Kernel Truncation Tests:
    - Tests kernel truncation with truncation_steps=4 (impulse input)

6. API Compatibility / Fail-Fast Tests:
    - Verifies that LinearLeaky raises NotImplementedError for APIs that do not
      conceptually apply (inhibition, mem_reset) — inherited from StateLeaky

7. Warnings (Ergonomics):
    - Verifies that inert spike-only settings emit a warning when output=False


Coverage:
--------
- Input/output shape consistency
- Input value bounds (≤ 1)
- Finite outputs and non-constancy (no amplitude assumption by default)
- Parameter learnability for learn_beta
- Chunking (smoke test) with gradients enabled and internal linear
- Equivalence vs separate nn.Linear + StateLeaky
"""


@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# Modules under test
@pytest.fixture(scope="module")
def linearleaky_single_channel(device):
    return LinearLeaky(
        beta=0.9, in_features=1, out_features=1, output=False
    ).to(device)


@pytest.fixture(scope="module")
def linearleaky_multi_channel(device):
    return LinearLeaky(
        beta=0.9, in_features=4, out_features=4, output=False
    ).to(device)


@pytest.fixture(scope="module")
def linearleaky_multi_beta(device):
    return LinearLeaky(
        beta=torch.tensor([0.9] * 4),
        in_features=4,
        out_features=4,
        learn_beta=True,
        output=False,
    ).to(device)


# Input fixtures
@pytest.fixture(scope="module")
def input_tensor_single_batch_single_channel(device):
    timesteps = 5
    batch = 1
    channels = 1

    input_ = (
        torch.arange(0, timesteps).float().view(timesteps, 1, 1) / timesteps
    )
    input_ = input_.expand(timesteps, batch, channels)

    return input_.to(device)


@pytest.fixture(scope="module")
def input_tensor_single_batch_multiple_channel(device):
    timesteps = 5
    batch = 1
    channels = 4

    input_ = (
        torch.arange(0, timesteps).float().view(timesteps, 1, 1) / timesteps
    )
    input_ = input_.expand(timesteps, batch, channels)

    return input_.to(device)


@pytest.fixture(scope="module")
def input_tensor_multiple_batches_single_channel(device):
    timesteps = 5
    batch = 2
    channels = 1

    input_ = (
        torch.arange(0, timesteps).float().view(timesteps, 1, 1) / timesteps
    )
    input_ = input_.expand(timesteps, batch, channels)

    return input_.to(device)


@pytest.fixture(scope="module")
def input_tensor_batch_multiple(device):
    timesteps = 5
    batch = 2
    channels = 4

    input_ = (
        torch.arange(0, timesteps).float().view(timesteps, 1, 1) / timesteps
    )
    input_ = input_.expand(timesteps, batch, channels)

    return input_.to(device)


# Channel configuration tests (output=False to validate membrane only)
def test_single_batch_single_channel(
    linearleaky_single_channel, input_tensor_single_batch_single_channel
):
    output = linearleaky_single_channel.forward(
        input_tensor_single_batch_single_channel
    )

    assert input_tensor_single_batch_single_channel.shape == output.shape
    assert (
        (input_tensor_single_batch_single_channel <= 1).all().item()
    ), "Some input elements are greater than 1."
    assert (
        output.shape == input_tensor_single_batch_single_channel.shape
    ), "Output shape does not match input shape."
    assert (
        torch.isfinite(output).all().item()
    ), "Output contains non-finite values."
    assert (output.std() > 0).item() or (
        output.abs().sum() > 0
    ).item(), "Output is constant or all zeros unexpectedly."


def test_single_batch_multi_channel(
    linearleaky_multi_channel, input_tensor_single_batch_multiple_channel
):
    output = linearleaky_multi_channel.forward(
        input_tensor_single_batch_multiple_channel
    )

    assert input_tensor_single_batch_multiple_channel.shape == output.shape
    assert (
        (input_tensor_single_batch_multiple_channel <= 1).all().item()
    ), "Some input elements are greater than 1."
    assert (
        output.shape == input_tensor_single_batch_multiple_channel.shape
    ), "Output shape does not match input shape."
    assert (
        torch.isfinite(output).all().item()
    ), "Output contains non-finite values."
    assert (output.std() > 0).item() or (
        output.abs().sum() > 0
    ).item(), "Output is constant or all zeros unexpectedly."


def test_multi_batch_single_channel(
    linearleaky_single_channel, input_tensor_multiple_batches_single_channel
):
    output = linearleaky_single_channel.forward(
        input_tensor_multiple_batches_single_channel
    )

    assert input_tensor_multiple_batches_single_channel.shape == output.shape
    assert (
        (input_tensor_multiple_batches_single_channel <= 1).all().item()
    ), "Some input elements are greater than 1."
    assert (
        output.shape == input_tensor_multiple_batches_single_channel.shape
    ), "Output shape does not match input shape."
    assert (
        torch.isfinite(output).all().item()
    ), "Output contains non-finite values."
    assert (output.std() > 0).item() or (
        output.abs().sum() > 0
    ).item(), "Output is constant or all zeros unexpectedly."


def test_multi_batch_multi_channel(
    linearleaky_multi_channel, input_tensor_batch_multiple
):
    output = linearleaky_multi_channel.forward(input_tensor_batch_multiple)

    assert input_tensor_batch_multiple.shape == output.shape
    assert (
        (input_tensor_batch_multiple <= 1).all().item()
    ), "Some input elements are greater than 1."
    assert (
        output.shape == input_tensor_batch_multiple.shape
    ), "Output shape does not match input shape."
    assert (
        torch.isfinite(output).all().item()
    ), "Output contains non-finite values."
    assert (output.std() > 0).item() or (
        output.abs().sum() > 0
    ).item(), "Output is constant or all zeros unexpectedly."


def test_multi_beta_forward(
    linearleaky_multi_beta, input_tensor_single_batch_multiple_channel
):
    output = linearleaky_multi_beta.forward(
        input_tensor_single_batch_multiple_channel
    )

    assert input_tensor_single_batch_multiple_channel.shape == output.shape
    assert (
        (input_tensor_single_batch_multiple_channel <= 1).all().item()
    ), "Some input elements are greater than 1."
    assert (
        output.shape == input_tensor_single_batch_multiple_channel.shape
    ), "Output shape does not match input shape."
    assert (
        torch.isfinite(output).all().item()
    ), "Output contains non-finite values."
    assert (output.std() > 0).item() or (
        output.abs().sum() > 0
    ).item(), "Output is constant or all zeros unexpectedly."

    # Verify learn_beta is a learnable parameter
    assert isinstance(
        linearleaky_multi_beta.tau, nn.Parameter
    ), "learn_beta should be a learnable parameter"


def test_output_channel_change_shape(device):
    """
    When in_features != out_features, LinearLeaky should output tensors with
    the out_features channel dimension.
    """
    timesteps = 7
    batch = 3
    in_features = 3
    out_features = 5
    x = torch.randn(timesteps, batch, in_features, device=device)

    module = LinearLeaky(
        beta=0.9,
        in_features=in_features,
        out_features=out_features,
        output=False,
    ).to(device)

    y = module(x)
    assert y.shape == (timesteps, batch, out_features)


def test_equivalence_vs_external_linear_and_stateleaky(device):
    """
    LinearLeaky(x) should be equivalent to: StateLeaky(Linear(x)) given the
    same parameters.
    """
    torch.manual_seed(0)
    timesteps = 8
    batch = 2
    in_features = 4
    out_features = 4
    x = torch.randn(timesteps, batch, in_features, device=device)

    beta = torch.full((out_features,), 0.9, device=device)

    # Reference path: external linear + StateLeaky
    ext_linear = nn.Linear(in_features, out_features, bias=True, device=device)
    lif_ref = StateLeaky(beta=beta, channels=out_features, output=True).to(
        device
    )

    # Under test: LinearLeaky with internal linear
    lif_under_test = LinearLeaky(
        beta=beta,
        in_features=in_features,
        out_features=out_features,
        output=True,
    ).to(device)

    # Synchronize parameters (copy weights and biases)
    with torch.no_grad():
        lif_under_test.linear.weight.copy_(ext_linear.weight)
        if ext_linear.bias is not None:
            lif_under_test.linear.bias.copy_(ext_linear.bias)

    # Run both paths
    spk_ref, mem_ref = lif_ref(ext_linear(x))
    spk_ut, mem_ut = lif_under_test(x)

    assert torch.allclose(mem_ref, mem_ut, atol=1e-6)
    assert torch.allclose(spk_ref, spk_ut, atol=1e-6)


def test_chunking_with_gd_internal_linear():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 256
    chunk_size = 64
    channels = 32
    timesteps = 4096

    input_tensor = (
        torch.arange(
            1,
            timesteps * batch_size * channels + 1,
            device=device,
            dtype=torch.float32,
        )
        .view(batch_size, channels, timesteps)
        .contiguous()
        .permute(2, 0, 1)  # (T, B, C)
    )

    beta = torch.full((channels,), 0.9).to(device)
    module = LinearLeaky(
        beta=beta,
        in_features=channels,
        out_features=channels,
        output=True,
        bias=False,
    ).to(device)

    # 1. Get ground truth with no chunking to setup comparison
    torch.set_grad_enabled(True)
    module.zero_grad()

    spk_full, mem_full = module(input_tensor)
    loss_full = spk_full.sum()
    loss_full.backward()
    grad_full = module.linear.weight.grad.clone()

    # 2. Get gradients with chunking to compare
    module.zero_grad()

    spk_chunks = []

    for b_start in range(0, batch_size, chunk_size):
        b_end = min(b_start + chunk_size, batch_size)

        # select a chunk of the input tensor
        input_chunk = input_tensor[:, b_start:b_end, :]

        # forward pass on chunk
        spk_chunk, _ = module(input_chunk)
        spk_chunks.append(spk_chunk)

        # backward pass on chunk
        loss_chunk = spk_chunk.sum()
        loss_chunk.backward()

        # assert grad is populated
        assert (
            module.linear.weight.grad is not None
        ), "Gradient is not populated."

    # concatenate chunks into a single tensor
    spk_chunked = torch.cat(spk_chunks, dim=1)
    grad_chunked = module.linear.weight.grad.clone()

    # assertions
    assert (
        spk_full.shape == spk_chunked.shape
    ), "Chunked spike tensor shape mismatch."
    assert torch.allclose(
        spk_full, spk_chunked, atol=1e-6
    ), "Forward pass (spikes) do not match."
    assert torch.allclose(
        grad_full, grad_chunked, atol=1e-6
    ), "Backward pass (gradients) do not match."


def test_kernel_truncation_impulse(device):
    """
    With a single impulse at t=0 and identity linear, LinearLeaky reduces to
    StateLeaky. Kernel truncation keeps first K taps and zeros after.
    """
    timesteps = 10
    batch = 1
    channels = 3

    beta = torch.tensor([0.9, 0.8, 0.7], device=device)

    # Create impulse input: 1 at t=0, zeros after
    x = torch.zeros(timesteps, batch, channels, device=device)
    x[0, 0, :] = 1.0

    # Build module with identity linear so behavior equals StateLeaky
    lif_full = LinearLeaky(
        beta=beta,
        in_features=channels,
        out_features=channels,
        output=False,
    ).to(device)
    lif_trunc = LinearLeaky(
        beta=beta,
        in_features=channels,
        out_features=channels,
        output=False,
        kernel_truncation_steps=4,
    ).to(device)

    with torch.no_grad():
        lif_full.linear.weight.copy_(torch.eye(channels, device=device))
        lif_full.linear.bias.zero_()
        lif_trunc.linear.weight.copy_(torch.eye(channels, device=device))
        lif_trunc.linear.bias.zero_()

    # Forward
    out_full = lif_full.forward(x)
    out_trunc = lif_trunc.forward(x)

    # Expected analytical decay (same as StateLeaky for impulse)
    tau = 1 / (1 - beta)
    t = torch.arange(timesteps, device=device).view(timesteps, 1, 1)
    expected_full = torch.exp(-t / tau.view(1, 1, channels))

    expected_trunc = expected_full.clone()
    expected_trunc[4:, :, :] = 0.0

    assert torch.allclose(out_full, expected_full, atol=1e-6)
    assert torch.allclose(out_trunc, expected_trunc, atol=1e-6)
    assert not torch.allclose(out_full[4:], out_trunc[4:], atol=1e-8)


def test_fail_fast_unimplemented_apis(device):
    """
    LinearLeaky computes membrane via a feed-forward causal convolution over
    the entire sequence and does not maintain a stepwise recurrent hidden
    state.

    As a result, inhibition and stepwise reset signaling (mem_reset) do not
    conceptually apply here (inherited from StateLeaky).
    """
    channels = 2
    lif = LinearLeaky(
        beta=0.9, in_features=channels, out_features=channels, output=False
    )

    # fire_inhibition should be unsupported
    with pytest.raises(NotImplementedError):
        lif.fire_inhibition(batch_size=1, mem=torch.zeros(1, channels))

    # mem_reset should be unsupported
    with pytest.raises(NotImplementedError):
        lif.mem_reset(torch.zeros(1, channels))


def test_warn_on_inert_spike_settings(device):
    """
    When output=False, LinearLeaky (via StateLeaky) does not emit spikes;
    spike-only knobs are inert. We warn to catch config mismatches.
    """
    with pytest.warns(UserWarning):
        _ = LinearLeaky(
            beta=0.9,
            in_features=2,
            out_features=2,
            output=False,
            spike_grad=lambda x: x,  # any non-default surrogate
        )

    with pytest.warns(UserWarning):
        _ = LinearLeaky(
            beta=0.9,
            in_features=2,
            out_features=2,
            output=False,
            surrogate_disable=True,
        )

    with pytest.warns(UserWarning):
        _ = LinearLeaky(
            beta=0.9,
            in_features=2,
            out_features=2,
            output=False,
            learn_threshold=True,
        )

    with pytest.warns(UserWarning):
        _ = LinearLeaky(
            beta=0.9,
            in_features=2,
            out_features=2,
            output=False,
            graded_spikes_factor=2.0,
        )

    with pytest.warns(UserWarning):
        _ = LinearLeaky(
            beta=0.9,
            in_features=2,
            out_features=2,
            output=False,
            learn_graded_spikes_factor=True,
        )


if __name__ == "__main__":
    pytest.main()

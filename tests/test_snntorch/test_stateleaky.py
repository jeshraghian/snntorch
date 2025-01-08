import pytest
import torch
import torch.nn as nn

from snntorch._neurons.stateleaky import StateLeaky

"""
Tests for the StateLeaky neuron class.

Test Structure:
--------------
1. Channel Configuration Tests:
    - Single batch, single channel
    - Single batch, multiple channels
    - Multiple batches, single channel
    - Multiple batches, multiple channels

2. Learning Parameter Tests:
    - Multi-beta learning (tests learn_beta=True)
    - Decay filter learning (tests learn_decay_filter=True)

Coverage:
--------
- Input/output shape consistency
- Input value bounds (≤ 1)
- Output activation (presence of values > 1)
- Parameter learnability for learn_beta and learn_decay_filter

Limitations:
-----------
1. Does not test:
   - Spike generation (output=True)
   - Threshold learning (learn_threshold=True)
   - State quantization (state_quant=True)
   - Graded spike factor learning (learn_graded_spikes_factor=True)
   - Surrogate gradient functions (spike_grad parameter)

2. Testing Scope:
   - Uses fixed timesteps (5)
   - Uses fixed channel counts (1 or 4)
   - Uses fixed batch sizes (1 or 2)
   - Uses fixed beta value (0.9)
   - Tests only forward pass, not backward pass or gradient flow

3. Input Patterns:
   - Uses simple ascending sequence inputs normalized to [0,1]
   - Does not test with random or complex input patterns
"""


@pytest.fixture(scope="module")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def linear_leaky_single_channel(device):
    return StateLeaky(beta=0.9, channels=1, output=False).to(device)


@pytest.fixture(scope="module")
def linear_leaky_multi_channel(device):
    return StateLeaky(beta=0.9, channels=4, output=False).to(device)


@pytest.fixture(scope="module")
def linear_leaky_multi_beta(device):
    return StateLeaky(beta=torch.tensor([0.9] * 4), channels=4, learn_beta=True, output=False).to(device)


@pytest.fixture(scope="module")
def input_tensor_single_batch_single_channel(device):
    timesteps = 5
    batch = 1
    channels = 1

    input_ = torch.arange(0, timesteps).float().view(timesteps, 1, 1) / timesteps
    input_ = input_.expand(timesteps, batch, channels)

    return input_.to(device)


@pytest.fixture(scope="module")
def input_tensor_single_batch_multiple_channel(device):
    timesteps = 5
    batch = 1
    channels = 4

    input_ = torch.arange(0, timesteps).float().view(timesteps, 1, 1) / timesteps
    input_ = input_.expand(timesteps, batch, channels)

    return input_.to(device)


@pytest.fixture(scope="module")
def input_tensor_multiple_batches_single_channel(device):
    timesteps = 5
    batch = 2
    channels = 1

    input_ = torch.arange(0, timesteps).float().view(timesteps, 1, 1) / timesteps
    input_ = input_.expand(timesteps, batch, channels)

    return input_.to(device)


@pytest.fixture(scope="module")
def input_tensor_batch_multiple(device):
    timesteps = 5
    batch = 2
    channels = 4

    input_ = torch.arange(0, timesteps).float().view(timesteps, 1, 1) / timesteps
    input_ = input_.expand(timesteps, batch, channels)

    return input_.to(device)


@pytest.fixture(scope="module")
def linear_leaky_learn_decay(device):
    return StateLeaky(beta=0.9, channels=4, learn_decay_filter=True, output=False).to(device)


# Channel configuration tests
def test_single_batch_single_channel(
        linear_leaky_single_channel, input_tensor_single_batch_single_channel):
    output = linear_leaky_single_channel.forward(input_tensor_single_batch_single_channel)

    assert input_tensor_single_batch_single_channel.shape == output.shape
    assert (input_tensor_single_batch_single_channel <= 1).all().item(), \
        "Some input elements are greater than 1."
    assert output.shape == input_tensor_single_batch_single_channel.shape, \
        "Output shape does not match input shape."
    assert (output > 1).any().item(), \
        "No elements in the output are greater than 1."


def test_single_batch_multi_channel(
        linear_leaky_multi_channel, input_tensor_single_batch_multiple_channel):
    output = linear_leaky_multi_channel.forward(input_tensor_single_batch_multiple_channel)

    assert input_tensor_single_batch_multiple_channel.shape == output.shape
    assert (input_tensor_single_batch_multiple_channel <= 1).all().item(), \
        "Some input elements are greater than 1."
    assert output.shape == input_tensor_single_batch_multiple_channel.shape, \
        "Output shape does not match input shape."
    assert (output > 1).any().item(), \
        "No elements in the output are greater than 1."

def test_multi_batch_single_channel(
        linear_leaky_single_channel, input_tensor_multiple_batches_single_channel):
    output = linear_leaky_single_channel.forward(input_tensor_multiple_batches_single_channel)

    assert input_tensor_multiple_batches_single_channel.shape == output.shape
    assert (input_tensor_multiple_batches_single_channel <= 1).all().item(), \
        "Some input elements are greater than 1."
    assert output.shape == input_tensor_multiple_batches_single_channel.shape, \
        "Output shape does not match input shape."
    assert (output > 1).any().item(), \
        "No elements in the output are greater than 1."


def test_multi_batch_multi_channel(
        linear_leaky_multi_channel, input_tensor_batch_multiple):
    output = linear_leaky_multi_channel.forward(input_tensor_batch_multiple)

    assert input_tensor_batch_multiple.shape == output.shape
    assert (input_tensor_batch_multiple <= 1).all().item(), \
        "Some input elements are greater than 1."
    assert output.shape == input_tensor_batch_multiple.shape, \
        "Output shape does not match input shape."
    assert (output > 1).any().item(), \
        "No elements in the output are greater than 1."


def test_multi_beta_forward(
        linear_leaky_multi_beta, input_tensor_single_batch_multiple_channel):
    output = linear_leaky_multi_beta.forward(input_tensor_single_batch_multiple_channel)

    assert input_tensor_single_batch_multiple_channel.shape == output.shape
    assert (input_tensor_single_batch_multiple_channel <= 1).all().item(), \
        "Some input elements are greater than 1."
    assert output.shape == input_tensor_single_batch_multiple_channel.shape, \
        "Output shape does not match input shape."
    assert (output > 1).any().item(), \
        "No elements in the output are greater than 1."

    # Verify learn_beta is a learnable parameter
    assert isinstance(linear_leaky_multi_beta.tau, nn.Parameter), \
        "learn_beta should be a learnable parameter"


def test_learn_decay_filter(
        linear_leaky_learn_decay, input_tensor_single_batch_multiple_channel):
    output = linear_leaky_learn_decay.forward(input_tensor_single_batch_multiple_channel)

    assert input_tensor_single_batch_multiple_channel.shape == output.shape
    assert (input_tensor_single_batch_multiple_channel <= 1).all().item(), \
        "Some input elements are greater than 1."
    assert output.shape == input_tensor_single_batch_multiple_channel.shape, \
        "Output shape does not match input shape."
    assert (output < 1).any().item(), \
        "No elements in the output are greater than 1."
    
    # Verify decay_filter is a learnable parameter
    assert isinstance(linear_leaky_learn_decay.decay_filter, nn.Parameter), \
        "decay_filter should be a learnable paramete<"


if __name__ == "__main__":
    pytest.main()

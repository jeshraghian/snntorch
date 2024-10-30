import pytest
import torch
from snntorch._neurons.stateleaky import StateLeaky  # Adjust the import path as needed


@pytest.fixture(scope="module")
def device():
    # Fixture to use GPU if available, otherwise fall back to CPU
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def linear_leaky_instance(device):
    # Fixture to initialize the StateLeaky instance
    return StateLeaky(beta=0.9, output=False).to(device)

@pytest.fixture(scope="module")
def linear_leaky_instance_multi_beta(device):
    # Fixture to initialize the StateLeaky instance
    return StateLeaky(beta=torch.tensor([0.9, 0.9, 0.9, 0.9]), output=False).to(device)

@pytest.fixture(scope="module")
def input_tensor_single_batch_single_channel(device):
    # Define the shape parameters
    timesteps = 5
    batch = 1
    channels = 1
    
    # Create a sequence for each timestep and expand along batch and channel dimensions
    input_ = torch.arange(0, timesteps).float().view(timesteps, 1, 1) / timesteps  # Shape (timesteps, 1, 1)
    input_ = input_.expand(timesteps, batch, channels)  # Expand to (timesteps, batch, channels)
    
    # Move to the specified device (e.g., CPU or GPU)
    return input_.to(device)

@pytest.fixture(scope="module")
def input_tensor_single_batch_multiple_channel(device):
    # Define the shape parameters
    timesteps = 5
    batch = 1
    channels = 4
    
    # Create a sequence for each timestep and expand along batch and channel dimensions
    input_ = torch.arange(0, timesteps).float().view(timesteps, 1, 1) / timesteps  # Shape (timesteps, 1, 1)
    input_ = input_.expand(timesteps, batch, channels)  # Expand to (timesteps, batch, channels)
    
    # Move to the specified device (e.g., CPU or GPU)
    return input_.to(device)

@pytest.fixture(scope="module")
def input_tensor_multiple_batches_single_channel(device):
    # Define the shape parameters
    timesteps = 5
    batch = 2
    channels = 1
    
    # Create a sequence for each timestep and expand along batch and channel dimensions
    input_ = torch.arange(0, timesteps).float().view(timesteps, 1, 1) / timesteps  # Shape (timesteps, 1, 1)
    input_ = input_.expand(timesteps, batch, channels)  # Expand to (timesteps, batch, channels)
    
    # Move to the specified device (e.g., CPU or GPU)
    return input_.to(device)

@pytest.fixture(scope="module")
def input_tensor_batch_multiple(device):
    # Define the shape parameters
    timesteps = 5
    batch = 2
    channels = 4
    
    # Create a sequence for each timestep and expand along batch and channel dimensions
    input_ = torch.arange(0, timesteps).float().view(timesteps, 1, 1) / timesteps  # Shape (timesteps, 1, 1)
    input_ = input_.expand(timesteps, batch, channels)  # Expand to (timesteps, batch, channels)
    
    # Move to the specified device (e.g., CPU or GPU)
    return input_.to(device)


def test_forward_method_correctness_single_batch_single_channel(
        linear_leaky_instance, input_tensor_single_batch_single_channel):
    output = linear_leaky_instance.forward(input_tensor_single_batch_single_channel)

    assert input_tensor_single_batch_single_channel.shape == output.shape

    # Ensure all input elements are <= 1
    assert (input_tensor_single_batch_single_channel <= 1).all().item(), \
        "Some input elements are greater than 1."

    # Ensure the output shape matches the input shape
    assert output.shape == input_tensor_single_batch_single_channel.shape, \
        "Output shape does not match input shape."

    # Check if there are some elements in the output that are > 1
    assert (output > 1).any().item(), \
        "No elements in the output are greater than 1."


def test_forward_method_correctness_multiple_batches_multiple_channels(
    linear_leaky_instance, input_tensor_batch_multiple
):
    output = linear_leaky_instance.forward(input_tensor_batch_multiple)

    assert input_tensor_batch_multiple.shape == output.shape

    print("input_tensor_batch_multiple: ", input_tensor_batch_multiple)

    # Ensure all input elements are <= 1
    assert (input_tensor_batch_multiple <= 1).all().item(), \
        "Some input elements are greater than 1."

    # Ensure the output shape matches the input shape
    assert output.shape == input_tensor_batch_multiple.shape, \
        "Output shape does not match input shape."

    # Check if there are some elements in the output that are > 1
    assert (output > 1).any().item(), \
        "No elements in the output are greater than 1."


def test_forward_method_correctness_multiple_batches_single_channel(
        linear_leaky_instance, input_tensor_multiple_batches_single_channel):
    output = linear_leaky_instance.forward(input_tensor_multiple_batches_single_channel)

    assert input_tensor_multiple_batches_single_channel.shape == output.shape

    print("input_tensor_batch_multiple: ", input_tensor_multiple_batches_single_channel)

    # Ensure all input elements are <= 1
    assert (input_tensor_multiple_batches_single_channel <= 1).all().item(), \
        "Some input elements are greater than 1."

    # Ensure the output shape matches the input shape
    assert output.shape == input_tensor_multiple_batches_single_channel.shape, \
        "Output shape does not match input shape."

    # Check if there are some elements in the output that are > 1
    assert (output > 1).any().item(), \
        "No elements in the output are greater than 1."


def test_forward_method_correctness_single_batch_multiple_channel(
        linear_leaky_instance_multi_beta, input_tensor_single_batch_multiple_channel):
    output = linear_leaky_instance_multi_beta.forward(input_tensor_single_batch_multiple_channel)

    assert input_tensor_single_batch_multiple_channel.shape == output.shape

    # Ensure all input elements are <= 1
    assert (input_tensor_single_batch_multiple_channel <= 1).all().item(), \
        "Some input elements are greater than 1."

    # Ensure the output shape matches the input shape
    assert output.shape == input_tensor_single_batch_multiple_channel.shape, \
        "Output shape does not match input shape."

    # Check if there are some elements in the output that are > 1
    assert (output > 1).any().item(), \
        "No elements in the output are greater than 1."

if __name__ == "__main__":
    pytest.main()

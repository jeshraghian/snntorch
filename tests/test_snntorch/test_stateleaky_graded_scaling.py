import pytest
import torch

from snntorch._neurons.stateleaky import StateLeaky


@pytest.mark.parametrize("factor", [0.0, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("learn_factor", [False, True])
def test_stateleaky_graded_spikes_scaled_once(factor, learn_factor):
    neuron = StateLeaky(
        beta=0.9,
        channels=2,
        graded_spikes_factor=factor,
        learn_graded_spikes_factor=learn_factor,
    )
    inputs = torch.tensor([[[0.0, 2.0]], [[2.0, 0.0]]])
    spikes, membrane = neuron(inputs)
    binary_spikes = (membrane > neuron.threshold).to(membrane.dtype)
    torch.testing.assert_close(spikes, binary_spikes * factor)
    if learn_factor:
        spikes.sum().backward()
        torch.testing.assert_close(
            neuron.graded_spikes_factor.grad, binary_spikes.sum()
        )


def test_stateleaky_per_channel_graded_spikes_scaled_once():
    factor = torch.tensor([0.5, 2.0])
    neuron = StateLeaky(
        beta=0.9,
        channels=2,
        graded_spikes_factor=factor,
        learn_graded_spikes_factor=True,
    )
    spikes, membrane = neuron(torch.tensor([[[0.0, 2.0]], [[2.0, 0.0]]]))
    binary_spikes = (membrane > neuron.threshold).to(membrane.dtype)
    torch.testing.assert_close(spikes, binary_spikes * factor)
    spikes.sum().backward()
    torch.testing.assert_close(
        neuron.graded_spikes_factor.grad, binary_spikes.sum(dim=(0, 1))
    )


def test_stateleaky_membrane_unchanged_by_graded_spikes():
    inputs = torch.tensor([[[0.0, 2.0]], [[2.0, 0.0]]])
    baseline = StateLeaky(beta=0.9, channels=2, output=False)
    with pytest.warns(UserWarning, match="unused when output=False"):
        scaled = StateLeaky(
            beta=0.9,
            channels=2,
            output=False,
            graded_spikes_factor=2.0,
        )
    torch.testing.assert_close(scaled(inputs), baseline(inputs))

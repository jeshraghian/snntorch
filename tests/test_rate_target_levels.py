"""Custom membrane target levels in time-varying rate encoding."""

import pytest
import torch

from snntorch import spikegen


@pytest.mark.parametrize(
    "encode", [spikegen.targets_rate, spikegen.targets_convert]
)
@pytest.mark.parametrize("first_spike_time", [0, 1])
@pytest.mark.parametrize(
    "on_target,off_target",
    [(1, 0), (0, -1), (-1, -2), (-2, -1), (0.5, -0.5), (2, 0.25)],
)
def test_time_varying_rate_target_levels(
    encode, first_spike_time, on_target, off_target
):
    targets = torch.tensor([2, 0])
    expected = torch.full((6, 2, 3), float(off_target))
    for sample, target in enumerate(targets):
        for neuron in range(3):
            # Correct classes spike every two steps, incorrect every four.
            interval = 2 if neuron == target else 4
            expected[first_spike_time::interval, sample, neuron] = on_target

    actual = encode(
        targets,
        num_classes=3,
        num_steps=6,
        first_spike_time=first_spike_time,
        correct_rate=0.5,
        incorrect_rate=0.25,
        on_target=on_target,
        off_target=off_target,
    )

    torch.testing.assert_close(actual, expected)

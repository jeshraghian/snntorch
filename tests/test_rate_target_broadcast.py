"""Regression coverage for time-axis broadcasting in target rate coding."""

import pytest
import torch

from snntorch import spikegen


@pytest.mark.parametrize("num_classes", [3, 6])
@pytest.mark.parametrize("incorrect_rate", [0, 0.25])
@pytest.mark.parametrize("first_spike_time", [0, 1])
def test_target_rates_broadcast_over_time(
    num_classes, incorrect_rate, first_spike_time
):
    targets = torch.tensor([0, num_classes - 1])
    expected = torch.zeros(6, 2, num_classes)
    correct_times = range(first_spike_time, 6, 2)
    incorrect_times = range(first_spike_time, 6, 4) if incorrect_rate else []
    for sample, label in enumerate(targets.tolist()):
        for step in correct_times:
            expected[step, sample, label] = 1
        for step in incorrect_times:
            expected[step, sample, :] = 1
            expected[step, sample, label] = int(step in correct_times)

    actual = spikegen.targets_rate(
        targets,
        num_classes=num_classes,
        num_steps=6,
        correct_rate=0.5,
        incorrect_rate=incorrect_rate,
        first_spike_time=first_spike_time,
    )

    torch.testing.assert_close(actual, expected)

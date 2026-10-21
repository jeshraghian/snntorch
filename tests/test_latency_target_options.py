"""Regression coverage for the latency target conversion wrapper."""

import pytest
import torch

from snntorch import spikegen


@pytest.mark.parametrize("linear", [False, True])
@pytest.mark.parametrize("interpolate", [False, True])
def test_targets_convert_latency_preserves_target_options(linear, interpolate):
    labels = torch.tensor([0, 2])
    expected = torch.full((5, 2, 3), 0.2)
    for batch, label in enumerate(labels.tolist()):
        expected[0, batch, label] = 1.5
        for other in range(3):
            if other != label:
                if interpolate:
                    expected[:, batch, other] = torch.linspace(0.2, 1.5, 5)
                else:
                    expected[4, batch, other] = 1.5

    actual = spikegen.targets_convert(
        labels,
        num_classes=3,
        code="latency",
        num_steps=5,
        normalize=True,
        linear=linear,
        on_target=1.5,
        off_target=0.2,
        interpolate=interpolate,
    )

    torch.testing.assert_close(actual, expected)


def test_targets_convert_latency_preserves_epsilon():
    labels = torch.tensor([0, 2])
    expected = torch.zeros(20, 2, 3)
    for batch, label in enumerate(labels.tolist()):
        expected[0, batch, label] = 1
        for other in range(3):
            if other != label:
                # log((threshold + epsilon) / epsilon) = log(2), rounded to 1.
                expected[1, batch, other] = 1

    actual = spikegen.targets_convert(
        labels, num_classes=3, code="latency", num_steps=20, epsilon=0.01
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("linear", [False, True])
def test_targets_convert_latency_defaults_match_direct_encoder(linear):
    labels = torch.tensor([0, 2])
    options = {"num_steps": 5, "normalize": True, "linear": linear}
    expected = spikegen.targets_latency(labels, num_classes=3, **options)
    actual = spikegen.targets_convert(
        labels, num_classes=3, code="latency", **options
    )

    torch.testing.assert_close(actual, expected)

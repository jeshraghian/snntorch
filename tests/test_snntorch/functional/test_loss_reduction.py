"""Reduction of class-weighted spike and membrane losses."""

import pytest
import torch
import torch.nn.functional as F

import snntorch.functional as sf


@pytest.mark.parametrize(
    "loss_name",
    [
        "ce_rate_loss",
        "ce_count_loss",
        "ce_max_membrane_loss",
        "mse_count_loss",
        "mse_membrane_loss",
    ],
)
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_weighted_loss_reduction(loss_name, reduction):
    # Unequal batch/output sizes distinguish sample and output axes.
    output = torch.tensor(
        [
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]],
        ],
        requires_grad=True,
    )
    targets = torch.tensor([1, 0])
    weights = torch.tensor([2.0, 0.5, 3.0])
    steps = output.size(0)

    if loss_name == "ce_rate_loss":
        errors = torch.stack(
            [
                F.cross_entropy(step, targets, reduction="none")
                for step in output
            ]
        ).mean(0)
    elif loss_name == "ce_count_loss":
        errors = F.cross_entropy(output.sum(0), targets, reduction="none")
    elif loss_name == "ce_max_membrane_loss":
        errors = F.cross_entropy(
            output.max(0).values, targets, reduction="none"
        )
    elif loss_name == "mse_count_loss":
        desired = F.one_hot(targets, num_classes=3) * steps
        errors = (output.sum(0) - desired).square() / steps
    else:
        desired = F.one_hot(targets, num_classes=3)
        errors = (output - desired).square().mean(0)

    sample_weights = weights[targets]
    if errors.ndim == 2:
        sample_weights = sample_weights[:, None]
    expected = errors * sample_weights
    if reduction == "mean":
        expected = expected.mean()
    elif reduction == "sum":
        expected = expected.sum()

    actual = getattr(sf, loss_name)(reduction=reduction, weight=weights)(
        output, targets
    )
    torch.testing.assert_close(actual, expected)
    if reduction != "none":
        assert actual.ndim == 0
        expected_grad = torch.autograd.grad(
            expected, output, retain_graph=True
        )[0]
        actual.backward()
        torch.testing.assert_close(output.grad, expected_grad)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_weighted_temporal_mse_reduction(reduction):
    output = torch.zeros((4, 2, 3))
    times = torch.tensor([[0, 1, 3], [2, 0, 1]])
    for sample in range(2):
        for neuron in range(3):
            output[times[sample, neuron], sample, neuron] = 1
    output.requires_grad_()
    targets = torch.tensor([0, 1])
    weights = torch.tensor([2.0, 0.5, 3.0])
    desired = torch.tensor([[0, 3, 3], [3, 0, 3]])
    errors = ((times - desired) / 4).square() * weights[targets, None]
    expected = errors
    if reduction == "mean":
        expected = errors.mean()
    elif reduction == "sum":
        expected = errors.sum()

    actual = sf.mse_temporal_loss(
        on_target=0, off_target=3, reduction=reduction, weight=weights
    )(output, targets)
    torch.testing.assert_close(actual, expected)
    if reduction != "none":
        assert actual.ndim == 0
        # FirstSpike uses a -1 surrogate derivative at each first spike.
        expected_grad = torch.zeros_like(output)
        per_spike = -2 * (times - desired) / 16 * weights[targets, None]
        if reduction == "mean":
            per_spike /= errors.numel()
        for sample in range(2):
            for neuron in range(3):
                expected_grad[times[sample, neuron], sample, neuron] = (
                    per_spike[sample, neuron]
                )
        actual.backward()
        torch.testing.assert_close(output.grad, expected_grad)

#!/usr/bin/env python

"""Tests for Loss."""

import pytest
import snntorch as snn
import snntorch.functional as sf
import torch

torch.manual_seed(42)
tolerance = 1e-5


@pytest.fixture(scope="module")
def spike_predicted_():
    # shape: time_steps x batch_size x num_out_neurons x
    return torch.randint(2, (3, 3, 3)).float()


@pytest.fixture(scope="module")
def targets_labels_():
    return torch.tensor([1, 2, 0], dtype=torch.int64)


@pytest.fixture(scope="module")
def membrane_predicted_():
    # shape: time_steps x batch_size x num_out_neurons
    return torch.rand((3, 3, 3))


@pytest.fixture(scope="module")
def class_weights_():
    return torch.tensor([0.35, 0.50, 0.15], dtype=torch.float32)


def assert_approximate_equality(actual, expected):
    assert actual == pytest.approx(expected, abs=tolerance)


class TestLoss:
    def test_ce_rate_loss_base(self, spike_predicted_, targets_labels_):
        loss_fn = sf.ce_rate_loss()

        assert loss_fn.weight is None
        assert loss_fn.reduction == "mean"

    def test_ce_rate_loss_unreduced(self, spike_predicted_, targets_labels_):
        unreduced_loss_fn = sf.ce_rate_loss(reduction="none")
        unreduced_loss = unreduced_loss_fn(spike_predicted_, targets_labels_)

        reduced_loss_fn = sf.ce_rate_loss()
        reduced_loss = reduced_loss_fn(spike_predicted_, targets_labels_)

        assert_approximate_equality(
            unreduced_loss.mean().item(), reduced_loss.item()
        )

    def test_ce_rate_loss_weighted(
        self, spike_predicted_, targets_labels_, class_weights_
    ):
        weighted_loss_fn = sf.ce_rate_loss(weight=class_weights_)
        weighted_loss = weighted_loss_fn(spike_predicted_, targets_labels_)

        # unreduced, unweighted loss
        vanilla_loss_fn = sf.ce_rate_loss(reduction="none")
        vanilla_loss = vanilla_loss_fn(spike_predicted_, targets_labels_)
        # weight multiplier
        weight_multiplier = class_weights_[targets_labels_]
        # expectation
        expected_weighted_loss = (vanilla_loss * weight_multiplier).mean()

        assert_approximate_equality(
            weighted_loss.item(), expected_weighted_loss.item()
        )

    def test_ce_count_loss_base(self, spike_predicted_, targets_labels_):
        loss_fn = sf.ce_count_loss()

        assert loss_fn.weight is None
        assert loss_fn.reduction == "mean"

    def test_ce_count_loss_unreduced(self, spike_predicted_, targets_labels_):
        unreduced_loss_fn = sf.ce_count_loss(reduction="none")
        unreduced_loss = unreduced_loss_fn(spike_predicted_, targets_labels_)

        reduced_loss_fn = sf.ce_count_loss()
        reduced_loss = reduced_loss_fn(spike_predicted_, targets_labels_)

        assert_approximate_equality(
            unreduced_loss.mean().item(), reduced_loss.item()
        )

    def test_ce_count_loss_population_code_dtype(self):
        # population code should preserve the input dtype (float64)
        loss_fn = sf.ce_count_loss(
            population_code=True,
            num_classes=2,
            weight=torch.tensor([1.0, 2.0], dtype=torch.float64),
        )
        spike_predicted = torch.randn(4, 1, 4, dtype=torch.float64)
        targets = torch.tensor([1])

        loss = loss_fn(spike_predicted, targets)

        assert loss.dtype == torch.float64

    def test_ce_count_loss_weighted(
        self, spike_predicted_, targets_labels_, class_weights_
    ):
        weighted_loss_fn = sf.ce_count_loss(weight=class_weights_)
        weighted_loss = weighted_loss_fn(spike_predicted_, targets_labels_)

        # unreduced, unweighted loss
        vanilla_loss_fn = sf.ce_count_loss(reduction="none")
        vanilla_loss = vanilla_loss_fn(spike_predicted_, targets_labels_)
        # weight multiplier
        weight_multiplier = class_weights_[targets_labels_]
        # expectation
        expected_weighted_loss = (vanilla_loss * weight_multiplier).mean()

        assert_approximate_equality(
            weighted_loss.item(), expected_weighted_loss.item()
        )

    def test_ce_max_membrane_loss_base(
        self, membrane_predicted_, targets_labels_
    ):
        loss_fn = sf.ce_max_membrane_loss()

        assert loss_fn.weight is None
        assert loss_fn.reduction == "mean"

    def test_ce_max_membrane_loss_unreduced(
        self, membrane_predicted_, targets_labels_
    ):
        unreduced_loss_fn = sf.ce_max_membrane_loss(reduction="none")
        unreduced_loss = unreduced_loss_fn(
            membrane_predicted_, targets_labels_
        )

        reduced_loss_fn = sf.ce_max_membrane_loss()
        reduced_loss = reduced_loss_fn(membrane_predicted_, targets_labels_)

        assert_approximate_equality(
            unreduced_loss.mean().item(), reduced_loss.item()
        )

    def test_ce_max_membrane_loss_weighted(
        self, spike_predicted_, targets_labels_, class_weights_
    ):
        weighted_loss_fn = sf.ce_max_membrane_loss(weight=class_weights_)
        weighted_loss = weighted_loss_fn(spike_predicted_, targets_labels_)

        # unreduced, unweighted loss
        vanilla_loss_fn = sf.ce_max_membrane_loss(reduction="none")
        vanilla_loss = vanilla_loss_fn(spike_predicted_, targets_labels_)
        # weight multiplier
        weight_multiplier = class_weights_[targets_labels_]
        # expectation
        expected_weighted_loss = (vanilla_loss * weight_multiplier).mean()

        assert_approximate_equality(
            weighted_loss.item(), expected_weighted_loss.item()
        )

    def test_mse_count_loss_base(self, spike_predicted_, targets_labels_):
        loss_fn = sf.mse_count_loss()

        assert loss_fn.weight is None
        assert loss_fn.reduction == "mean"

    def test_mse_count_loss_unreduced(self, spike_predicted_, targets_labels_):
        unreduced_loss_fn = sf.mse_count_loss(reduction="none")
        unreduced_loss = unreduced_loss_fn(spike_predicted_, targets_labels_)

        reduced_loss_fn = sf.mse_count_loss()
        reduced_loss = reduced_loss_fn(spike_predicted_, targets_labels_)

        assert_approximate_equality(
            unreduced_loss.mean().item(), reduced_loss.item()
        )

    def test_mse_count_loss_weighted(
        self, spike_predicted_, targets_labels_, class_weights_
    ):
        weighted_loss_fn = sf.mse_count_loss(weight=class_weights_)
        weighted_loss = weighted_loss_fn(spike_predicted_, targets_labels_)

        # unreduced, unweighted loss
        vanilla_loss_fn = sf.mse_count_loss(reduction="none")
        vanilla_loss = vanilla_loss_fn(spike_predicted_, targets_labels_)
        # weight multiplier (applied per sample, i.e., along dim 0)
        weight_multiplier = class_weights_[targets_labels_].unsqueeze(-1)
        # expectation
        expected_weighted_loss = (vanilla_loss * weight_multiplier).mean()

        assert_approximate_equality(
            weighted_loss.item(), expected_weighted_loss.item()
        )

    def test_mse_membrane_loss_base(
        self, membrane_predicted_, targets_labels_
    ):
        loss_fn = sf.mse_membrane_loss()

        assert loss_fn.weight is None
        assert loss_fn.reduction == "mean"

    def test_mse_membrane_loss_unreduced(
        self, membrane_predicted_, targets_labels_
    ):
        unreduced_loss_fn = sf.mse_membrane_loss(reduction="none")
        unreduced_loss = unreduced_loss_fn(
            membrane_predicted_, targets_labels_
        )

        reduced_loss_fn = sf.mse_membrane_loss()
        reduced_loss = reduced_loss_fn(membrane_predicted_, targets_labels_)

        assert_approximate_equality(
            unreduced_loss.mean().item(), reduced_loss.item()
        )

    def test_mse_membrane_loss_weighted(
        self, spike_predicted_, targets_labels_, class_weights_
    ):
        weighted_loss_fn = sf.mse_membrane_loss(weight=class_weights_)
        weighted_loss = weighted_loss_fn(spike_predicted_, targets_labels_)

        # unreduced, unweighted loss
        vanilla_loss_fn = sf.mse_membrane_loss(reduction="none")
        vanilla_loss = vanilla_loss_fn(spike_predicted_, targets_labels_)
        # weight multiplier (applied per sample, i.e., along dim 0)
        weight_multiplier = class_weights_[targets_labels_].unsqueeze(-1)
        # expectation
        expected_weighted_loss = (vanilla_loss * weight_multiplier).mean()

        assert_approximate_equality(
            weighted_loss.item(), expected_weighted_loss.item()
        )

    def test_mse_temporal_loss_base(self, spike_predicted_, targets_labels_):
        loss_fn = sf.mse_temporal_loss(on_target=1, off_target=0)

        assert loss_fn.weight is None
        assert loss_fn.reduction == "mean"

    def test_mse_temporal_loss_unreduced(
        self, spike_predicted_, targets_labels_
    ):
        unreduced_loss_fn = sf.mse_temporal_loss(reduction="none")
        unreduced_loss = unreduced_loss_fn(spike_predicted_, targets_labels_)

        reduced_loss_fn = sf.mse_temporal_loss()
        reduced_loss = reduced_loss_fn(spike_predicted_, targets_labels_)

        assert_approximate_equality(
            unreduced_loss.mean().item(), reduced_loss.item()
        )

    def test_mse_temporal_loss_weighted(
        self, spike_predicted_, targets_labels_, class_weights_
    ):
        weighted_loss_fn = sf.mse_temporal_loss(weight=class_weights_)
        weighted_loss = weighted_loss_fn(spike_predicted_, targets_labels_)

        # unreduced, unweighted loss
        vanilla_loss_fn = sf.mse_temporal_loss(reduction="none")
        vanilla_loss = vanilla_loss_fn(spike_predicted_, targets_labels_)
        # weight multiplier (applied per sample, i.e., along dim 0)
        weight_multiplier = class_weights_[targets_labels_].unsqueeze(-1)
        # expectation
        expected_weighted_loss = (vanilla_loss * weight_multiplier).mean()

        assert_approximate_equality(
            weighted_loss.item(), expected_weighted_loss.item()
        )

    def test_ce_temporal_loss_base(self, spike_predicted_, targets_labels_):
        loss_fn = sf.ce_temporal_loss()

        assert loss_fn.weight is None
        assert loss_fn.reduction == "mean"

    def test_ce_temporal_loss_unreduced(
        self, spike_predicted_, targets_labels_
    ):
        unreduced_loss_fn = sf.ce_temporal_loss(reduction="none")
        unreduced_loss = unreduced_loss_fn(spike_predicted_, targets_labels_)

        reduced_loss_fn = sf.ce_temporal_loss()
        reduced_loss = reduced_loss_fn(spike_predicted_, targets_labels_)

        assert_approximate_equality(
            unreduced_loss.mean().item(), reduced_loss.item()
        )

    def test_ce_temporal_loss_weighted(
        self, spike_predicted_, targets_labels_, class_weights_
    ):
        weighted_loss_fn = sf.ce_temporal_loss(weight=class_weights_)
        weighted_loss = weighted_loss_fn(spike_predicted_, targets_labels_)

        # unreduced, unweighted loss
        vanilla_loss_fn = sf.ce_temporal_loss(reduction="none")
        vanilla_loss = vanilla_loss_fn(spike_predicted_, targets_labels_)
        # weight multiplier
        weight_multiplier = class_weights_[targets_labels_]
        # expectation
        expected_weighted_loss = (
            vanilla_loss * weight_multiplier
        ).sum() / weight_multiplier.sum()

        assert_approximate_equality(
            weighted_loss.item(), expected_weighted_loss.item()
        )



class TestClassWeightAxis:
    """Class weights scale every output of their own sample.

    Regression tests for #460, #461 and #462: the weighted MSE losses
    multiply a ``[batch, outputs]`` error tensor by ``weight[targets]``,
    so the weight was broadcast over the output axis whenever
    ``batch == outputs``, and raised a size mismatch otherwise.
    """

    def test_mse_count_loss_weight_is_per_sample(self):
        # T x B x C, with B == C so a transposed broadcast stays silent
        spikes = torch.tensor(
            [
                [[1.0, 1.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 0.0]],
            ]
        )
        targets = torch.tensor([0, 1])
        weights = torch.tensor([1.0, 3.0])
        loss_fn = sf.mse_count_loss(
            correct_rate=1, incorrect_rate=0, weight=weights
        )

        desired_counts = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        per_output_error = (spikes.sum(0) - desired_counts).square()
        expected = (per_output_error * weights[targets, None]).mean() / 2

        assert_approximate_equality(
            loss_fn(spikes, targets).item(), expected.item()
        )
        # independent samples must not interact through the weights
        swapped_loss = loss_fn(spikes[:, [1, 0]], targets[[1, 0]])
        assert_approximate_equality(swapped_loss.item(), expected.item())

    def test_mse_membrane_loss_weight_is_per_sample(self):
        membrane = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])  # T x B x C
        targets = torch.tensor([0, 1])
        weights = torch.tensor([1.0, 3.0])
        loss_fn = sf.mse_membrane_loss(weight=weights)

        desired = torch.eye(2)[targets]
        expected = (
            (membrane[0] - desired).square() * weights[targets, None]
        ).mean()

        assert_approximate_equality(
            loss_fn(membrane, targets).item(), expected.item()
        )
        swapped_loss = loss_fn(membrane[:, [1, 0]], targets[[1, 0]])
        assert_approximate_equality(swapped_loss.item(), expected.item())

    def test_mse_temporal_loss_weight_is_per_sample(self):
        spikes = torch.zeros((4, 2, 2))
        spikes[0, 0, 0] = 1
        spikes[1, 0, 1] = 1
        spikes[2, 1, 0] = 1
        spikes[0, 1, 1] = 1
        targets = torch.tensor([0, 1])
        weights = torch.tensor([1.0, 3.0])
        loss_fn = sf.mse_temporal_loss(
            on_target=0,
            off_target=3,
            tolerance=0,
            multi_spike=False,
            weight=weights,
        )

        observed = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
        desired = torch.tensor([[0.0, 3.0], [3.0, 0.0]])
        expected = (
            ((observed - desired) / 4).square() * weights[targets, None]
        ).mean()

        assert_approximate_equality(
            loss_fn(spikes, targets).item(), expected.item()
        )
        swapped_loss = loss_fn(spikes[:, [1, 0]], targets[[1, 0]])
        assert_approximate_equality(swapped_loss.item(), expected.item())

    def test_weighted_mse_losses_accept_non_square_shapes(self):
        # B=2 and C=3: the old broadcast raised a size mismatch here
        spikes = torch.zeros((2, 2, 3))
        spikes[0, 0, 0] = 1
        spikes[0, 1, 2] = 1
        targets = torch.tensor([0, 1])
        weights = torch.tensor([1.0, 3.0, 2.0])

        for loss_fn in (
            sf.mse_count_loss(weight=weights),
            sf.mse_membrane_loss(weight=weights),
            sf.mse_temporal_loss(weight=weights),
        ):
            assert torch.isfinite(loss_fn(spikes, targets))

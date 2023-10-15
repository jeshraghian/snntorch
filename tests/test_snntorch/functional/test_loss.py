#!/usr/bin/env python

"""Tests for Loss."""

import pytest
import snntorch as snn
import snntorch.functional as sf
import torch
import math

import snntorch.spikegen as spikegen

torch.manual_seed(42)
tolerance = 1e-3

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


class TestLoss:
    def test_ce_rate_loss_base(self, spike_predicted_, targets_labels_):
        loss_fn = sf.ce_rate_loss()
        loss = loss_fn(spike_predicted_, targets_labels_)

        assert math.isclose(loss.item(), 1.1099, rel_tol=tolerance)

    def test_ce_rate_loss_unreduced(self, spike_predicted_, targets_labels_):
        unreduced_loss_fn = sf.ce_rate_loss(reduction='none')
        unreduced_loss = unreduced_loss_fn(spike_predicted_, targets_labels_)

        reduced_loss_fn = sf.ce_rate_loss()
        reduced_loss = reduced_loss_fn(spike_predicted_, targets_labels_)

        assert math.isclose(unreduced_loss.mean().item(), reduced_loss.item(), rel_tol=tolerance)

    def test_ce_count_loss_base(self, spike_predicted_, targets_labels_):
        loss_fn = sf.ce_count_loss()
        loss = loss_fn(spike_predicted_, targets_labels_)

        assert math.isclose(loss.item(), 1.1944, rel_tol=tolerance)

    def test_ce_count_loss_unreduced(self, spike_predicted_, targets_labels_):
        unreduced_loss_fn = sf.ce_count_loss(reduction='none')
        unreduced_loss = unreduced_loss_fn(spike_predicted_, targets_labels_)

        reduced_loss_fn = sf.ce_count_loss()
        reduced_loss = reduced_loss_fn(spike_predicted_, targets_labels_)

        assert math.isclose(unreduced_loss.mean().item(), reduced_loss.item(), rel_tol=tolerance)

    def test_ce_max_membrane_loss_base(self, membrane_predicted_, targets_labels_):
        loss_fn = sf.ce_max_membrane_loss()
        loss = loss_fn(membrane_predicted_, targets_labels_)

        assert math.isclose(loss.item(), 1.0639, rel_tol=1e-4)

    def test_ce_max_membrane_loss_unreduced(self, membrane_predicted_, targets_labels_):
        unreduced_loss_fn = sf.ce_max_membrane_loss(reduction='none')
        unreduced_loss = unreduced_loss_fn(membrane_predicted_, targets_labels_)

        reduced_loss_fn = sf.ce_max_membrane_loss()
        reduced_loss = reduced_loss_fn(membrane_predicted_, targets_labels_)

        assert math.isclose(unreduced_loss.mean().item(), reduced_loss.item(), rel_tol=tolerance)

    def test_mse_count_loss_base(self, spike_predicted_, targets_labels_):
        loss_fn = sf.mse_count_loss()
        loss = loss_fn(spike_predicted_, targets_labels_)

        assert math.isclose(loss.item(), 0.8148, rel_tol=tolerance)

    def test_mse_count_loss_unreduced(self, spike_predicted_, targets_labels_):
        unreduced_loss_fn = sf.mse_count_loss(reduction='none')
        unreduced_loss = unreduced_loss_fn(spike_predicted_, targets_labels_)

        reduced_loss_fn = sf.mse_count_loss()
        reduced_loss = reduced_loss_fn(spike_predicted_, targets_labels_)

        assert math.isclose(unreduced_loss.mean().item(), reduced_loss.item(), rel_tol=tolerance)

    def test_mse_membrane_loss_base(self, membrane_predicted_, targets_labels_):
        loss_fn = sf.mse_membrane_loss()
        loss = loss_fn(membrane_predicted_, targets_labels_)

        assert math.isclose(loss.item(), 0.3214, rel_tol=tolerance)

    def test_mse_membrane_loss_unreduced(self, membrane_predicted_, targets_labels_):
        unreduced_loss_fn = sf.mse_membrane_loss(reduction='none')
        unreduced_loss = unreduced_loss_fn(membrane_predicted_, targets_labels_)

        reduced_loss_fn = sf.mse_membrane_loss()
        reduced_loss = reduced_loss_fn(membrane_predicted_, targets_labels_)

        assert math.isclose(unreduced_loss.mean().item(), reduced_loss.item(), rel_tol=tolerance)

    def test_mse_temporal_loss_base(self, spike_predicted_, targets_labels_):
        loss_fn = sf.mse_temporal_loss(on_target=1, off_target=0)
        loss = loss_fn(spike_predicted_, targets_labels_)

        assert math.isclose(loss.item(), 0.22222, rel_tol=1e-4)

    def test_mse_temporal_loss_unreduced(self, spike_predicted_, targets_labels_):
        unreduced_loss_fn = sf.mse_temporal_loss(reduction='none')
        unreduced_loss = unreduced_loss_fn(spike_predicted_, targets_labels_)

        reduced_loss_fn = sf.mse_temporal_loss()
        reduced_loss = reduced_loss_fn(spike_predicted_, targets_labels_)

        assert math.isclose(unreduced_loss.mean().item(), reduced_loss.item(), rel_tol=tolerance)

    def test_cse_temporal_loss_base(self, spike_predicted_, targets_labels_):
        loss_fn = sf.ce_temporal_loss()
        loss = loss_fn(spike_predicted_, targets_labels_)

        assert math.isclose(loss.item(), 0.8364, rel_tol=1e-4)

    def test_cse_temporal_loss_unreduced(self, spike_predicted_, targets_labels_):
        unreduced_loss_fn = sf.ce_temporal_loss(reduction='none')
        unreduced_loss = unreduced_loss_fn(spike_predicted_, targets_labels_)

        reduced_loss_fn = sf.ce_temporal_loss()
        reduced_loss = reduced_loss_fn(spike_predicted_, targets_labels_)

        assert math.isclose(unreduced_loss.mean().item(), reduced_loss.item(), rel_tol=tolerance)

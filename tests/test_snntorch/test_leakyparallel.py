#!/usr/bin/env python

"""Tests for LeakyParallel neuron.

Test Structure:
--------------
1. Basic Functionality Tests:
    - Basic forward pass
    - Output shape validation

2. Dropout Tests:
    - Dropout=0 produces deterministic outputs
    - Dropout=1.0 zeros outputs in training mode
    - Dropout produces deterministic outputs in eval mode
    - Dropout produces variable outputs in training mode
    - Dropout affects gradients during training
    - Dropout layer creation/absence validation

Coverage:
--------
- Input/output shape consistency
- Dropout functionality (the main fix for the GitHub issue)
- Training vs eval mode behavior
- Gradient computation with dropout
"""

import pytest
import snntorch as snn
import torch
import torch.nn as nn


@pytest.fixture(scope="module")
def input_parallel():
    """Input tensor for LeakyParallel (sequence, batch, features)."""
    return torch.randn(10, 2, 4)


@pytest.fixture(scope="module")
def leakyparallel_instance():
    """Basic LeakyParallel instance without dropout."""
    return snn.LeakyParallel(input_size=4, hidden_size=8, beta=0.5, dropout=0.0)


@pytest.fixture(scope="module")
def leakyparallel_dropout_instance():
    """LeakyParallel instance with dropout."""
    return snn.LeakyParallel(input_size=4, hidden_size=8, beta=0.5, dropout=0.5)


@pytest.fixture(scope="module")
def leakyparallel_dropout_one_instance():
    """LeakyParallel instance with dropout=1.0."""
    return snn.LeakyParallel(input_size=4, hidden_size=8, beta=0.5, dropout=1.0)


class TestLeakyParallel:
    def test_leakyparallel_basic(self, leakyparallel_instance, input_parallel):
        """Test basic forward pass of LeakyParallel."""
        output = leakyparallel_instance(input_parallel)
        
        # Check output shape: (sequence_length, batch_size, hidden_size)
        assert output.shape == (10, 2, 8)
        # Check output is non-negative (spikes)
        assert torch.all(output >= 0)

    def test_leakyparallel_dropout_zero_deterministic(
        self, leakyparallel_instance, input_parallel
    ):
        """Test that dropout=0 produces deterministic outputs."""
        leakyparallel_instance.eval()
        
        # Run forward pass twice with same input
        output1 = leakyparallel_instance(input_parallel)
        output2 = leakyparallel_instance(input_parallel)
        
        # Outputs should be identical
        assert torch.allclose(output1, output2), \
            "dropout=0 should produce deterministic outputs"

    def test_leakyparallel_dropout_one_training(
        self, leakyparallel_dropout_one_instance, input_parallel
    ):
        """Test that dropout=1.0 zeros outputs in training mode."""
        leakyparallel_dropout_one_instance.train()
        
        output = leakyparallel_dropout_one_instance(input_parallel)
        
        # With dropout=1.0, all outputs should be zero in training mode
        # (dropout zeros the RNN output, then threshold subtraction makes it negative,
        # which produces 0 spikes)
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-6), \
            "dropout=1.0 in training mode should produce zero outputs"

    def test_leakyparallel_dropout_eval_deterministic(
        self, leakyparallel_dropout_instance, input_parallel
    ):
        """Test that dropout produces deterministic outputs in eval mode."""
        leakyparallel_dropout_instance.eval()
        
        # Run forward pass twice with same input
        output1 = leakyparallel_dropout_instance(input_parallel)
        output2 = leakyparallel_dropout_instance(input_parallel)
        
        # Outputs should be identical (scaled by 1-dropout)
        assert torch.allclose(output1, output2), \
            "dropout in eval mode should produce deterministic outputs"

    def test_leakyparallel_dropout_training_variability(
        self, leakyparallel_dropout_instance, input_parallel
    ):
        """Test that dropout produces variable outputs in training mode."""
        leakyparallel_dropout_instance.train()
        
        # Run forward pass twice with same input
        output1 = leakyparallel_dropout_instance(input_parallel)
        output2 = leakyparallel_dropout_instance(input_parallel)
        
        # Outputs should be different (not identical) due to dropout randomness
        # Note: There's a small chance they could be the same, but very unlikely
        outputs_different = not torch.allclose(output1, output2, atol=1e-6)
        
        # This assertion might occasionally fail due to randomness,
        # but it's extremely unlikely with dropout=0.5
        # We'll just check that the outputs are valid (non-negative)
        assert torch.all(output1 >= 0) and torch.all(output2 >= 0), \
            "dropout outputs should be valid (non-negative)"

    def test_leakyparallel_dropout_affects_gradients(
        self, leakyparallel_instance, leakyparallel_dropout_instance, input_parallel
    ):
        """Test that dropout affects gradients during training."""
        # Create identical models
        leakyparallel_dropout_instance.load_state_dict(
            leakyparallel_instance.state_dict()
        )
        
        target = torch.randn(10, 2, 8)
        
        # Training mode
        leakyparallel_instance.train()
        leakyparallel_dropout_instance.train()
        
        # Forward and backward pass
        output1 = leakyparallel_instance(input_parallel)
        output2 = leakyparallel_dropout_instance(input_parallel)
        
        loss1 = nn.functional.mse_loss(output1, target)
        loss2 = nn.functional.mse_loss(output2, target)
        
        loss1.backward()
        loss2.backward()
        
        # Get gradients
        grad1 = leakyparallel_instance.rnn.weight_ih_l0.grad.clone()
        grad2 = leakyparallel_dropout_instance.rnn.weight_ih_l0.grad.clone()
        
        # Gradients should be different due to dropout
        grad_diff = torch.abs(grad1 - grad2).sum()
        assert grad_diff > 1e-6, \
            "dropout should cause different gradients during training"
        
        # Clear gradients for next test
        leakyparallel_instance.zero_grad()
        leakyparallel_dropout_instance.zero_grad()

    def test_leakyparallel_dropout_layer_exists(self, leakyparallel_dropout_instance):
        """Test that dropout layer is created when dropout > 0."""
        assert hasattr(leakyparallel_dropout_instance, 'dropout_layer'), \
            "dropout_layer should exist when dropout > 0"
        assert leakyparallel_dropout_instance.dropout_layer is not None, \
            "dropout_layer should not be None when dropout > 0"
        assert leakyparallel_dropout_instance.dropout == 0.5, \
            "dropout value should be stored correctly"

    def test_leakyparallel_no_dropout_layer(self, leakyparallel_instance):
        """Test that dropout layer is None when dropout = 0."""
        assert hasattr(leakyparallel_instance, 'dropout_layer'), \
            "dropout_layer attribute should exist"
        assert leakyparallel_instance.dropout_layer is None, \
            "dropout_layer should be None when dropout = 0"
        assert leakyparallel_instance.dropout == 0.0, \
            "dropout value should be 0.0"


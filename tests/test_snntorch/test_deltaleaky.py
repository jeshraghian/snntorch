#!/usr/bin/env python

"""Tests for the DeltaLeaky neuron."""

import pytest
from snntorch._neurons.deltaleaky import DeltaLeaky
import torch


@pytest.fixture(scope="module")
def step_input():
    """Input with step changes"""
    return torch.Tensor([1.0, 0.1, 2.0, 0.1]).unsqueeze(-1)


@pytest.fixture(scope="module")
def delta_low():
    return DeltaLeaky(beta=0.5, delta_threshold=0.1, init_hidden=True)


@pytest.fixture(scope="module")
def delta_high():
    return DeltaLeaky(beta=0.5, delta_threshold=1.0, init_hidden=True)


class TestDeltaLeakyCore:
    """Core tests for the DeltaLeaky neuron"""

    def test_spike_condition(self, step_input):
        """Verify the neuron actually generates spikes"""
        delta = DeltaLeaky(beta=0.5, delta_threshold=0.2, init_hidden=True)

        #reset internal states
        delta.mem = torch.zeros_like(step_input[0])
        delta.mem_prev = torch.zeros_like(step_input[0])

        spikes = [delta(step_input[i])[0] for i in range(len(step_input))]
        assert sum(spikes) > 0, "Spikes were not generated"


    def test_threshold_sensitivity(self, delta_low, delta_high, step_input):
        """Prove that higher threshold = fewer spikes (core delta property)"""
        def count_spikes(neuron):
            neuron.mem = torch.zeros_like(step_input[0])
            neuron.mem_prev = torch.zeros_like(step_input[0])
            return sum([neuron(step_input[i])[0] for i in range(len(step_input))])

        low_spikes = count_spikes(delta_low)
        high_spikes = count_spikes(delta_high)

        assert low_spikes >= high_spikes, "Higher threshold produced more spikes, should be less"

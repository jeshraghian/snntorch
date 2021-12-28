#!/usr/bin/env python

"""Tests for `snntorch` package."""

import pytest
import snntorch as snn
import torch


@pytest.fixture(scope="module")
def input_():
    return torch.Tensor([0.25, 0]).unsqueeze(-1)


def test_fire():
    synaptic = snn.Synaptic(alpha=0.5, beta=0.5)
    input_large = torch.Tensor([synaptic.threshold * 10])
    assert synaptic.fire(input_large)[0] == 1


def test_instances():
    snn.SpikingNeuron.instances = []
    snn.Synaptic(alpha=0.5, beta=0.5)
    snn.Alpha(alpha=0.5, beta=0.4)
    assert len(snn.SpikingNeuron.instances) == 2

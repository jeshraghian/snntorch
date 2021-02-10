"""Tests for `snntorch.spikegen` module."""


import pytest
import snntorch as snn
from snntorch import spikegen
import torch


def input_(a):
    return torch.Tensor([a])


@pytest.mark.parametrize("test_input, expected", [(input_(0), 0), (input_(1), 1)])
def test_rate(test_input, expected):
    assert spikegen.rate(test_input) == expected


@pytest.mark.parametrize("test_input, expected", [(input_(0), 0), (input_(1), 1)])
def test_latency(test_input, expected):
    assert spikegen.latency(test_input) == expected

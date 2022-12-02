#!/usr/bin/env python

"""Tests for graded spikes."""

import pytest
import snntorch as snn
import torch


@pytest.fixture(scope="module")
def input_():
    return torch.Tensor([0.0, 1.0, 2.0]).unsqueeze(-1)


@pytest.fixture(scope="module")
def graded_spikes_instance():
    return snn.GradedSpikes(size=3, constant_factor=None)


@pytest.fixture(scope="module")
def graded_spikes_constant_factor_one_instance():
    return snn.GradedSpikes(size=3, constant_factor=1.0)


@pytest.fixture(scope="module")
def graded_spikes_constant_factor_two_instance():
    return snn.GradedSpikes(size=3, constant_factor=2.0)


class TestLeaky:
    def test_graded_spikes(self, graded_spikes_instance, input_):
        out = graded_spikes_instance(input_)
        assert False

    def test_graded_spikes_constant_factor_one(
        self, graded_spikes_constant_factor_one_instance, input_
    ):
        out = graded_spikes_constant_factor_one_instance(input_)
        assert torch.all(torch.eq(input=input_, other=out))

#!/usr/bin/env python

"""Tests for graded spikes."""

import pytest
import snntorch as snn
import torch


@pytest.fixture(scope="module")
def input_():
    return torch.Tensor([0.0, 1.0, 2.0]).unsqueeze(-1)


@pytest.fixture(scope="module")
def label_():
    return torch.Tensor([0.1, 1.1, 2.1]).unsqueeze(-1)


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
        true = graded_spikes_instance.weights * input_
        out = graded_spikes_instance(input_)

        assert torch.all(true == out)

    def test_graded_spikes_learning(
        self, graded_spikes_instance, input_, label_
    ):

        optimizer = torch.optim.AdamW(
            graded_spikes_instance.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.01,
        )
        loss_function = torch.nn.MSELoss()
        out = graded_spikes_instance(input_)
        loss_start = loss_function(out, label_)

        for i in range(100):
            out = graded_spikes_instance(input_)
            loss = loss_function(out, label_)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert loss < loss_start

    def test_graded_spikes_shape(self, graded_spikes_instance, input_):
        shapes_true = input_.shape
        shapes_out = graded_spikes_instance(input_).shape

        assert shapes_true == shapes_out

    def test_graded_spikes_constant_shape(
        self, graded_spikes_constant_factor_one_instance, input_
    ):
        shapes_true = input_.shape
        shapes_out = graded_spikes_constant_factor_one_instance(input_).shape

        assert shapes_true == shapes_out

    def test_graded_spikes_constant_factor_one(
        self, graded_spikes_constant_factor_one_instance, input_
    ):
        out = graded_spikes_constant_factor_one_instance(input_)
        assert torch.all(torch.eq(input=input_, other=out))

    def test_graded_spikes_constant_factor_two(
        self, graded_spikes_constant_factor_two_instance, input_
    ):
        out = graded_spikes_constant_factor_two_instance(input_)
        assert torch.all(torch.eq(input=input_, other=0.5 * out))

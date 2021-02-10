#!/usr/bin/env python

"""Tests for `snntorch` package."""

import pytest
import snntorch as snn
import snntorch.backprop as bp
from tests.conftest import Net
from unittest import mock
import torch


def test_TBPTT():
    mock_criterion = mock.Mock(return_value=torch.ones([1], requires_grad=True))
    mock_optimizer = mock.Mock(**{"zero_grad.return_value": 1, "step.return_value": 2})
    criterion = mock_criterion
    optimizer = mock_optimizer
    net = Net()
    data = torch.Tensor([1])

    loss_avg = bp.TBPTT(
        net=net,
        data=data,
        target=data,
        num_steps=1,
        batch_size=1,
        optimizer=optimizer,
        criterion=criterion,
        K=1,
    )

    assert loss_avg is not None

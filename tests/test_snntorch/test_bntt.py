#!/usr/bin/env python

"""Tests for Batch Normalization Through Time."""

import pytest
import snntorch as snn
import torch


# @pytest.fixture(scope="module")
# @pytest.mark.parametrize("value", params=[1, 2, 3])
# def time_steps_(value):
#     return value
#
#
# @pytest.fixture(scope="module")
# def input_():
#     # 2 time_steps, 2 batch_size, 2 features
#     return torch.rand(2, 2, 2)
#
#
@pytest.fixture(scope="module")
def batchnormtt1d_instance():
    return snn.BatchNormTT1d(2, time_steps_)


class TestBatchNormTT1d:
    @pytest.mark.parametrize("input_features, time_steps", ([1, 1], [2, 3], [3, 6]))
    def test_batchnormtt1d(
        self,
        input_features,
        time_steps
    ):
        batchnormtt1d_instance = snn.BatchNormTT1d(input_features, time_steps)
        assert len(batchnormtt1d_instance) == time_steps
        for module in batchnormtt1d_instance:
            assert module.num_features == input_features
            assert module.eps == 1e-5
            assert module.momentum == 0.1
            assert module.affine



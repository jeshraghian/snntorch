#!/usr/bin/env python

"""Tests for Batch Normalization Through Time."""

import pytest
import snntorch as snn
import torch


@pytest.fixture(scope="module")
def input2d_():
    # 2 time_steps, 2, batch size, 4 features
    return torch.rand(2, 2, 4)


@pytest.fixture(scope="module")
def input3d_():
    # 2 time_steps, 2, batch size, 4 features, 5 sequence length
    return torch.rand(2, 2, 4, 5)


@pytest.fixture(scope="module")
def input4d_():
    # 2 time_steps, 2, batch size, 4 features, height 2, width 2
    return torch.rand(2, 2, 4, 2, 2)


@pytest.fixture(scope="module")
def batchnormtt1d_instance():
    return snn.BatchNormTT1d(4, 2)


class TestBatchNormTT1d:
    @pytest.mark.parametrize(
        "time_steps, num_features", ([1, 1], [3, 2], [6, 3])
    )
    def test_batchnormtt1d_init(self, time_steps, num_features):
        batchnormtt1d_instance = snn.BatchNormTT1d(num_features, time_steps)

        assert len(batchnormtt1d_instance) == time_steps
        for module in batchnormtt1d_instance:
            assert module.num_features == num_features
            assert module.eps == 1e-5
            assert module.momentum == 0.1
            assert module.affine
            assert module.bias is None

    def test_batchnormtt1d_with_2d_input(
        self, batchnormtt1d_instance, input2d_
    ):
        for step, batchnormtt1d_module in enumerate(batchnormtt1d_instance):
            out = batchnormtt1d_module(input2d_[step])

            assert out.shape == input2d_[step].shape

    def test_batchnormtt1d_with_3d_input(
        self, batchnormtt1d_instance, input3d_
    ):
        for step, batchnormtt1d_module in enumerate(batchnormtt1d_instance):
            out = batchnormtt1d_module(input3d_[step])

            assert out.shape == input3d_[step].shape


@pytest.fixture(scope="module")
def batchnormtt2d_instance():
    return snn.BatchNormTT2d(4, 2)


class TestBatchNormTT2d:
    @pytest.mark.parametrize(
        "time_steps, num_features", ([1, 1], [3, 2], [6, 3])
    )
    def test_batchnormtt2d_init(self, time_steps, num_features):
        batchnormtt2d_instance = snn.BatchNormTT2d(num_features, time_steps)

        assert len(batchnormtt2d_instance) == time_steps
        for module in batchnormtt2d_instance:
            assert module.num_features == num_features
            assert module.eps == 1e-5
            assert module.momentum == 0.1
            assert module.affine
            assert module.bias is None

    def test_batchnormtt2d_with_4d_input(
        self, batchnormtt2d_instance, input4d_
    ):
        for step, batchnormtt2d_module in enumerate(batchnormtt2d_instance):
            out = batchnormtt2d_module(input4d_[step])

            assert out.shape == input4d_[step].shape


@pytest.mark.parametrize(
    "make, shape",
    ([snn.BatchNormTT1d, (3, 4)], [snn.BatchNormTT2d, (3, 4, 5, 5)]),
)
class TestBatchNormTTCall:
    def test_call_with_step_matches_indexing(self, make, shape):
        torch.manual_seed(0)
        bntt = make(4, 2)
        x = torch.rand(*shape)
        for step in range(2):
            expected = bntt[step](x)
            torch.testing.assert_close(bntt(x, step), expected)

    def test_call_without_step_explains_usage(self, make, shape):
        bntt = make(4, 2)
        with pytest.raises(TypeError, match="step"):
            bntt(torch.rand(*shape))

    def test_still_a_modulelist_with_same_state_dict(self, make, shape):
        bntt = make(4, 2)
        assert isinstance(bntt, torch.nn.ModuleList)
        assert "0.weight" in bntt.state_dict()
        assert "1.running_mean" in bntt.state_dict()
        assert "0.bias" not in bntt.state_dict()

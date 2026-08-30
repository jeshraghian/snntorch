#!/usr/bin/env python

"""Tests for dtype preservation across neuron modules and surrogate
gradients (issues #421 and #422)."""

import pytest
import snntorch as snn
from snntorch import surrogate
import torch

DTYPES = [torch.float32, torch.float64]


def _dtype_input(dtype):
    return torch.randn(2, 4, dtype=dtype)


class TestNeuronDtypePreservation:
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_leaky_dtype(self, dtype):
        x = _dtype_input(dtype)
        lif = snn.Leaky(beta=0.9).to(dtype)
        spk, mem = lif(x)
        assert spk.dtype == dtype
        assert mem.dtype == dtype

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_synaptic_dtype(self, dtype):
        x = _dtype_input(dtype)
        lif = snn.Synaptic(alpha=0.8, beta=0.9).to(dtype)
        spk, syn, mem = lif(x)
        assert spk.dtype == dtype
        assert syn.dtype == dtype
        assert mem.dtype == dtype

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_alpha_dtype(self, dtype):
        x = _dtype_input(dtype)
        lif = snn.Alpha(alpha=0.9, beta=0.8).to(dtype)
        spk, syn_exc, syn_inh, mem = lif(x)
        assert spk.dtype == dtype
        assert mem.dtype == dtype

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_lapicque_dtype(self, dtype):
        x = _dtype_input(dtype)
        lif = snn.Lapicque(beta=0.9).to(dtype)
        spk, mem = lif(x)
        assert spk.dtype == dtype
        assert mem.dtype == dtype

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_rleaky_dtype(self, dtype):
        # issue #421: second step fed the float32 spike back into the
        # recurrent layer and crashed under float64
        x = _dtype_input(dtype)
        lif = snn.RLeaky(beta=0.9, linear_features=4).to(dtype)
        spk, mem = lif(x)
        spk, mem = lif(x, spk, mem)
        assert spk.dtype == dtype
        assert mem.dtype == dtype

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_rsynaptic_dtype(self, dtype):
        x = _dtype_input(dtype)
        lif = snn.RSynaptic(alpha=0.8, beta=0.9, linear_features=4).to(dtype)
        spk, syn, mem = lif(x)
        spk, syn, mem = lif(x, spk, syn, mem)
        assert spk.dtype == dtype
        assert syn.dtype == dtype
        assert mem.dtype == dtype

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_leaky_parallel_dtype(self, dtype):
        x = torch.randn(3, 2, 4, dtype=dtype)
        lif = snn.LeakyParallel(input_size=4, hidden_size=4).to(dtype)
        spk = lif(x)
        assert spk.dtype == dtype

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_slstm_dtype(self, dtype):
        x = _dtype_input(dtype)
        lif = snn.SLSTM(input_size=4, hidden_size=4).to(dtype)
        spk, syn, mem = lif(x)
        spk, syn, mem = lif(x, syn, mem)
        assert spk.dtype == dtype
        assert syn.dtype == dtype
        assert mem.dtype == dtype

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_sconv2dlstm_dtype(self, dtype):
        x = torch.randn(2, 3, 8, 8, dtype=dtype)
        lif = snn.SConv2dLSTM(in_channels=3, out_channels=5, kernel_size=3).to(
            dtype
        )
        spk, syn, mem = lif(x)
        assert spk.dtype == dtype
        assert syn.dtype == dtype
        assert mem.dtype == dtype

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_surrogate_disable_dtype(self, dtype):
        x = _dtype_input(dtype)
        lif = snn.Leaky(beta=0.9, surrogate_disable=True).to(dtype)
        spk, mem = lif(x)
        assert spk.dtype == dtype
        assert mem.dtype == dtype


class TestSurrogateDtypePreservation:
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize(
        "spike_grad",
        [
            surrogate.atan(),
            surrogate.fast_sigmoid(),
            surrogate.sigmoid(),
            surrogate.triangular(),
            surrogate.straight_through_estimator(),
            surrogate.heaviside(),
            surrogate.spike_rate_escape(),
            surrogate.SSO(),
            surrogate.SFS(),
        ],
    )
    def test_surrogate_forward_backward_dtype(self, spike_grad, dtype):
        x = torch.randn(5, dtype=dtype, requires_grad=True)
        spk = spike_grad(x)
        spk.sum().backward()
        assert spk.dtype == dtype
        assert x.grad.dtype == dtype

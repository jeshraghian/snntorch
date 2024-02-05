#!/usr/bin/env python

"""Tests for NIR import and export."""

import nir
import pytest
import snntorch as snn
import torch


@pytest.fixture(scope="module")
def sample_data():
    return torch.ones((4, 784))


@pytest.fixture(scope="module")
def snntorch_sequential():
    lif1 = snn.Leaky(beta=0.9, init_hidden=True)
    lif2 = snn.Leaky(beta=0.9, init_hidden=True, output=True)

    return torch.nn.Sequential(
        torch.nn.Linear(784, 500),
        lif1,
        torch.nn.Linear(500, 10),
        lif2,
    )


@pytest.fixture(scope="module")
def snntorch_recurrent():
    v = torch.ones((500,))
    lif1 = snn.RSynaptic(
        alpha=0.5, beta=0.9, V=v, all_to_all=False, init_hidden=True
    )
    lif2 = snn.Leaky(beta=0.9, init_hidden=True, output=True)

    return torch.nn.Sequential(
        torch.nn.Linear(784, 500),
        lif1,
        torch.nn.Linear(500, 10),
        lif2,
    )


class TestNIR:
    """Test import and export from snnTorch to NIR."""

    def test_export_sequential(self, snntorch_sequential, sample_data):
        nir_graph = snn.export_to_nir(snntorch_sequential, sample_data)
        assert nir_graph is not None
        assert set(nir_graph.nodes.keys()) == set(
            ["input", "output"] + [str(i) for i in range(4)]
        ), nir_graph.nodes.keys()
        assert set(nir_graph.edges) == set(
            [
                ("3", "output"),
                ("input", "0"),
                ("2", "3"),
                ("1", "2"),
                ("0", "1"),
            ]
        )
        assert isinstance(nir_graph.nodes["input"], nir.Input)
        assert isinstance(nir_graph.nodes["output"], nir.Output)
        assert isinstance(nir_graph.nodes["0"], nir.Affine)
        assert isinstance(nir_graph.nodes["1"], nir.LIF)
        assert isinstance(nir_graph.nodes["2"], nir.Affine)
        assert isinstance(nir_graph.nodes["3"], nir.LIF)

    def test_export_recurrent(self, snntorch_recurrent, sample_data):
        nir_graph = snn.export_to_nir(snntorch_recurrent, sample_data)
        assert nir_graph is not None
        assert set(nir_graph.nodes.keys()) == set(
            ["input", "output", "0", "1.lif", "1.w_rec", "2", "3"]
        ), nir_graph.nodes.keys()
        assert isinstance(nir_graph.nodes["input"], nir.Input)
        assert isinstance(nir_graph.nodes["output"], nir.Output)
        assert isinstance(nir_graph.nodes["0"], nir.Affine)
        assert isinstance(nir_graph.nodes["1.lif"], nir.CubaLIF)
        assert isinstance(nir_graph.nodes["1.w_rec"], nir.Linear)
        assert isinstance(nir_graph.nodes["2"], nir.Affine)
        assert isinstance(nir_graph.nodes["3"], nir.LIF)
        assert set(nir_graph.edges) == set(
            [
                ("1.lif", "1.w_rec"),
                ("1.w_rec", "1.lif"),
                ("0", "1.lif"),
                ("3", "output"),
                ("2", "3"),
                ("input", "0"),
                ("1.lif", "2"),
            ]
        )

    def test_import_nir(self):
        graph = nir.read("tests/lif.nir")
        net = snn.import_from_nir(graph)
        out, _ = net(torch.ones(1, 1))
        assert out.shape == (1, 1), out.shape

    def test_commute_sequential(self, snntorch_sequential, sample_data):
        x = torch.rand((4, 784))
        y_snn, state = snntorch_sequential(x)
        assert y_snn.shape == (4, 10)
        nir_graph = snn.export_to_nir(snntorch_sequential, sample_data)
        net = snn.import_from_nir(nir_graph)
        y_nir, state = net(x)
        assert y_nir.shape == (4, 10), y_nir.shape
        assert torch.allclose(y_snn, y_nir)

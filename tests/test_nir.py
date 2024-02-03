#!/usr/bin/env python

"""Tests for NIR import and export."""

import pytest
import snntorch as snn
import torch


@pytest.fixture(scope="module")
def snntorch_sequential():
    lif1 = snn.Leaky(beta=0.9, init_hidden=False)
    lif2 = snn.Leaky(beta=0.9, init_hidden=False, output=True)

    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(784, 500),
        lif1,
        torch.nn.Linear(500, 10),
        lif2
    )


@pytest.fixture(scope="module")
def snntorch_sequential_hidden():
    lif1 = snn.Leaky(beta=0.9, init_hidden=True)
    lif2 = snn.Leaky(beta=0.9, init_hidden=True, output=True)

    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(784, 500),
        lif1,
        torch.nn.Linear(500, 10),
        lif2
    )


@pytest.fixture(scope="module")
def snntorch_recurrent():
    lif1 = snn.RLeaky(beta=0.9, V=1, all_to_all=True, init_hidden=False)
    lif2 = snn.Leaky(beta=0.9, init_hidden=False, output=True)

    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(784, 500),
        lif1,
        torch.nn.Linear(500, 10),
        lif2
    )


@pytest.fixture(scope="module")
def snntorch_recurrent_hidden():
    lif1 = snn.RLeaky(beta=0.9, V=1, all_to_all=True, init_hidden=True)
    lif2 = snn.Leaky(beta=0.9, init_hidden=True, output=True)

    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(784, 500),
        lif1,
        torch.nn.Linear(500, 10),
        lif2
    )


class NIRTestExport:
    """Test exporting snnTorch network to NIR."""
    def test_export_sequential(snntorch_sequential):
        pass

    def test_export_sequential_hidden(snntorch_sequential_hidden):
        pass

    def test_export_recurrent(snntorch_recurrent):
        pass

    def test_export_recurrent_hidden(snntorch_recurrent_hidden):
        pass


class NIRTestImport:
    """Test importing NIR graph to snnTorch."""
    def test_import_nir():
        # load a NIR graph from a file?
        pass


class NIRTestCommute:
    """Test that snnTorch -> NIR -> snnTorch doesn't change the network."""
    def test_commute_sequential(snntorch_sequential):
        pass

    def test_commute_sequential_hidden(snntorch_sequential_hidden):
        pass

    def test_commute_recurrent(snntorch_recurrent):
        pass

    def test_commute_recurrent_hidden(snntorch_recurrent_hidden):
        pass

#!/usr/bin/env python

"""Tests for NIR import and export."""

import nir
import numpy as np
import pytest
import snntorch as snn
from snntorch.export_nir import export_to_nir
from snntorch.import_nir import import_from_nir
import torch


# sample data for snntorch_sequential
@pytest.fixture(scope="module")
def sample_data():
    return torch.ones((4, 784))


# sample data for snntorch with conv2d_avgpool
@pytest.fixture(scope="module")
def sample_data2():
    return torch.randn(1, 1, 28, 28)


class NetWithAvgPool(torch.nn.Module):
    def __init__(self):
        super(NetWithAvgPool, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = torch.nn.Flatten()
        self.lif1 = snn.Leaky(
            beta=0.9 * torch.ones(28 * 28 * 16 // 4),
            threshold=torch.ones(28 * 28 * 16 // 4),
            init_hidden=True,
        )
        self.fc1 = torch.nn.Linear(28 * 28 * 16 // 4, 500)
        self.lif2 = snn.Leaky(
            beta=0.9 * torch.ones(500),
            threshold=torch.ones(500),
            init_hidden=True,
            output=True,
        )

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.flatten(x)
        x = self.lif1(x)
        x = self.fc1(x)
        x = self.lif2(x)
        return x


@pytest.fixture(scope="module")
def net_with_avg_pool():
    net = NetWithAvgPool()
    return net


@pytest.fixture(scope="module")
def snntorch_sequential():
    lif1 = snn.Leaky(
        beta=0.9 * torch.ones(500), threshold=torch.ones(500), init_hidden=True
    )
    lif2 = snn.Leaky(
        beta=0.9 * torch.ones(10),
        threshold=torch.ones(10),
        init_hidden=True,
        output=True,
    )

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
        alpha=0.5 * torch.ones(500),
        beta=0.9 * torch.ones(500),
        V=v,
        all_to_all=False,
        init_hidden=True,
    )
    lif2 = snn.Leaky(
        beta=0.9 * torch.ones(10),
        threshold=torch.ones(10),
        init_hidden=True,
        output=True,
    )

    return torch.nn.Sequential(
        torch.nn.Linear(784, 500),
        lif1,
        torch.nn.Linear(500, 10),
        lif2,
    )


@pytest.fixture(scope="module")
def snntorch_rleaky():
    v = torch.ones((500,))
    lif1 = snn.RLeaky(
        beta=0.9 * torch.ones(500),
        threshold=torch.ones(500),
        V=v,
        all_to_all=False,
        init_hidden=True,
    )
    lif2 = snn.Leaky(
        beta=0.9 * torch.ones(10),
        threshold=torch.ones(10),
        init_hidden=True,
        output=True,
    )

    return torch.nn.Sequential(
        torch.nn.Linear(784, 500),
        lif1,
        torch.nn.Linear(500, 10),
        lif2,
    )


class TestNIR:
    """Test import and export from snnTorch to NIR."""

    def test_export_sequential(self, snntorch_sequential, sample_data):
        nir_graph = export_to_nir(
            snntorch_sequential, sample_data, ignore_dims=[0]
        )
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

    def test_export_NetWithAvgPool(self, net_with_avg_pool, sample_data2):
        nir_graph = export_to_nir(
            net_with_avg_pool, sample_data2, ignore_dims=[0]
        )
        assert nir_graph is not None
        assert set(nir_graph.nodes.keys()) == set(
            [
                "input",
                "output",
                "conv1",
                "pool",
                "flatten",
                "lif1",
                "fc1",
                "lif2",
            ]
        ), nir_graph.nodes.keys()
        assert isinstance(nir_graph.nodes["conv1"], nir.Conv2d)
        assert isinstance(nir_graph.nodes["pool"], nir.AvgPool2d)
        assert isinstance(nir_graph.nodes["flatten"], nir.Flatten)
        assert isinstance(nir_graph.nodes["lif1"], nir.LIF)
        assert isinstance(nir_graph.nodes["fc1"], nir.Affine)
        assert isinstance(nir_graph.nodes["lif2"], nir.LIF)
        # every edge must have consistent port types (export calls
        # infer_types, so reaching this point already proves consistency)
        assert set(nir_graph.edges) == set(
            [
                ("input", "conv1"),
                ("conv1", "pool"),
                ("pool", "flatten"),
                ("flatten", "lif1"),
                ("lif1", "fc1"),
                ("fc1", "lif2"),
                ("lif2", "output"),
            ]
        )

    def test_export_scalar_neuron_params(self, sample_data):
        """Regression test for issue #491.

        Scalar ``beta``/``threshold`` are stored as 0-dim arrays; exported
        as-is they degenerate the NIR node's port types to ``[]`` and
        ``infer_types`` rejects the graph. They must be broadcast to the
        measured layer width. Uses the exact docstring example.
        """
        lif1 = snn.Leaky(beta=0.9, init_hidden=True)
        lif2 = snn.Leaky(beta=0.9, init_hidden=True, output=True)
        net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(784, 500),
            lif1,
            torch.nn.Linear(500, 10),
            lif2,
        )
        nir_graph = export_to_nir(net, sample_data, ignore_dims=[0])

        lif1_node = nir_graph.nodes["2"]
        assert isinstance(lif1_node, nir.LIF)
        assert lif1_node.tau.shape == (500,)
        assert lif1_node.v_threshold.shape == (500,)
        assert lif1_node.input_type["input"].tolist() == [500]
        assert lif1_node.output_type["output"].tolist() == [500]
        assert np.allclose(lif1_node.v_threshold, 1.0)

        lif2_node = nir_graph.nodes["4"]
        assert isinstance(lif2_node, nir.LIF)
        assert lif2_node.tau.shape == (10,)
        assert lif2_node.input_type["input"].tolist() == [10]

    def test_export_mixed_neuron_param_shapes(self, sample_data):
        """Vector ``beta`` with scalar ``threshold`` must also broadcast.

        Previously this hit nir's "All parameters must have the same shape"
        assertion.
        """
        lif = snn.Leaky(
            beta=0.9 * torch.ones(500), threshold=1.0, init_hidden=True
        )
        net = torch.nn.Sequential(torch.nn.Linear(784, 500), lif)
        nir_graph = export_to_nir(net, sample_data, ignore_dims=[0])

        lif_node = nir_graph.nodes["1"]
        assert isinstance(lif_node, nir.LIF)
        assert lif_node.tau.shape == (500,)
        assert lif_node.v_threshold.shape == (500,)

    def test_export_scalar_synaptic_params(self, sample_data):
        """Scalar ``alpha``/``beta``/``threshold`` broadcast for CubaLIF."""
        syn = snn.Synaptic(alpha=0.5, beta=0.9, init_hidden=True)
        lif = snn.Leaky(beta=0.9, init_hidden=True, output=True)
        net = torch.nn.Sequential(
            torch.nn.Linear(784, 500), syn, torch.nn.Linear(500, 10), lif
        )
        nir_graph = export_to_nir(net, sample_data, ignore_dims=[0])

        syn_node = nir_graph.nodes["1"]
        assert isinstance(syn_node, nir.CubaLIF)
        assert syn_node.tau_mem.shape == (500,)
        assert syn_node.tau_syn.shape == (500,)
        assert syn_node.v_threshold.shape == (500,)
        assert syn_node.input_type["input"].tolist() == [500]

    def test_export_tuple_output_midnet_rejected(self, sample_data):
        """init_hidden=False makes Leaky return (spk, mem) tuples; feeding a
        tuple into the next layer must raise a clear error, not a cryptic
        TypeError from inside the tracer."""
        net = torch.nn.Sequential(
            torch.nn.Linear(784, 500),
            snn.Leaky(beta=0.9),
            torch.nn.Linear(500, 10),
        )
        with pytest.raises(TypeError, match="tuple"):
            export_to_nir(net, sample_data, ignore_dims=[0])

    def test_export_recurrent(self, snntorch_recurrent, sample_data):
        nir_graph = export_to_nir(
            snntorch_recurrent, sample_data, ignore_dims=[0]
        )
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

    def test_export_rleaky(self, snntorch_rleaky, sample_data):
        nir_graph = export_to_nir(
            snntorch_rleaky, sample_data, ignore_dims=[0]
        )
        assert nir_graph is not None
        assert set(nir_graph.nodes.keys()) == set(
            ["input", "output", "0", "1.lif", "1.w_rec", "2", "3"]
        ), nir_graph.nodes.keys()
        assert isinstance(nir_graph.nodes["input"], nir.Input)
        assert isinstance(nir_graph.nodes["output"], nir.Output)
        assert isinstance(nir_graph.nodes["0"], nir.Affine)
        assert isinstance(nir_graph.nodes["1.lif"], nir.LIF)
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
        net = import_from_nir(graph)
        out, _ = net(torch.ones(1, 1))
        if isinstance(out, tuple):
            out = out[0]
        assert out.shape == (1, 1), out.shape

    def test_import_if_node_is_ideal_integrator(self):
        """Regression test for issue #416.

        NIR defines the IF primitive as an ideal integrate-and-fire
        neuron, v[t+1] = v[t] + r * i[t], so it must be imported as a
        leak-free ``snn.Leaky`` with ``beta = 1``. Previously the import
        hardcoded a leak (``beta=0.9``, later ``beta=0``), which silently
        changed the neuron dynamics of imported models.
        """
        n = 3
        graph = nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type=np.array([n])),
                "if": nir.IF(r=np.ones(n), v_threshold=np.ones(n)),
                "output": nir.Output(output_type=np.array([n])),
            },
            edges=[("input", "if"), ("if", "output")],
        )
        net = import_from_nir(graph)

        leaky_mods = [m for m in net.modules() if isinstance(m, snn.Leaky)]
        assert len(leaky_mods) == 1, "expected exactly one imported neuron"
        lif = leaky_mods[0]

        # An ideal integrator has no leak.
        assert torch.all(lif.beta == 1.0), lif.beta

        # Sub-threshold, the membrane must accumulate linearly:
        # v[t] = t * i for constant input i (no decay between steps).
        x = 0.0625 * torch.ones(1, n)
        lif.reset_mem()
        for step in range(1, 5):
            lif(x)
            assert torch.allclose(lif.mem, step * x), (step, lif.mem)

        # A constant sub-threshold input must eventually accumulate up to
        # the threshold and emit a spike. With a leak (beta < 1), the
        # membrane would plateau below threshold and never fire.
        lif.reset_mem()
        spk_count = torch.zeros(1, n)
        for _ in range(20):
            spk_count += lif(x)
        assert torch.all(spk_count >= 1), spk_count

    def test_import_conv_nir(self):
        pytest.xfail("conv2d import unsupported")

    def test_commute_sequential(self, snntorch_sequential, sample_data):
        x = torch.rand((4, 784))
        y_snn, state = snntorch_sequential(x)
        assert y_snn.shape == (4, 10)
        nir_graph = export_to_nir(
            snntorch_sequential, sample_data, ignore_dims=[0]
        )
        net = import_from_nir(nir_graph)
        y_nir, state = net(x)
        if isinstance(y_nir, tuple):
            y_nir = y_nir[0]
        assert y_nir.shape == (4, 10), y_nir.shape
        assert torch.allclose(y_snn, y_nir)

    def test_commute_rleaky(self, snntorch_rleaky, sample_data):
        x = torch.rand((4, 784))
        y_snn, state = snntorch_rleaky(x)
        assert y_snn.shape == (4, 10)
        nir_graph = export_to_nir(
            snntorch_rleaky, sample_data, ignore_dims=[0]
        )
        net = import_from_nir(nir_graph)
        y_nir, state = net(x)
        assert y_nir.shape == (4, 10), y_nir.shape
        assert torch.allclose(y_snn, y_nir)

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
        pytest.xfail("conv2d export currently unsupported")

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


def _roundtrip(graph, path):
    nir.write(str(path), graph)
    return nir.read(str(path))


def _scalar_synaptic_net(in_features, width):
    return torch.nn.Sequential(
        torch.nn.Linear(in_features, width),
        snn.Synaptic(
            alpha=0.5,
            beta=0.9,
            threshold=1.0,
            init_hidden=True,
            output=True,
        ),
    )


def _scalar_leaky_net(in_features, width):
    return torch.nn.Sequential(
        torch.nn.Linear(in_features, width),
        snn.Leaky(beta=0.9, threshold=1.0, init_hidden=True, output=True),
    )


class TestScalarNeuronExport:
    """Regression tests for issues #410 and #334."""

    def test_issue_410_scalar_synaptic_roundtrip(self, tmp_path):
        net = _scalar_synaptic_net(2450, 128)
        graph = export_to_nir(net, torch.ones(1, 2450), ignore_dims=[0])

        neuron = graph.nodes["1"]
        assert isinstance(neuron, nir.CubaLIF)
        assert np.asarray(neuron.v_threshold).shape == (128,)
        assert np.asarray(neuron.tau_syn).shape == (128,)
        assert np.asarray(neuron.tau_mem).shape == (128,)
        assert np.asarray(neuron.r).shape == (128,)
        assert np.asarray(neuron.v_leak).shape == (128,)
        assert np.asarray(neuron.v_reset).shape == (128,)
        assert np.asarray(neuron.w_in).shape == (128,)
        assert list(np.asarray(neuron.input_type["input"])) == [128]
        assert list(np.asarray(neuron.output_type["output"])) == [128]

        reloaded = _roundtrip(graph, tmp_path / "issue410.nir")
        reloaded_neuron = reloaded.nodes["1"]
        assert np.allclose(
            np.asarray(reloaded_neuron.v_threshold),
            np.asarray(neuron.v_threshold),
        )
        assert np.allclose(
            np.asarray(reloaded_neuron.tau_mem),
            np.asarray(neuron.tau_mem),
        )
        assert np.allclose(
            np.asarray(reloaded_neuron.tau_syn),
            np.asarray(neuron.tau_syn),
        )
        assert np.asarray(reloaded_neuron.v_threshold).shape == (128,)

    def test_scalar_synaptic_width_4(self, tmp_path):
        net = _scalar_synaptic_net(16, 4)
        graph = export_to_nir(net, torch.ones(1, 16), ignore_dims=[0])
        neuron = graph.nodes["1"]
        assert isinstance(neuron, nir.CubaLIF)
        assert np.asarray(neuron.v_threshold).shape == (4,)
        reloaded = _roundtrip(graph, tmp_path / "width4.nir")
        assert np.asarray(reloaded.nodes["1"].v_threshold).shape == (4,)

    def test_scalar_leaky_roundtrip(self, tmp_path):
        net = _scalar_leaky_net(2450, 128)
        graph = export_to_nir(net, torch.ones(1, 2450), ignore_dims=[0])

        neuron = graph.nodes["1"]
        assert isinstance(neuron, nir.LIF)
        assert np.asarray(neuron.v_threshold).shape == (128,)
        assert np.asarray(neuron.tau).shape == (128,)
        assert np.asarray(neuron.r).shape == (128,)
        assert np.asarray(neuron.v_leak).shape == (128,)
        assert np.asarray(neuron.v_reset).shape == (128,)
        assert list(np.asarray(neuron.input_type["input"])) == [128]
        assert list(np.asarray(neuron.output_type["output"])) == [128]

        reloaded = _roundtrip(graph, tmp_path / "scalar_leaky.nir")
        assert np.asarray(reloaded.nodes["1"].v_threshold).shape == (128,)
        assert np.allclose(
            np.asarray(reloaded.nodes["1"].v_threshold),
            np.asarray(neuron.v_threshold),
        )

    def test_vector_params_unchanged(self, tmp_path):
        n = 16
        beta = 0.9 * torch.ones(n)
        thr = torch.ones(n)
        alpha = 0.5 * torch.ones(n)
        net = torch.nn.Sequential(
            torch.nn.Linear(32, n),
            snn.Synaptic(
                alpha=alpha,
                beta=beta,
                threshold=thr,
                init_hidden=True,
                output=True,
            ),
        )
        graph = export_to_nir(net, torch.ones(1, 32), ignore_dims=[0])
        neuron = graph.nodes["1"]
        assert np.asarray(neuron.v_threshold).shape == (n,)
        assert np.allclose(np.asarray(neuron.v_threshold), 1.0)
        reloaded = _roundtrip(graph, tmp_path / "vector.nir")
        assert np.asarray(reloaded.nodes["1"].v_threshold).shape == (n,)
        assert np.allclose(np.asarray(reloaded.nodes["1"].v_threshold), 1.0)

    def test_broadcast_helper_affine_predecessor(self):
        from snntorch.export_nir import (
            _broadcast_scalar_neuron_params_to_width,
        )

        width = 7
        graph = nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type=np.array([5])),
                "affine": nir.Affine(
                    weight=np.zeros((width, 5)),
                    bias=np.zeros(width),
                ),
                "lif": nir.LIF(
                    tau=np.array(2.0),
                    r=np.array(1.0),
                    v_leak=np.array(0.0),
                    v_threshold=np.array(1.0),
                    v_reset=np.array(0.0),
                ),
                "output": nir.Output(output_type=np.array([width])),
            },
            edges=[
                ("input", "affine"),
                ("affine", "lif"),
                ("lif", "output"),
            ],
            type_check=False,
        )
        _broadcast_scalar_neuron_params_to_width(graph)
        lif = graph.nodes["lif"]
        assert np.asarray(lif.v_threshold).shape == (width,)
        assert np.asarray(lif.tau).shape == (width,)
        assert list(np.asarray(lif.input_type["input"])) == [width]
        assert np.allclose(np.asarray(lif.v_threshold), 1.0)

    def test_unresolved_width_raises(self):
        from snntorch.export_nir import (
            _broadcast_scalar_neuron_params_to_width,
        )

        graph = nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type=np.array([1, 8, 8])),
                "pool": nir.AvgPool2d(kernel_size=2, stride=2, padding=(0, 0)),
                "lif": nir.LIF(
                    tau=np.array(2.0),
                    r=np.array(1.0),
                    v_leak=np.array(0.0),
                    v_threshold=np.array(1.0),
                    v_reset=np.array(0.0),
                ),
                "output": nir.Output(output_type=np.array([4, 4])),
            },
            edges=[
                ("input", "pool"),
                ("pool", "lif"),
                ("lif", "output"),
            ],
            type_check=False,
        )
        with pytest.raises(ValueError, match="Cannot infer neuron width"):
            _broadcast_scalar_neuron_params_to_width(graph)

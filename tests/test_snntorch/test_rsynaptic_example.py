"""Execute the RSynaptic documentation example as a regression test."""

from textwrap import dedent

import snntorch as snn
import torch


def test_rsynaptic_documentation_example_runs_and_backpropagates():
    example = snn.RSynaptic.__doc__.split("Example::", 1)[1]
    example = dedent(example.split(":param alpha:", 1)[0])
    namespace = {
        "num_inputs": 3,
        "num_hidden": 5,
        "num_outputs": 2,
        "num_steps": 4,
    }
    exec(example, namespace)
    net = namespace["Net"]()
    data = torch.ones(2, namespace["num_inputs"])

    spikes, membrane = net(data)

    assert spikes.shape == (4, 2, 2)
    assert membrane.shape == (4, 2, 2)
    assert isinstance(net.lif1, snn.RSynaptic)
    assert isinstance(net.lif2, snn.RSynaptic)
    assert "init_rsynaptic" not in example
    membrane.sum().backward()
    assert net.fc1.weight.grad is not None
    assert torch.isfinite(net.fc1.weight.grad).all()

    repeated_spikes, repeated_membrane = net(data)
    torch.testing.assert_close(repeated_spikes, spikes)
    torch.testing.assert_close(repeated_membrane, membrane)

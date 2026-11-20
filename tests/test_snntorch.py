#!/usr/bin/env python

"""Tests for `snntorch` package."""

import pytest
import snntorch as snn
import torch


@pytest.fixture(scope="module")
def input_():
    return torch.Tensor([0.25, 0]).unsqueeze(-1)


def test_fire():
    synaptic = snn.Synaptic(alpha=0.5, beta=0.5)
    input_large = torch.Tensor([synaptic.threshold * 10])
    assert synaptic.fire(input_large)[0] == 1


def test_instances():
    snn.SpikingNeuron.init()
    synaptic = snn.Synaptic(alpha=0.5, beta=0.5)
    alpha = snn.Alpha(alpha=0.5, beta=0.4)
    assert len(snn.SpikingNeuron.instances) == 2
    assert list(snn.SpikingNeuron.instances) == [synaptic, alpha]


def test_init_from_subclass_does_not_shadow_instances():
    # Leaky.init() used to bind a new empty list on Leaky, hiding the shared
    # SpikingNeuron.instances list from Leaky.reset_hidden/detach_hidden.
    try:
        snn.Leaky.init()
        assert "instances" not in vars(snn.Leaky)
        assert snn.Leaky.instances is snn.SpikingNeuron.instances
        assert len(snn.SpikingNeuron.instances) == 0

        lif = snn.Leaky(beta=0.5, init_hidden=True)
        lif(torch.ones(2))
        assert lif.mem.sum() > 0

        snn.Leaky.reset_hidden()
        assert torch.equal(lif.mem, torch.zeros(2))
    finally:
        # do not leak a shadowing attribute into other tests if this fails
        if "instances" in vars(snn.Leaky):
            del snn.Leaky.instances


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is not available"
            ),
        ),
    ],
)
def test_instances_do_not_keep_neurons_alive(device):
    # the registry must not prevent neurons (and the tensors they own) from
    # being garbage collected, see issue #328
    import gc
    import weakref

    snn.SpikingNeuron.init()
    kept = snn.Leaky(beta=0.5, init_hidden=True).to(device)
    dropped = snn.Synaptic(alpha=0.5, beta=0.5).to(device)
    dropped(torch.ones(32, 32, device=device))
    ref = weakref.ref(dropped)
    tensor_ref = weakref.ref(dropped.mem)
    assert len(snn.SpikingNeuron.instances) == 2

    del dropped
    gc.collect()
    assert ref() is None
    assert tensor_ref() is None
    assert list(snn.SpikingNeuron.instances) == [kept]
    assert snn.SpikingNeuron.instances[0] is kept

    # class-level helpers still reach the neurons that are alive
    kept(torch.ones(2, device=device))
    snn.Leaky.detach_hidden()
    snn.Leaky.reset_hidden()
    assert torch.equal(kept.mem, torch.zeros(2, device=device))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)
def test_deleted_neurons_release_cuda_memory():
    import gc
    import weakref

    snn.SpikingNeuron.init()
    gc.collect()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()

    for _ in range(3):
        neuron = snn.Leaky(beta=0.5, init_hidden=True).cuda()
        neuron(torch.ones(1024, 1024, device="cuda"))
        ref = weakref.ref(neuron)
        assert torch.cuda.memory_allocated() > baseline

        del neuron
        # Neurons may contain bound-method cycles; collect them before checking
        # live tensor memory. Cached/reserved CUDA memory need not be released.
        gc.collect()
        torch.cuda.synchronize()
        assert ref() is None
        assert torch.cuda.memory_allocated() == baseline

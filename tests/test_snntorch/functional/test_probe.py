#!/usr/bin/env python

"""Tests for `snntorch.functional.probe`."""

import pytest
import snntorch as snn
import torch
import torch.nn as nn
from snntorch.functional import probe


@pytest.fixture
def net():
    return nn.Sequential(
        nn.Linear(4, 3), snn.Leaky(beta=0.5, init_hidden=True)
    )


@pytest.mark.parametrize("pre_forward", [False, True])
def test_attribute_monitor_pre_and_post_forward(net, pre_forward):
    mon = probe.AttributeMonitor(
        "mem", pre_forward=pre_forward, net=net, instance=snn.Leaky
    )
    x = torch.ones(2, 4)
    net(x)
    mem_after_first_step = net[1].mem.clone()
    net(x)

    assert mon.monitored_layers == ["1"]
    assert len(mon.records) == 2
    assert len(mon["1"]) == 2
    # the second record is taken before (pre_forward) or after the second
    # forward pass, i.e. it does or does not equal the state after step one
    assert torch.equal(mon.records[1], mem_after_first_step) is pre_forward
    mon.remove_hooks()


def test_attribute_monitor_plain_attribute(net):
    # ``reset`` is a plain attribute, not a parameter / buffer / submodule
    mon = probe.AttributeMonitor(
        "reset", pre_forward=False, net=net, instance=snn.Leaky
    )
    net(torch.ones(2, 4))

    assert len(mon.records) == 1
    assert torch.equal(mon.records[0], net[1].reset)
    mon.remove_hooks()

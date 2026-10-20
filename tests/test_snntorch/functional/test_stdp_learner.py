#!/usr/bin/env python

"""Tests for STDPLearner."""

import pytest
import snntorch as snn
import torch
import torch.nn as nn

from snntorch.functional.stdp_learner import STDPLearner


@pytest.fixture
def learner_and_net():
    torch.manual_seed(42)
    fc = nn.Linear(4, 3, bias=False)
    lif = snn.Leaky(beta=0.9, init_hidden=True)
    net = nn.Sequential(fc, lif)
    learner = STDPLearner(fc, lif, tau_pre=2.0, tau_post=2.0)
    return learner, net


def run_steps(net, steps):
    for _ in range(steps):
        net(torch.full((1, 4), 2.0))


class TestSTDPLearner:
    def test_reset_clears_traces_and_records(self, learner_and_net):
        learner, net = learner_and_net
        run_steps(net, 3)
        learner.step(on_grad=False)
        run_steps(net, 2)
        assert learner.trace_pre is not None
        assert learner.trace_post is not None
        assert len(learner.in_spike_monitor.records) == 2
        assert len(learner.out_spike_monitor.records) == 2

        learner.reset()

        assert learner.trace_pre is None
        assert learner.trace_post is None
        assert len(learner.in_spike_monitor.records) == 0
        assert len(learner.out_spike_monitor.records) == 0

    def test_reset_before_any_step(self, learner_and_net):
        learner, _ = learner_and_net
        learner.reset()
        assert learner.trace_pre is None
        assert learner.trace_post is None

    def test_traces_start_from_zero_after_reset(self, learner_and_net):
        learner, net = learner_and_net
        run_steps(net, 3)
        learner.step(on_grad=False)
        learner.reset()

        run_steps(net, 1)
        in_spike = learner.in_spike_monitor.records[0].clone()
        out_spike = learner.out_spike_monitor.records[0].clone()
        learner.step(on_grad=False)

        assert torch.equal(learner.trace_pre, in_spike)
        assert torch.equal(learner.trace_post, out_spike)
